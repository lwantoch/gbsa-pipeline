"""tleap/antechamber helpers and the tleap-based parametrization path."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import gemmi
import parmed as pmd

from gbsa_pipeline.mol2_utils import _strip_mol2_or_original
from gbsa_pipeline.parametrization_enum import LigandFF, ProteinFF
from gbsa_pipeline.parametrization_models import (
    _WATER_RESIDUE_NAMES_SET,
    ParametrisedComplex,
    ParametrizationInput,
    _write_crystal_waters_pdb,
)

logger = logging.getLogger(__name__)

_PROTEIN_FF_LEAPRC: dict[ProteinFF, str] = {
    ProteinFF.FF14SB: "leaprc.protein.ff14SB",
    ProteinFF.FF19SB: "leaprc.protein.ff19SB",
    ProteinFF.FF99SB: "leaprc.protein.ff99SBildn",
}

_LIGAND_FF_LEAPRC: dict[LigandFF, str] = {
    LigandFF.GAFF: "leaprc.gaff",
    LigandFF.GAFF2: "leaprc.gaff2",
}


def _write_dry_protein_pdb(protein_pdb: Path, output_pdb: Path) -> Path:
    """Write a PDB for tleap: strip water and H, adapt heavy-atom names to AMBER.

    Uses gemmi to preserve original PDB residue numbers — critical for MCPB.py
    workflows where tleap bond commands reference specific residue IDs.

    Renames heavy atoms that differ between standard PDB/GROMACS convention and
    AMBER ff14SB (OC1→O, OC2→OXT, ILE CD→CD1). H atoms are stripped so tleap
    re-adds them with correct ff14SB names.

    TER placement: gemmi writes TER after the last polymer (ATOM) residue in each
    chain, before any non-polymer (HETATM) residues such as metals.  This is the
    exact behaviour tleap requires to recognise the C-terminal residue and apply
    the CLYS/CXXX template; without TER, saveAmberParm fails on the OXT atom.
    """
    amber_renames: dict[tuple[str, str], str] = {
        ("OC1", ""): "O",
        ("OC2", ""): "OXT",
        ("CD", "ILE"): "CD1",
    }

    st = gemmi.read_pdb(str(protein_pdb))

    for model in st:
        for chain in model:
            res_indices_to_remove: list[int] = []
            for ri, residue in enumerate(chain):
                if residue.name.upper() in _WATER_RESIDUE_NAMES_SET:
                    res_indices_to_remove.append(ri)
                    continue
                resname = residue.name.upper()
                atom_indices_to_remove: list[int] = []
                for ai, atom in enumerate(residue):
                    if atom.is_hydrogen():
                        atom_indices_to_remove.append(ai)
                    else:
                        new_name = amber_renames.get((atom.name, resname)) or amber_renames.get((atom.name, ""))
                        if new_name:
                            atom.name = new_name
                for ai in reversed(atom_indices_to_remove):
                    del residue[ai]
            for ri in reversed(res_indices_to_remove):
                del chain[ri]

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    opts = gemmi.PdbWriteOptions()
    opts.ter_ignores_type = True
    st.write_pdb(str(output_pdb), opts)
    return output_pdb


def _compute_net_charge_from_sdf(sdf_path: Path) -> int:
    """Return the total formal charge of the first molecule in an SDF file."""
    from rdkit import Chem  # noqa: PLC0415

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"Could not read any molecule from {sdf_path}")
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def _resolve_executable(name: str) -> str:
    """Return an absolute executable path resolved from the active environment."""
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")
    return executable


def _run_antechamber(sdf_path: Path, work_dir: Path, net_charge: int | None) -> tuple[Path, Path]:
    """Run antechamber + parmchk2 on a ligand SDF; return (mol2, frcmod) paths."""
    if net_charge is None:
        net_charge = _compute_net_charge_from_sdf(sdf_path)

    mol2_out = work_dir / "antechamber.mol2"
    frcmod_out = work_dir / "antechamber.frcmod"

    ante_result = subprocess.run(  # noqa: S603
        [
            _resolve_executable("antechamber"),
            "-i",
            str(sdf_path.resolve()),
            "-fi",
            "sdf",
            "-o",
            str(mol2_out),
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-s",
            "2",
            "-nc",
            str(net_charge),
            "-at",
            "gaff2",
        ],
        capture_output=True,
        cwd=str(work_dir),
        text=True,
        check=False,
    )
    if ante_result.returncode != 0 or not mol2_out.exists():
        raise RuntimeError(f"antechamber failed for {sdf_path.name}:\n{ante_result.stderr}\n{ante_result.stdout}")

    parmchk_result = subprocess.run(  # noqa: S603
        [
            _resolve_executable("parmchk2"),
            "-s",
            "2",
            "-i",
            str(mol2_out),
            "-f",
            "mol2",
            "-o",
            str(frcmod_out),
        ],
        capture_output=True,
        cwd=str(work_dir),
        text=True,
        check=False,
    )
    if parmchk_result.returncode != 0 or not frcmod_out.exists():
        raise RuntimeError(f"parmchk2 failed for {sdf_path.name}:\n{parmchk_result.stderr}")

    return mol2_out, frcmod_out


def _write_tleap_script(
    protein_pdb: Path,
    protein_mol2s: list[Path],
    protein_frcmods: list[Path],
    ligand_mol2: Path,
    ligand_frcmod: Path,
    cofactor_mol2s: list[Path],
    cofactor_frcmods: list[Path],
    output_prefix: Path,
    add_atom_types_block: str | None = None,
    bond_commands: list[str] | None = None,
    extra_frcmod_names: list[str] | None = None,
    leaprc_extra_sources: tuple[str, ...] | list[str] | None = None,
    protein_ff: ProteinFF = ProteinFF.FF14SB,
    ligand_ff: LigandFF = LigandFF.GAFF2,
) -> Path:
    """Write a tleap script that combines protein + ligand + cofactors.

    ``add_atom_types_block``, ``bond_commands``, and ``extra_frcmod_names`` are
    populated from an MCPB.py-generated tleap.in when a metal site is present.
    They are otherwise empty and the script degenerates to the standard path.
    The addAtomTypes block must appear before mol2 templates are loaded so that
    custom atom-type symbols (M1, Y1-Yn for any metal / any geometry) are
    recognised when the mol2 files are parsed.  Bond commands are appended after
    combine so that residue numbers still refer to the original PDB numbering.
    ``leaprc_extra_sources`` adds additional leaprc sources (e.g.
    leaprc.phosaa14SB for PTR/SEP/TPO) after the main protein and ligand FFs.
    """
    bond_commands = bond_commands or []
    extra_frcmod_names = extra_frcmod_names or []
    leaprc_extra_sources = leaprc_extra_sources or []

    lines = [
        f"source {_PROTEIN_FF_LEAPRC[protein_ff]}",
        f"source {_LIGAND_FF_LEAPRC[ligand_ff]}",
    ]

    # Additional leaprc files (e.g. phosphorylated amino acids, custom residues).
    for leaprc in leaprc_extra_sources:
        lines.append(f"source {leaprc}")

    # MCPB.py custom atom types (must come before loadMol2 calls).
    if add_atom_types_block:
        lines.append(add_atom_types_block)

    # Standard AMBER frcmods (e.g. frcmod.ions1lm_126_tip3p) from MCPB.py script.
    for fname in extra_frcmod_names:
        lines.append(f"loadAmberParams {fname}")

    # Non-standard protein residue parameters (frcmod then mol2 templates).
    for frcmod in protein_frcmods:
        lines.append(f"loadAmberParams {frcmod.resolve()}")
    for mol2 in protein_mol2s:
        resname = mol2.stem.replace("_stripped", "").upper()
        lines.append(f"{resname} = loadMol2 {mol2.resolve()}")

    # Ligand parameters and coordinates (antechamber mol2 carries docked pose).
    lines.append(f"loadAmberParams {ligand_frcmod.resolve()}")
    lines.append(f"mol_lig = loadMol2 {ligand_mol2.resolve()}")

    # Cofactor parameters.
    for i, (cof_mol2, cof_frcmod) in enumerate(zip(cofactor_mol2s, cofactor_frcmods)):
        lines.append(f"loadAmberParams {cof_frcmod.resolve()}")
        lines.append(f"mol_cof{i} = loadMol2 {cof_mol2.resolve()}")

    # Load protein directly from PDB (handles HETATM non-standard residues and
    # MCPB.py-renamed residues like CS1-4 / ZN1).
    lines.append(f"mol_prot = loadPdb {protein_pdb.resolve()}")

    # Metal-coordination and backbone-reconnection bond commands from MCPB.py.
    # Must be applied to mol_prot BEFORE combine: tleap preserves PDB residue
    # numbers on a loadPdb unit but renumbers sequentially after combine, so
    # bond commands like "bond mol.400.ZN" fail post-combine when the protein
    # has fewer than 400 sequential residues.
    for bond_cmd in bond_commands:
        lines.append(bond_cmd.replace("mol.", "mol_prot."))

    # Combine protein + ligand + cofactors.
    parts = "mol_prot mol_lig"
    for i in range(len(cofactor_mol2s)):
        parts += f" mol_cof{i}"
    lines.append(f"mol = combine {{{parts}}}")

    lines.extend(
        [
            f"saveAmberParm mol {output_prefix}.prmtop {output_prefix}.inpcrd",
            "quit",
        ]
    )

    script_path = output_prefix.parent / "tleap.in"
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def _run_tleap(script_path: Path, work_dir: Path) -> None:
    """Run tleap and raise RuntimeError on failure."""
    result = subprocess.run(  # noqa: S603
        [_resolve_executable("tleap"), "-f", str(script_path)],
        capture_output=True,
        cwd=str(work_dir),
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"tleap failed (exit {result.returncode}):\n{output}")
    if "FATAL" in output.upper():
        raise RuntimeError(f"tleap reported FATAL error:\n{output}")
    logger.debug("tleap completed successfully.")
    if "Warning" in output or "error" in output.lower():
        logger.debug("tleap warnings/non-fatal messages:\n%s", output)


def _parse_mcpb_tleap_in(tleap_in: Path) -> dict:
    """Parse an MCPB.py-generated tleap.in and extract elements for our combined script.

    MCPB.py step 4 generates a complete tleap script for any metal / any geometry.
    This parser extracts the portable parts (addAtomTypes block, bond commands,
    protein PDB path, and standard ion frcmod names) without hardcoding residue
    names, atom types, coordination numbers, or metal identity.  Works for
    3-8-coordinate sites with any transition metal.
    """
    tleap_dir = tleap_in.parent
    add_atom_types_lines: list[str] = []
    bond_commands: list[str] = []
    protein_pdb: Path | None = None
    extra_frcmod_names: list[str] = []

    in_add_atom_types = False

    for line in tleap_in.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        low = stripped.lower()

        if low.startswith("addatomtypes"):
            in_add_atom_types = True
            add_atom_types_lines.append(stripped)
            continue

        if in_add_atom_types:
            add_atom_types_lines.append(stripped)
            if "}" in stripped and "{" not in stripped:
                in_add_atom_types = False
            continue

        # mol = loadPdb X  (the MCPB.py-renamed protein PDB, no ligand)
        if re.match(r"mol\s*=\s*loadpdb\s+", low):
            pdb_name = re.split(r"loadpdb\s+", stripped, flags=re.IGNORECASE, maxsplit=1)[-1].strip()
            candidate = (tleap_dir / pdb_name).resolve()
            if candidate.exists():
                protein_pdb = candidate
            continue

        # bond commands — generic for any coordination geometry
        if low.startswith("bond "):
            bond_commands.append(stripped)
            continue

        # Standard AMBER frcmod files (no leading "/" = resolved from $AMBERHOME)
        if low.startswith("loadamberparams "):
            frcmod_name = stripped.split(None, 1)[1].strip()
            if not frcmod_name.startswith("/") and not (tleap_dir / frcmod_name).exists():
                extra_frcmod_names.append(frcmod_name)
            continue

    return {
        "add_atom_types_block": "\n".join(add_atom_types_lines) if add_atom_types_lines else None,
        "bond_commands": bond_commands,
        "protein_pdb": protein_pdb,
        "extra_frcmod_names": extra_frcmod_names,
    }


def _collect_pdb_resnums(pdb: Path) -> tuple[set[int], set[int]]:
    """Return (atom_resnums, hetatm_resnums) from a PDB file.

    Used to detect residue-number gaps between ATOM and HETATM chains so tleap
    bond commands referencing HETATM residue numbers can be remapped to tleap's
    sequential numbering scheme.
    """
    st = gemmi.read_pdb(str(pdb))
    atom_resnums: set[int] = set()
    hetatm_resnums: set[int] = set()
    for model in st:
        for chain in model:
            for residue in chain:
                rn = residue.seqid.num
                if rn is None:
                    continue
                if residue.het_flag == "A":
                    atom_resnums.add(rn)
                elif residue.het_flag == "H":
                    hetatm_resnums.add(rn)
    return atom_resnums, hetatm_resnums


def _remap_bond_residue_numbers(bond_commands: list[str], hetatm_resnum_map: dict[int, int]) -> list[str]:
    """Remap PDB HETATM residue numbers in bond commands to tleap sequential positions.

    Modern tleap (teLeap) assigns HETATM residues that follow a gap in the PDB
    residue numbering a sequential position immediately after the last ATOM
    residue, not the actual PDB residue number.  E.g. a ZN at PDB residue 400
    in a protein whose last ATOM residue is 191 gets tleap index 192.

    hetatm_resnum_map: {pdb_resnum: tleap_seq_num} for all HETATM residues.
    """
    if not hetatm_resnum_map:
        return bond_commands

    def _repl(m: re.Match) -> str:
        n = int(m.group(1))
        return f".{hetatm_resnum_map.get(n, n)}."

    return [re.sub(r"\.(\d+)\.", _repl, cmd) for cmd in bond_commands]


def _parametrize_tleap(inp: ParametrizationInput) -> ParametrisedComplex:
    """Parametrize using tleap for the protein and antechamber for the ligand.

    Proteins with non-standard residues supplied as mol2+frcmod files (e.g.
    MCPB.py metal sites, RESP-charge non-standard amino acids) cannot be
    reliably handled by OpenMM template matching when those residues appear as
    HETATM records in the PDB. tleap reads the original PDB directly and
    resolves residue templates by name, so HETATM non-standard residues are
    parametrized correctly as long as a matching mol2 template is loaded first.
    """
    work_dir = inp.work_dir or Path(tempfile.mkdtemp(prefix="gbsa_param_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    all_frcmod_files = [p for p in inp.config.extra_ff_files if p.suffix.lower() == ".frcmod"]
    all_mol2_files = [p for p in inp.config.extra_ff_files if p.suffix.lower() == ".mol2"]
    # XML files (e.g. phosaa14SB.xml) are OpenMM-only; tleap uses leaprc instead.

    # When a cofactor SDF has a same-stem mol2+frcmod in extra_ff_files, those
    # pre-computed parameters are used directly (antechamber is skipped for that
    # cofactor). Files not matching any cofactor stem are treated as protein residue
    # templates and passed to _write_tleap_script as protein_mol2s/protein_frcmods.
    _cof_stems = {cof_sdf.stem for cof_sdf in inp.cofactor_sdfs}
    precomp_mol2 = {p.stem: p for p in all_mol2_files if p.stem in _cof_stems}
    precomp_frcmod = {p.stem: p for p in all_frcmod_files if p.stem in _cof_stems}
    frcmod_files = [p for p in all_frcmod_files if p.stem not in _cof_stems]
    mol2_files = [p for p in all_mol2_files if p.stem not in _cof_stems]

    # MCPB.py path: use the MCPB.py-renamed PDB (with metal residues and
    # metal ion already in place) instead of the original protein PDB.
    # The tleap.in provides the addAtomTypes block (generic for any metal /
    # any geometry) and all bond commands (coordination + backbone reconnection).
    if inp.config.mcpb_tleap_in is not None:
        mcpb_info = _parse_mcpb_tleap_in(inp.config.mcpb_tleap_in)
        source_pdb = mcpb_info["protein_pdb"]
        if source_pdb is None:
            raise RuntimeError(f"Could not locate loadpdb line in MCPB.py tleap.in: {inp.config.mcpb_tleap_in}")
        add_atom_types_block: str | None = mcpb_info["add_atom_types_block"]
        bond_commands: list[str] = mcpb_info["bond_commands"]
        extra_frcmod_names: list[str] = mcpb_info["extra_frcmod_names"]
    else:
        source_pdb = inp.protein_pdb
        add_atom_types_block = None
        bond_commands = []
        extra_frcmod_names = []

    # Crystal waters: save for solvate_bss to restore, strip from protein PDB.
    crystal_waters_pdb = _write_crystal_waters_pdb(source_pdb, work_dir / "crystal_waters.pdb")
    dry_pdb = _write_dry_protein_pdb(source_pdb, work_dir / "protein_dry.pdb")

    # Modern tleap (teLeap) assigns HETATM residues that appear after a gap in
    # the PDB residue numbering a sequential position immediately after the last
    # ATOM residue (e.g. ZN at PDB 400 → tleap index 192 when last ATOM is 191).
    # Remap any such residue numbers in the MCPB.py bond commands.
    if bond_commands:
        _atom_resnums, _hetatm_resnums = _collect_pdb_resnums(dry_pdb)
        _hetatm_map: dict[int, int] = {}
        if _atom_resnums and _hetatm_resnums:
            _max_atom = max(_atom_resnums)
            for _i, _pdb_rn in enumerate(sorted(_hetatm_resnums)):
                _hetatm_map[_pdb_rn] = _max_atom + 1 + _i
        bond_commands = _remap_bond_residue_numbers(bond_commands, _hetatm_map)

    # Strip ACE/NME caps from mol2 files prepared as capped dipeptides.
    # MCPB.py mol2s (CS1-4, ZN1) are already stripped residue templates;
    stripped_mol2s = [_strip_mol2_or_original(mol2, work_dir, source_pdb) for mol2 in mol2_files]

    # Parametrize ligand with antechamber + parmchk2 (GAFF2 + AM1-BCC).
    lig_work = work_dir / "ligand"
    lig_work.mkdir(exist_ok=True)
    lig_mol2, lig_frcmod = _run_antechamber(inp.ligand_sdf, lig_work, inp.net_charge)

    # Parametrize cofactors (use pre-computed mol2+frcmod from extra_ff_files when
    # both files are present for a given cofactor stem; fall back to antechamber).
    cof_mol2s: list[Path] = []
    cof_frcmods: list[Path] = []
    for i, cof_sdf in enumerate(inp.cofactor_sdfs):
        pm2 = precomp_mol2.get(cof_sdf.stem)
        pfc = precomp_frcmod.get(cof_sdf.stem)
        if pm2 is not None and pfc is not None:
            cof_mol2s.append(pm2)
            cof_frcmods.append(pfc)
        else:
            cof_work = work_dir / f"cofactor_{i}"
            cof_work.mkdir(exist_ok=True)
            c_mol2, c_frcmod = _run_antechamber(cof_sdf, cof_work, net_charge=None)
            cof_mol2s.append(c_mol2)
            cof_frcmods.append(c_frcmod)

    # Write and run tleap to get combined AMBER topology + coordinates.
    script = _write_tleap_script(
        protein_pdb=dry_pdb,
        protein_mol2s=stripped_mol2s,
        protein_frcmods=frcmod_files,
        ligand_mol2=lig_mol2,
        ligand_frcmod=lig_frcmod,
        cofactor_mol2s=cof_mol2s,
        cofactor_frcmods=cof_frcmods,
        output_prefix=work_dir / "complex",
        add_atom_types_block=add_atom_types_block,
        bond_commands=bond_commands,
        extra_frcmod_names=extra_frcmod_names,
        leaprc_extra_sources=list(inp.config.leaprc_extra_sources),
        protein_ff=inp.config.protein_ff,
        ligand_ff=inp.config.ligand_ff,
    )
    _run_tleap(script, work_dir=work_dir)

    # Convert AMBER prmtop/inpcrd → GROMACS GRO/TOP with ParmEd.
    prmtop = work_dir / "complex.prmtop"
    inpcrd = work_dir / "complex.inpcrd"
    if not prmtop.exists() or not inpcrd.exists():
        raise RuntimeError(
            f"tleap did not produce expected output files in {work_dir}. Check tleap.in and the tleap output."
        )
    struct = pmd.load_file(str(prmtop), str(inpcrd))
    gro_file = work_dir / "complex.gro"
    top_file = work_dir / "complex.top"
    gro_file.unlink(missing_ok=True)
    top_file.unlink(missing_ok=True)
    struct.save(str(top_file), format="gromacs")
    struct.save(str(gro_file))

    return ParametrisedComplex(
        gro_file=gro_file,
        top_file=top_file,
        config=inp.config,
        forcefield=None,
        parmed_structure=struct,
        crystal_waters_pdb=crystal_waters_pdb,
    )
