"""BioSimSpace solvation: BSS.Solvent.tip3p (wraps gmx solvate + gmx genion).

BSS writes a fully self-consistent GROMACS topology that BSS/Sire can load back
for MD.  The previous gmx-direct approach (gmx solvate + manual topology
injection) produced topologies that BSS/Sire rejected with "There are no
molecule groups called 'all'".

Crystallographic waters carried on the parametrized complex
(``crystal_waters_pdb``) are normalized to AMBER TIP3P naming, parametrized as
TIP3P (``BSS.Parameters.ff14SB(..., water_model="tip3p")``), and merged into the
system BEFORE bulk solvation, so BSS.Solvent fills bulk water + ions around
them.  The whole path stays in AMBER/GROMACS — no OpenMM.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gbsa_pipeline.solvation_box import SolvatedComplex, SolvationParams, WaterModel, run_solvation

if TYPE_CHECKING:
    from gbsa_pipeline.parametrization import ParametrisedComplex

logger = logging.getLogger(__name__)


def solvate_bss(
    parametrized: ParametrisedComplex,
    params: SolvationParams,
    output_gro: Path,
    output_top: Path,
) -> SolvatedComplex:
    """Solvate a parametrized complex via BSS.Solvent.

    Loads the dry GROMACS complex into BSS, calls the water/ion placement
    via BSS.Solvent (which internally runs gmx solvate and gmx genion), and
    saves the result as GROMACS GRO/TOP files that BSS/Sire can load for MD.

    Parameters
    ----------
    parametrized:
        Dry protein-ligand complex from :func:`~gbsa_pipeline.parametrization.parametrize`.
    params:
        Solvation box parameters (water model, padding, ion concentration, ...).
    output_gro:
        Path for the solvated GROMACS coordinate file.
    output_top:
        Path for the solvated GROMACS topology file.

    Returns:
    -------
    SolvatedComplex
        Dataclass holding paths to the written GROMACS files.
    """
    import BioSimSpace as BSS  # noqa: PLC0415

    work_dir = output_gro.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    system = BSS.IO.readMolecules([str(parametrized.gro_file), str(parametrized.top_file)])
    logger.debug(
        "Dry complex loaded: %d molecules, %d atoms.",
        system.nMolecules(),
        system.nAtoms(),
    )

    # Re-insert crystallographic waters BEFORE bulk solvation (AMBER/GROMACS
    # path, no OpenMM). BSS.Solvent then fills bulk water + ions around them.
    if parametrized.crystal_waters_pdb is not None and parametrized.crystal_waters_pdb.exists():
        het_mols = _load_crystal_het(
            parametrized.crystal_waters_pdb, work_dir, complex_gro=parametrized.gro_file
        )
        if het_mols:
            n_before = system.nMolecules()
            for mol in het_mols:
                system = system + mol
            logger.info(
                "Re-inserted crystal HET (waters + structural ions) before solvation: +%d molecules.",
                system.nMolecules() - n_before,
            )

    bss_work = work_dir / "_bss_solvate"
    bss_work.mkdir(exist_ok=True)
    solvated = run_solvation(system, params, work_dir=bss_work)
    logger.debug(
        "Solvated system: %d molecules, %d atoms.",
        solvated.nMolecules(),
        solvated.nAtoms(),
    )

    # BSS.IO.saveMolecules appends .gro / .top to the given prefix.
    prefix = str(output_gro.with_suffix(""))
    BSS.IO.saveMolecules(prefix, solvated, ["gro87", "grotop"])

    return SolvatedComplex(gro_file=output_gro, top_file=output_top)


_WATER_RESNAMES = {"HOH", "WAT", "SOL", "TIP3", "T3P"}
_AMBER_WATER_ATOM = {
    "OW": " O  ", "HW1": " H1 ", "HW2": " H2 ",
    "O": " O  ", "H1": " H1 ", "H2": " H2 ",
}

# Monatomic crystallographic ions recognised by AMBER's atomic_ions.lib. Kept
# (resname unchanged) through the crystal-HET parametrization. The standard
# ``leaprc.water.tip3p`` (sourced by BSS.Parameters.ff14SB(water_model="tip3p"))
# already loads atomic_ions.lib + Joung-Cheatham monovalent + Li-Merz 12-6
# divalent ion parameters, so structural ions (Ca2+/Zn2+/Mg2+/Na+/...) are
# parametrized automatically with no extra frcmod load.
_ION_RESNAMES = {
    "CA", "MG", "ZN", "MN", "FE", "FE2", "FE3", "CU", "CU1", "CO", "NI", "CD",
    "HG", "BA", "SR", "LI", "RB", "CS", "NA", "K", "CL", "BR", "IOD",
}


def _guess_amberhome() -> str:
    """Best-effort AMBERHOME from the location of ``tleap`` (conda/pixi env prefix)."""
    tleap = shutil.which("tleap")
    if tleap:
        return str(Path(tleap).resolve().parent.parent)
    return os.environ.get("CONDA_PREFIX", "")


def _normalize_crystal_het(src: Path, dst: Path) -> tuple[int, int]:
    """Rewrite crystallographic waters (-> AMBER TIP3P) and KEEP monatomic ions.

    DEKOIS ``*_WAT.pdb`` files store waters with GROMACS atom names
    (OW/HW1/HW2) and sometimes pack two waters into a single residue, both of
    which make tleap (and therefore ``BSS.Parameters``) fail with
    "atom ... does not have a type".  Each water is rewritten as a single
    ``WAT`` residue with AMBER atom names (O/H1/H2).  Monatomic ions
    (``_ION_RESNAMES``, e.g. a structural Ca2+) are written through unchanged
    (their resname matches AMBER's atomic_ions.lib) so they are parametrized
    alongside the waters.  Coordinate columns are preserved byte-for-byte.
    Returns ``(n_waters, n_ions)``.
    """
    resid = n_waters = n_ions = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if line[:6] not in ("ATOM  ", "HETATM"):
                continue
            res = line[17:20].strip()
            if res in _WATER_RESNAMES:
                atom_name = _AMBER_WATER_ATOM.get(line[12:16].strip())
                if atom_name is None:
                    continue
                if atom_name == " O  ":  # an oxygen starts a new water
                    resid += 1
                    n_waters += 1
                fout.write(line[:12] + atom_name + line[16:17] + "WAT" + line[20:22] + f"{resid:>4}" + line[26:])
            elif res in _ION_RESNAMES:
                resid += 1
                n_ions += 1
                # keep atom name + resname; only renumber the residue sequence
                fout.write(line[:22] + f"{resid:>4}" + line[26:])
        fout.write("TER\nEND\n")
    return n_waters, n_ions


def _gro_heavy_atom_coords(gro: Path) -> list[tuple[float, float, float]]:
    """Heavy-atom (non-H) coordinates from a GRO file, in Angstrom."""
    coords: list[tuple[float, float, float]] = []
    lines = gro.read_text().splitlines()
    try:
        natoms = int(lines[1])
    except (IndexError, ValueError):
        return coords
    for line in lines[2 : 2 + natoms]:
        name = line[10:15].strip()
        # element guess: first alphabetic char of the atom name
        sym = next((c for c in name if c.isalpha()), "")
        if sym.upper() == "H":
            continue
        try:
            x = float(line[20:28]) * 10.0
            y = float(line[28:36]) * 10.0
            z = float(line[36:44]) * 10.0
        except ValueError:
            continue
        coords.append((x, y, z))
    return coords


def _filter_clashing_het(
    norm_pdb: Path, complex_gro: Path, cutoff_a: float = 2.0, keep_shell_a: float = 5.0
) -> tuple[int, int]:
    """Keep only crystal-HET residues in the modeled unit's hydration shell.

    Crystal waters/ions are re-inserted from the apo/holo structure, which both
    (a) did not contain the docked ligand and (b) for oligomers contains waters
    belonging to the OTHER chain(s) / bulk region we don't model. Two failure
    modes follow:

    * A water that overlaps the docked complex (< ``cutoff_a``) gives a
      near-zero interatomic distance and ~infinite LJ energy that crashes
      minimization at step 0.
    * An ORPHAN water far from the modeled unit (> ``keep_shell_a`` from any
      complex atom) sits out in what becomes bulk solvent, where ``gmx
      solvate``/``genion`` can place a counter-ion at the same spot -> a
      coincident ion/water pair that also crashes minimization.

    So keep a HET residue only when its closest atom is within
    ``[cutoff_a, keep_shell_a]`` of a complex heavy atom (the real hydration
    shell). Rewrites ``norm_pdb`` in place; returns ``(n_clashing, n_orphan)``.
    Requires numpy; if the complex coords can't be read, nothing is dropped.
    """
    try:
        import numpy as np  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return (0, 0)

    complex_coords = _gro_heavy_atom_coords(complex_gro)
    if not complex_coords:
        logger.warning("Could not read complex coords from %s; skipping HET clash filter.", complex_gro)
        return (0, 0)
    comp = np.asarray(complex_coords, dtype=float)

    # Group HET atom lines by residue (resSeq), keeping original line order.
    # The normalized PDB contains only ATOM/HETATM lines plus a trailing
    # TER/END, which we regenerate; non-atom lines are ignored.
    residues: dict[str, list[str]] = {}
    order: list[str] = []
    for line in norm_pdb.read_text().splitlines(keepends=True):
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        key = line[22:26]
        if key not in residues:
            residues[key] = []
            order.append(key)
        residues[key].append(line)

    kept: list[str] = []
    n_clash = n_orphan = 0
    cutoff2 = cutoff_a * cutoff_a
    shell2 = keep_shell_a * keep_shell_a
    for key in order:
        atoms = residues[key]
        xyz = np.asarray(
            [(float(a[30:38]), float(a[38:46]), float(a[46:54])) for a in atoms], dtype=float
        )
        # min squared distance from any residue atom to any complex heavy atom
        d2 = ((xyz[:, None, :] - comp[None, :, :]) ** 2).sum(-1).min()
        if d2 < cutoff2:
            n_clash += 1
        elif d2 > shell2:
            n_orphan += 1
        else:
            kept.extend(atoms)

    n_dropped = n_clash + n_orphan
    if n_dropped:
        norm_pdb.write_text("".join(kept) + "TER\nEND\n")
        logger.info(
            "Crystal-HET filter: dropped %d clashing (<%.1f A) + %d orphan (>%.1f A from complex); "
            "kept %d of %d residues.",
            n_clash, cutoff_a, n_orphan, keep_shell_a, len(order) - n_dropped, len(order),
        )
    return (n_clash, n_orphan)


def _load_crystal_het(pdb: Path, work_dir: Path, complex_gro: Path | None = None) -> list[Any]:
    """Parametrize crystal waters (TIP3P) and monatomic ions (Li-Merz 12-6-4).

    Normalizes the input (waters -> AMBER ``WAT``; monatomic ions kept) then
    runs ``BSS.Parameters.ff14SB(..., water_model="tip3p", ensure_compatible=False)``
    — the openbiosim crystal-water tutorial path (tleap-backed, no OpenMM).
    ``water_model="tip3p"`` sources ``leaprc.water.tip3p``, which already loads
    atomic_ions.lib + Joung-Cheatham monovalent + Li-Merz 12-6 divalent ion
    parameters, so structural ions (Ca2+/Zn2+/Mg2+/Na+/...) are parametrized
    automatically. Returns the list of parametrized BSS molecules (empty if
    none found). ``BSS.Parameters`` needs ``AMBERHOME``; if unset it is derived
    from tleap.
    """
    import BioSimSpace as BSS  # noqa: PLC0415

    if "AMBERHOME" not in os.environ:
        amberhome = _guess_amberhome()
        if amberhome:
            os.environ["AMBERHOME"] = amberhome

    norm = work_dir / "crystal_het.pdb"
    n_waters, n_ions = _normalize_crystal_het(pdb, norm)
    if n_waters == 0 and n_ions == 0:
        logger.warning("No crystal waters/ions found in %s; skipping re-insertion.", pdb)
        return []

    # Drop any crystal water/ion that overlaps the docked protein+ligand complex
    # (the ligand was absent in the crystal, so some waters land on top of it →
    # ~infinite LJ energy that crashes minimization).
    if complex_gro is not None and Path(complex_gro).exists():
        _filter_clashing_het(norm, Path(complex_gro), cutoff_a=2.0)

    loaded = BSS.IO.readMolecules([str(norm)])
    parametrized: list[Any] = []
    for i in range(loaded.nMolecules()):
        parametrized.append(
            BSS.Parameters.ff14SB(
                loaded[i], water_model="tip3p", ensure_compatible=False,
                work_dir=str(work_dir),
            ).getMolecule()
        )
    logger.info(
        "Parametrized crystal HET: %d waters (TIP3P) + %d ions (JC monovalent / Li-Merz 12-6 divalent).",
        n_waters, n_ions,
    )
    return parametrized


# ---------------------------------------------------------------------------
# Legacy BSS wrapper
# ---------------------------------------------------------------------------


def solvate_parametrized_complex(
    parametrized: ParametrisedComplex,
    *,
    shell_nm: float = 1.0,
    water_model: WaterModel | str = WaterModel.TIP3P,
    work_dir: Path | None = None,
) -> Any:
    """Deprecated — use solvate_bss instead."""
    import BioSimSpace as BSS  # noqa: PLC0415

    logger.warning("solvate_parametrized_complex is deprecated; use solvate_bss instead.")
    dry_system = BSS.IO.readMolecules([str(parametrized.gro_file), str(parametrized.top_file)])
    solvent_fn = getattr(BSS.Solvent, WaterModel(water_model).value)
    kwargs: dict[str, Any] = {
        "ion_conc": 0,
        "is_neutral": False,
        "shell": shell_nm * BSS.Units.Length.nanometer,
    }
    if work_dir is not None:
        kwargs["work_dir"] = str(work_dir)
    return solvent_fn(dry_system, **kwargs)
