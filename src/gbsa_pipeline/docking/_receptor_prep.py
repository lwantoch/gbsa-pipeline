"""Receptor preparation: PDB → PDBQT conversion and crystal-water merging."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import gemmi
from rdkit import Chem

from gbsa_pipeline.docking._utils import _require_file

LOGGER = logging.getLogger(__name__)


def _strip_hetatm(receptor_pdb: Path, dest: Path) -> Path:
    """Write a copy of receptor_pdb with HETATM records removed.

    Meeko processes ATOM and HETATM identically, so modified residues stored
    as HETATM (e.g. CSD, PTR) that share a residue number with a protein ATOM
    residue cause ``each residue key must have exactly 1 resname`` errors.
    Docking receptors should only contain protein atoms anyway.
    """
    st = gemmi.read_pdb(str(receptor_pdb))
    for model in st:
        for chain in model:
            to_remove = [i for i, res in enumerate(chain) if res.het_flag == "H"]
            for i in reversed(to_remove):
                del chain[i]
    st.write_pdb(str(dest))
    return dest


def _merge_sdfs_into_pdb(pdb: Path, sdfs: list[Path], output: Path) -> Path:
    """Append cofactor SDF atoms to a protein PDB as HETATM records.

    Each SDF is read with RDKit (hydrogens preserved) and its coordinate records
    are appended after the protein ATOM lines so meeko assigns AutoDock atom
    types to protein and cofactor(s) together in one pass.
    """
    protein_lines = [
        line
        for line in pdb.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.startswith(("TER", "END"))
    ]

    cofactor_lines: list[str] = []
    for sdf in sdfs:
        supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
        mol = next(iter(supplier), None)
        if mol is None:
            raise ValueError(f"Could not read cofactor SDF: {sdf}")
        pdb_block = Chem.MolToPDBBlock(mol) or ""
        cofactor_lines.extend(
            line for line in pdb_block.splitlines(keepends=True) if line.startswith(("ATOM", "HETATM"))
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(protein_lines) + "".join(cofactor_lines) + "END\n", encoding="utf-8")
    return output


def _build_polymer(pdb_string: str, set_template: dict[str, str]) -> object:
    """Build a Meeko Polymer from a PDB string, applying set_template overrides."""
    from meeko import (  # noqa: PLC0415
        MoleculePreparation,
        Polymer,
        PolymerCreationError,
        ResidueChemTemplates,
    )

    mk_prep = MoleculePreparation.from_config({})
    templates = ResidueChemTemplates.create_from_defaults()
    try:
        return Polymer.from_pdb_string(pdb_string, templates, mk_prep, set_template, residues_to_delete=[])
    except PolymerCreationError as exc:
        raise RuntimeError(f"Meeko could not build polymer from PDB: {exc}") from exc


def convert_receptor_pdb_to_pdbqt(
    receptor_pdb: Path,
    output_path: Path | None = None,
    *,
    cofactor_sdfs: list[Path] | None = None,
) -> Path:
    """Convert a receptor PDB to rigid receptor PDBQT using the Meeko Python API.

    Receptor hydrogen addition, protonation-state decisions, and structural
    cleanup are expected to happen upstream (e.g. in stack_protein_prep).

    Cross-chain disulfide bonds (CYX) are detected automatically: if Meeko
    raises a paddings RuntimeError, the residues flagged as having excess
    inter-residue bonds are retried as CYX.
    """
    from meeko import PDBQTWriterLegacy  # noqa: PLC0415

    receptor_pdb = _require_file(Path(receptor_pdb), "Receptor PDB")

    if receptor_pdb.suffix.lower() != ".pdb":
        raise ValueError(f"Expected a .pdb receptor input, got: {receptor_pdb}")

    if output_path is None:
        output_path = receptor_pdb.with_suffix(".pdbqt")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip HETATM before Meeko so modified residues (CSD, PTR, …) stored as
    # HETATM with the same residue number as a protein ATOM don't cause errors.
    stripped = _strip_hetatm(receptor_pdb, output_path.parent / f"{receptor_pdb.stem}_protein_only.pdb")

    # Merge cofactors after stripping so meeko assigns AutoDock atom types to
    # both protein and cofactor atoms in one pass.
    if cofactor_sdfs:
        pdb_for_meeko = _merge_sdfs_into_pdb(
            stripped,
            cofactor_sdfs,
            output_path.parent / f"{receptor_pdb.stem}_with_cofactor.pdb",
        )
        LOGGER.info("Merged %d cofactor(s) into receptor PDB for meeko.", len(cofactor_sdfs))
    else:
        pdb_for_meeko = stripped

    pdb_string = pdb_for_meeko.read_text(encoding="utf-8")

    LOGGER.info("Preparing receptor with Meeko Python API: %s → %s", receptor_pdb.name, output_path.name)

    # Capture meeko polymer warnings so we can extract CYX residues if the
    # paddings check fails (cross-chain disulfide bonds not in the CYS template).
    _meeko_warnings: list[str] = []

    class _WarningCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            _meeko_warnings.append(record.getMessage())

    _handler = _WarningCapture()
    _meeko_logger = logging.getLogger("meeko")
    _meeko_logger.addHandler(_handler)

    try:
        polymer = _build_polymer(pdb_string, set_template={})
    except RuntimeError as exc:
        if "paddings" not in str(exc):
            raise
        # Identify the CYS residues with excess inter-residue bonds from logged warnings.
        residues = re.findall(
            r"matched with excess inter-residue bond\(s\): (\S+)",
            "\n".join(_meeko_warnings),
        )
        if not residues:
            raise
        set_template = dict.fromkeys(residues, "CYX")
        LOGGER.warning(
            "Meeko: cross-chain CYS bonds detected (%s), retrying with CYX template.",
            ", ".join(residues),
        )
        polymer = _build_polymer(pdb_string, set_template=set_template)
    finally:
        _meeko_logger.removeHandler(_handler)

    pdbqt_string, _flex = PDBQTWriterLegacy.write_from_polymer(polymer)

    if not pdbqt_string.strip():
        raise RuntimeError(
            f"Meeko produced an empty PDBQT for receptor: {receptor_pdb}\n"
            "Check that the PDB contains valid protein ATOM records."
        )

    output_path.write_text(pdbqt_string, encoding="utf-8")
    LOGGER.info("Meeko receptor PDBQT written: %s", output_path.name)
    return output_path


def merge_pdb_structures(base_pdb: Path, extra_pdb: Path, output_pdb: Path) -> Path:
    """Merge two PDB files by appending all chains from extra_pdb into base_pdb.

    Chain names that conflict with base_pdb are automatically renamed. The
    merged structure is written to output_pdb.
    """
    base_st = gemmi.read_pdb(str(base_pdb))
    extra_st = gemmi.read_pdb(str(extra_pdb))
    for chain in extra_st[0]:
        base_st[0].add_chain(chain, unique_name=True)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    base_st.write_pdb(str(output_pdb))
    return output_pdb
