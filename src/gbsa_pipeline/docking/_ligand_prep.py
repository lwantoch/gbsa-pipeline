"""Ligand preparation: SMILES/RDKit molecule → PDBQT; PDBQT → SDF with bond-order repair."""

from __future__ import annotations

import logging
from pathlib import Path

from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

from gbsa_pipeline.docking._utils import _extract_pdbqt_string_from_meeko_result, _require_file
from gbsa_pipeline.mol_utils import assign_bond_orders_from_template

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mol_from_smiles(smiles: str, name: str) -> Chem.Mol:
    """Parse SMILES, add hydrogens, embed, and UFF-optimise."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")

    mol.SetProp("_Name", name)
    mol = Chem.AddHs(mol)

    if EmbedMolecule(mol) != 0:
        raise RuntimeError(f"RDKit failed to embed 3D coordinates for ligand: {smiles!r}")

    status = UFFOptimizeMolecule(mol)
    if status not in (0, 1):
        LOGGER.warning("UFF optimisation returned non-standard status %s", status)

    return mol


def _prepare_mol_for_meeko(mol: Chem.Mol, name: str | None) -> Chem.Mol:
    """Validate an RDKit molecule has 3D coords, set name, and add explicit hydrogens."""
    mol = Chem.Mol(mol)

    if mol.GetNumConformers() == 0:
        raise ValueError(
            "Chem.Mol input must contain at least one conformer. "
            "Use a SMILES string input to generate 3D coordinates automatically."
        )

    if name is not None:
        mol.SetProp("_Name", name)
    elif not mol.HasProp("_Name"):
        mol.SetProp("_Name", "LIG")

    return Chem.AddHs(mol, addCoords=True)


def _mol_to_pdbqt_string(mol: Chem.Mol) -> str:
    """Run Meeko on a prepared molecule and return the PDBQT string."""
    mol_setups = MoleculePreparation()(mol)
    if not mol_setups:
        raise RuntimeError("Meeko produced no molecule setups.")
    return _extract_pdbqt_string_from_meeko_result(PDBQTWriterLegacy.write_string(mol_setups[0]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Preserved for backward compatibility — the generic implementation lives in mol_utils.
assign_bond_orders_from_template_mol = assign_bond_orders_from_template


def export_pdbqt_to_sdf(
    pdbqt_path: Path,
    output_sdf: Path,
    *,
    template_mol: Chem.Mol | None = None,
    add_hydrogens_after_template: bool = True,
) -> Path:
    """Export a PDBQT file to SDF, optionally rebuilding bond orders from a template.

    When ``template_mol`` is given every pose is passed through
    :func:`assign_bond_orders_from_template` before writing.
    """
    pdbqt_path = _require_file(Path(pdbqt_path), "PDBQT file")
    output_sdf = Path(output_sdf).resolve()
    output_sdf.parent.mkdir(parents=True, exist_ok=True)

    raw_molecules = [
        mol
        for mol in RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule.from_file(str(pdbqt_path), skip_typing=True))
        if mol is not None
    ]

    if not raw_molecules:
        raise RuntimeError(f"Meeko could not reconstruct any molecules from docking output.\nPDBQT: {pdbqt_path}")

    def _to_output(raw_mol: Chem.Mol) -> Chem.Mol:
        if template_mol is None:
            return raw_mol
        repaired = assign_bond_orders_from_template(template_mol, raw_mol, add_hydrogens=add_hydrogens_after_template)
        if raw_mol.HasProp("_Name"):
            repaired.SetProp("_Name", raw_mol.GetProp("_Name"))
        return repaired

    writer = Chem.SDWriter(str(output_sdf))
    try:
        for mol in map(_to_output, raw_molecules):
            writer.write(mol)
    finally:
        writer.close()

    if not output_sdf.exists():
        raise RuntimeError(f"SDF output missing after write.\nExpected: {output_sdf}")

    return output_sdf


def prepare_ligand_with_meeko(
    ligand: str | Chem.Mol,
    output_path: Path,
    name: str | None = None,
) -> Path:
    """Prepare a SMILES string or RDKit molecule as a ligand PDBQT file.

    SMILES input generates 3D coordinates via RDKit embedding and UFF
    optimisation.  ``Chem.Mol`` input must already carry a conformer.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preparing ligand with Meeko: %s", output_path.name)

    if isinstance(ligand, str):
        mol = _mol_from_smiles(ligand, name or "LIG")
    elif isinstance(ligand, Chem.Mol):
        mol = _prepare_mol_for_meeko(ligand, name)
    else:
        raise TypeError("prepare_ligand_with_meeko accepts only SMILES strings or RDKit Chem.Mol objects.")

    pdbqt_string = _mol_to_pdbqt_string(mol)
    if not pdbqt_string.strip():
        raise RuntimeError("Generated ligand PDBQT string is empty.")

    output_path.write_text(pdbqt_string, encoding="utf-8")
    LOGGER.info("Ligand PDBQT written: %s", output_path.name)

    return output_path
