"""Ligand preparation: SMILES/RDKit molecule → PDBQT; PDBQT → SDF with bond-order repair."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

from gbsa_pipeline.docking._utils import _extract_pdbqt_string_from_meeko_result, _require_file

LOGGER = logging.getLogger(__name__)


def assign_bond_orders_from_template_mol(
    template_mol: Chem.Mol,
    target_mol: Chem.Mol,
    *,
    add_hydrogens: bool = False,
) -> Chem.Mol:
    """Repair target bond orders from a template while preserving target geometry.

    This is the central chemistry-restoration function in the module, and it is
    intentionally separate from raw export so chemistry and coordinates can be
    reasoned about independently.
    The `template_mol` parameter is required because it is the trusted source of
    bond orders, while `target_mol` is required because it carries the docked
    pose geometry that should remain attached to the repaired molecule.
    We are currently relying on RDKit's `AssignBondOrdersFromTemplate()` to
    repair bond orders directly on the docked heavy-atom graph, then inferring
    stereochemistry from the existing 3D coordinates and optionally re-adding
    hydrogens afterward.

    Reference:
    https://www.rdkit.org/docs/source/rdkit.Chem.AllChem.html
    """
    template_no_h = Chem.RemoveHs(Chem.Mol(template_mol))
    target_no_h = Chem.RemoveHs(Chem.Mol(target_mol))

    try:
        rebuilt = AllChem.AssignBondOrdersFromTemplate(template_no_h, target_no_h)
    except Exception as exc:
        raise RuntimeError(
            "RDKit AssignBondOrdersFromTemplate failed. Template and target molecule likely do not match."
        ) from exc

    rdmolops.AssignStereochemistryFrom3D(rebuilt)

    if add_hydrogens:
        rebuilt = Chem.AddHs(rebuilt, addCoords=True)

    return rebuilt


def export_pdbqt_to_sdf(
    pdbqt_path: Path,
    output_sdf: Path,
    *,
    template_mol: Chem.Mol | None = None,
    add_hydrogens_after_template: bool = True,
) -> Path:
    """Export a PDBQT file to SDF, optionally rebuilding bond orders from a template.

    This function combines two related but distinct steps: raw PDBQT-to-SDF
    export and optional chemistry reconstruction from a trusted template.
    The `pdbqt_path` and `output_sdf` parameters define the file conversion,
    while `template_mol` determines whether bond-order repair should be applied
    and `add_hydrogens_after_template` controls hydrogen re-addition afterward.
    We are currently using the presence of `template_mol` as the single switch
    for template-based repair, because a separate boolean flag would add a
    redundant and potentially contradictory state to this API.
    """
    pdbqt_path = _require_file(Path(pdbqt_path), "PDBQT file")
    output_sdf = Path(output_sdf).resolve()
    output_sdf.parent.mkdir(parents=True, exist_ok=True)

    pdbqt_molecule = PDBQTMolecule.from_file(str(pdbqt_path), skip_typing=True)
    raw_molecules = RDKitMolCreate.from_pdbqt_mol(pdbqt_molecule)
    valid_raw_molecules = [mol for mol in raw_molecules if mol is not None]

    if not valid_raw_molecules:
        raise RuntimeError(f"Meeko could not reconstruct any molecules from docking output.\nPDBQT: {pdbqt_path}")

    output_molecules: list[Chem.Mol] = []

    for raw_mol in valid_raw_molecules:
        output_mol: Any = raw_mol

        if template_mol is not None:
            output_mol = assign_bond_orders_from_template_mol(
                template_mol=template_mol,
                target_mol=raw_mol,
                add_hydrogens=add_hydrogens_after_template,
            )

            if raw_mol.HasProp("_Name"):
                output_mol.SetProp("_Name", raw_mol.GetProp("_Name"))

        output_molecules.append(output_mol)

    writer = Chem.SDWriter(str(output_sdf))
    try:
        for output_mol in output_molecules:
            writer.write(output_mol)
    finally:
        writer.close()

    if not output_sdf.exists():
        raise RuntimeError(f"Meeko reported success but expected SDF output is missing.\nExpected: {output_sdf}")

    return output_sdf


def prepare_ligand_with_meeko(
    ligand: str | Chem.Mol,
    output_path: Path,
    name: str | None = None,
) -> Path:
    """Prepare a SMILES string or RDKit molecule as ligand PDBQT.

    This helper exists so ligand preparation is handled in one place and the
    rest of the module can assume prepared PDBQT ligand inputs when needed.
    The `ligand` parameter accepts either a SMILES string or an RDKit molecule
    because those are the two explicit input types chosen for this minimal API.
    Meeko expects an RDKit molecule with explicit hydrogens and 3D coordinates,
    so this function prepares those prerequisites before calling the current
    documented Meeko Python API.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preparing ligand with Meeko: %s", output_path.name)

    if isinstance(ligand, str):
        mol = Chem.MolFromSmiles(ligand)
        if mol is None:
            raise ValueError("Failed to parse SMILES.")

        mol.SetProp("_Name", name or "LIG")
        mol = Chem.AddHs(mol)

        embed_status = EmbedMolecule(mol)
        if embed_status != 0:
            raise RuntimeError(f"RDKit failed to embed 3D coordinates for ligand: {ligand}")

        optimize_status = UFFOptimizeMolecule(mol)
        if optimize_status not in (0, 1):
            LOGGER.warning(
                "UFF optimization returned non-standard status %s",
                optimize_status,
            )

    elif isinstance(ligand, Chem.Mol):
        mol = Chem.Mol(ligand)

        if mol.GetNumConformers() == 0:
            raise ValueError(
                "Chem.Mol input must contain at least one conformer. "
                "Use a SMILES string input to generate 3D coordinates automatically."
            )

        if name is not None:
            mol.SetProp("_Name", name)
        elif not mol.HasProp("_Name"):
            mol.SetProp("_Name", "LIG")

        mol = Chem.AddHs(mol, addCoords=True)

    else:
        raise TypeError("prepare_ligand_with_meeko supports only SMILES strings and RDKit Chem.Mol objects.")

    preparator = MoleculePreparation()
    mol_setups = preparator(mol)

    if not mol_setups:
        raise RuntimeError("Meeko produced no molecule setups.")

    meeko_result = PDBQTWriterLegacy.write_string(mol_setups[0])
    pdbqt_string = _extract_pdbqt_string_from_meeko_result(meeko_result)

    if not pdbqt_string.strip():
        raise RuntimeError("Generated ligand PDBQT string is empty.")

    output_path.write_text(pdbqt_string, encoding="utf-8")
    LOGGER.info("Ligand PDBQT written: %s", output_path.name)

    return output_path
