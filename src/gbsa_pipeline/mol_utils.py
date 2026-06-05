"""General RDKit molecule utilities: SDF loading, hydrogen removal, centroid, bond-order repair."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdMolTransforms

if TYPE_CHECKING:
    from pathlib import Path

    from rdkit.Geometry.rdGeometry import Point3D


def load_first_sdf_molecule(path: Path, *, remove_hs: bool = False) -> Chem.Mol:
    """Read the first valid molecule from an SDF file."""
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"SDF file not found: {path}")

    supplier = Chem.SDMolSupplier(str(path), removeHs=remove_hs)
    molecule = supplier[0]

    if molecule is None:
        raise ValueError(f"Could not read first molecule from SDF: {path}")

    return molecule


def remove_hydrogens_copy(molecule: Chem.Mol) -> Chem.Mol:
    """Return a copy of a molecule with hydrogens removed."""
    return Chem.RemoveHs(Chem.Mol(molecule))


def molecule_centroid(
    molecule: Chem.Mol,
    *,
    conf_id: int = -1,
    ignore_hs: bool = False,
) -> Point3D:
    """Compute the geometric centroid of one molecular conformer.

    Raises ``ValueError`` when the molecule has no conformers or the requested
    ``conf_id`` is not present.
    """
    conformer_ids = {conformer.GetId() for conformer in molecule.GetConformers()}

    if not conformer_ids:
        raise ValueError("Molecule has no conformer.")

    if conf_id != -1 and conf_id not in conformer_ids:
        raise ValueError(
            f"Requested conformer id {conf_id} is not present. Available conformer ids: {sorted(conformer_ids)}"
        )

    return rdMolTransforms.ComputeCentroid(
        molecule.GetConformer(conf_id),
        ignoreHs=ignore_hs,
    )


def assign_bond_orders_from_template(
    template_mol: Chem.Mol,
    target_mol: Chem.Mol,
    *,
    add_hydrogens: bool = False,
) -> Chem.Mol:
    """Repair target bond orders from a template while preserving target geometry.

    Strips hydrogens from both molecules, applies
    ``AllChem.AssignBondOrdersFromTemplate``, then infers stereochemistry from
    the existing 3D coordinates.  Optionally re-adds hydrogens with coordinates.

    Raises ``RuntimeError`` when RDKit cannot match the template to the target
    (e.g. different heavy-atom graph).
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
