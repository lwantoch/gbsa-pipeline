"""Molecule utilities: SDF loading, hydrogen removal, centroid computation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from gbsa_pipeline.docking._utils import _require_file

if TYPE_CHECKING:
    from rdkit.Geometry.rdGeometry import Point3D


def load_first_sdf_molecule(path: Path, *, remove_hs: bool = False) -> Chem.Mol:
    """Read the first valid molecule from an SDF file."""
    path = _require_file(Path(path), "SDF file")

    supplier = Chem.SDMolSupplier(str(path), removeHs=remove_hs)
    molecule = supplier[0]

    if molecule is None:
        raise ValueError(f"Could not read first molecule from SDF: {path}")

    return molecule


def remove_hydrogens_copy(molecule: Chem.Mol) -> Chem.Mol:
    """Return a copy of a molecule with hydrogens removed.

    This helper exists because pose and chemistry comparisons are often more
    stable on the heavy-atom graph than on a hydrogen-complete representation.
    The `molecule` parameter is required because callers may want to normalize
    molecules coming from templates, raw exports, or rebuilt structures in the
    same way before comparison.
    We are currently using RDKit's hydrogen removal on a copied molecule so the
    original input object stays unchanged, which is important when the same
    molecule is reused later for export or debugging.
    """
    return Chem.RemoveHs(Chem.Mol(molecule))


def molecule_centroid(
    molecule: Chem.Mol,
    *,
    conf_id: int = -1,
    ignore_hs: bool = False,
) -> Point3D:
    """Compute the geometric centroid of one molecular conformer.

    This helper exists because pose-comparison tests need a compact spatial
    summary without introducing a separate alignment step.
    The `molecule` parameter provides the coordinates, while `conf_id` allows
    callers to select a specific conformer when needed.
    We validate the requested conformer before computing the centroid so callers
    get a clear error when they ask for a conformer that is not present.
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
