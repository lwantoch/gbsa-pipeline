"""Docking helpers for preparing ligands and running AutoDock Vina.

The docking subpackage is organised into focused modules:

- :mod:`._models` — validated input/output dataclasses and the
  :class:`DockingEngine` Protocol.
- :mod:`._ligand_prep` — SMILES/RDKit → PDBQT preparation and PDBQT → SDF
  export with bond-order repair.
- :mod:`._receptor_prep` — PDB → PDBQT conversion and crystal-water merging.
- :mod:`._crystal_waters` — active-site water selection, post-docking
  validation, and the dual-run orchestrator.
- :mod:`._vina` — :class:`VinaEngine` (subprocess-based Vina wrapper).

Every name that was previously public in ``docking.py`` is re-exported here so
existing imports like ``from gbsa_pipeline.docking import VinaEngine`` continue
to work without change.
"""

from gbsa_pipeline.docking._crystal_waters import (
    dock_with_and_without_crystal_waters,
    select_docking_crystal_waters,
    validate_docked_pose,
)
from gbsa_pipeline.docking._ligand_prep import (
    assign_bond_orders_from_template_mol,
    export_pdbqt_to_sdf,
    prepare_ligand_with_meeko,
)
from gbsa_pipeline.docking._models import (
    DockedPose,
    DockingBox,
    DockingEngine,
    DockingManifest,
    DockingRequest,
    DockingResult,
    DockingValidation,
)
from gbsa_pipeline.docking._receptor_prep import (
    convert_receptor_pdb_to_pdbqt,
    merge_pdb_structures,
)
from gbsa_pipeline.docking._vina import VinaEngine
from gbsa_pipeline.mol_utils import load_first_sdf_molecule, molecule_centroid, remove_hydrogens_copy

__all__ = [
    "DockedPose",
    "DockingBox",
    "DockingEngine",
    "DockingManifest",
    "DockingRequest",
    "DockingResult",
    "DockingValidation",
    "VinaEngine",
    "assign_bond_orders_from_template_mol",
    "convert_receptor_pdb_to_pdbqt",
    "dock_with_and_without_crystal_waters",
    "export_pdbqt_to_sdf",
    "load_first_sdf_molecule",
    "merge_pdb_structures",
    "molecule_centroid",
    "prepare_ligand_with_meeko",
    "remove_hydrogens_copy",
    "select_docking_crystal_waters",
    "validate_docked_pose",
]
