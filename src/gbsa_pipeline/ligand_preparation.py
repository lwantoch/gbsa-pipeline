"""Reading ligand from SDF file, standardizing with MolVS and adding hydrogens with RDKIT."""

from os import PathLike
from pathlib import Path

import sire as sr
from molvs import Standardizer
from rdkit import Chem

from gbsa_pipeline.mol_utils import load_first_sdf_molecule


def load_ligand_sdf(sdf_path: PathLike | str) -> Chem.Mol:
    """Load the first molecule from an SDF file.

    NOTE: RDKit reads a list of molecules from an SDF file.
    Only the first molecule is processed.
    """
    return load_first_sdf_molecule(Path(sdf_path), remove_hs=False)


def ligand_standardizer(mol: Chem.Mol) -> Chem.Mol:
    """Standardize a ligand with MolVS and add hydrogen using RDKit."""
    s = Standardizer()
    mol = s.standardize(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    return mol


def ligand_converter(sdf_path: PathLike | str) -> sr.mol:
    """Read a BioSimSpace ligand from SDF after standardization and hydrogenation.

    Standardize with MolVS and hydrogenate with RDKit.
    """
    mol = load_ligand_sdf(sdf_path)
    mol_standard = ligand_standardizer(mol)

    sire_mol = sr.convert.to(mol_standard, "sire")
    return sr.convert.to(sire_mol, "BioSimSpace")
