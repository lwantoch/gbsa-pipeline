"""Reading ligand from SDF file, standardizing with MolVS and adding hydrogens with RDKIT."""

from os import PathLike
from pathlib import Path

import sire as sr
from molvs import Standardizer
from rdkit import Chem


def load_ligand_sdf(sdf_path: PathLike | str) -> Chem.Mol:
    """Load the first molecule from an SDF file.

    NOTE: RDKit reads a list of molecules from an SDF file.
    Only the first molecule is processed.
    """
    path = Path(sdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {sdf_path}")

    molecule_list = Chem.SDMolSupplier(str(sdf_path), removeHs=False)

    if len(molecule_list) == 0 or molecule_list[0] is None:
        raise ValueError(f"No valid molecules found in {sdf_path}")

    mol = molecule_list[0]
    return mol


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
