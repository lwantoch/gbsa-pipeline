from __future__ import annotations
import BioSimSpace as BSS
import pytest
from src.gbsa_pipeline.ligand_preparation import (
    load_ligand_sdf,
    ligand_standardizer,
    ligand_converter,
)


def test_load_ligand_valid() -> None:
    """Test reading molecule from file"""

    mol = load_ligand_sdf("testdata/complex3.sdf")
    assert mol is not None


def test_hydrogens_added() -> None:
    """Test standardization adds correct number of hydrogens"""
    mol = load_ligand_sdf("testdata/complex3.sdf")
    mol = ligand_standardizer(mol)

    n_atoms = mol.GetNumAtoms()
    assert n_atoms == 72


def test_read_empty() -> None:

    with pytest.raises(OSError):
        load_ligand_sdf("testdata/empty.sdf.sdf")


def test_ligand_conversion() -> None:
    """Test conversion of the ligand"""
    mol = ligand_converter("testdata/complex3.sdf")
    assert isinstance(mol, BSS._SireWrappers.Molecule)
