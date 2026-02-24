"""Testing the ligand_preparation module."""

from __future__ import annotations

import BioSimSpace as BSS
import pytest

from gbsa_pipeline.ligand_preparation import (
    ligand_converter,
    ligand_standardizer,
    load_ligand_sdf,
)


def test_load_ligand_valid() -> None:
    """Test reading molecule from file."""
    mol = load_ligand_sdf("tests/testdata/complex3.sdf")
    assert mol is not None


def test_hydrogens_added() -> None:
    """Test standardization adds correct number of hydrogen."""
    mol = load_ligand_sdf("tests/testdata/complex3.sdf")
    mol = ligand_standardizer(mol)

    n_atoms = mol.GetNumAtoms()
    assert n_atoms == 72


def test_read_empty() -> None:
    """Checking the strange case of giving an empty file."""
    with pytest.raises(FileNotFoundError):
        load_ligand_sdf("tests/testdata/empty.sdf.sdf")


def test_ligand_conversion() -> None:
    """Test conversion of the ligand."""
    mol = ligand_converter("tests/testdata/complex3.sdf")
    assert isinstance(mol, BSS._SireWrappers.Molecule)
