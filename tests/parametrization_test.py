"""Checking functions from parametrization module."""

from __future__ import annotations

import BioSimSpace as BSS
import pytest

from src.gbsa_pipeline.parametrization import load_protein_pdb


def test_read_1of1_molecules() -> None:
    """Test reading molecule from file."""
    mol_1 = load_protein_pdb("tests/testdata/test1.pdb")
    system = BSS.IO.readMolecules("tests/testdata/test1.pdb")

    assert system.nAtoms() == mol_1.nAtoms()


def test_read_1of2_molecules() -> None:
    """Test reading molecule from file."""
    mol_1 = load_protein_pdb("tests/testdata/test2.pdb")
    system = BSS.IO.readMolecules("tests/testdata/test2.pdb")

    assert system.nAtoms() != mol_1.nAtoms()


def test_read_empty() -> None:
    """Checking the strange case of giving an empty file."""
    with pytest.raises(OSError):
        load_protein_pdb("tests/testdata/empty.pdb")
