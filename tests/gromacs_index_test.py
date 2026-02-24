"""Unit tests for gromacs_index.write_index_from_system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gbsa_pipeline.gromacs_index import write_index_from_system

if TYPE_CHECKING:
    from pathlib import Path


class _FakeMol:
    def __init__(self, n_atoms: int) -> None:
        self._n = n_atoms

    def nAtoms(self) -> int:  # noqa: N802
        return self._n


class _FakeSystem:
    def __init__(self, molecules: list[_FakeMol]) -> None:
        self._mols = molecules

    def getIndex(self, mol: _FakeMol) -> int:  # noqa: N802
        try:
            return self._mols.index(mol)
        except ValueError:
            return -1

    def __iter__(self):
        return iter(self._mols)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_index(path: Path) -> str:
    return path.read_text()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_two_molecule_system(tmp_path: Path) -> None:
    """Protein at idx 0, ligand at idx 1 - correct 1-based atom numbers."""
    protein = _FakeMol(3)
    ligand = _FakeMol(2)
    system = _FakeSystem([protein, ligand])

    out = tmp_path / "test.ndx"
    write_index_from_system(system, protein, ligand, out)

    content = _read_index(out)
    assert "[ Receptor ]" in content
    assert "[ Ligand ]" in content
    # Protein atoms: 1 2 3
    assert "1 2 3" in content
    # Ligand atoms: 4 5 (offset by protein size)
    assert "4 5" in content


def test_three_molecule_system(tmp_path: Path) -> None:
    """Protein + solvent + ligand - only protein and ligand atoms written; offsets correct."""
    protein = _FakeMol(5)
    solvent = _FakeMol(10)
    ligand = _FakeMol(3)
    system = _FakeSystem([protein, solvent, ligand])

    out = tmp_path / "test.ndx"
    write_index_from_system(system, protein, ligand, out)

    content = _read_index(out)
    # Protein atoms: 1-5
    assert "1 2 3 4 5" in content
    # Ligand atoms: 16-18  (5 protein + 10 solvent + 1-based start)
    assert "16 17 18" in content


def test_write_group_line_wrapping(tmp_path: Path) -> None:
    """16 protein atoms - first line has 15 atoms, second has 1."""
    protein = _FakeMol(16)
    ligand = _FakeMol(1)
    system = _FakeSystem([protein, ligand])

    out = tmp_path / "test.ndx"
    write_index_from_system(system, protein, ligand, out)

    content = _read_index(out)
    lines = [ln for ln in content.splitlines() if ln and not ln.startswith("[")]
    # First non-header line should have 15 numbers
    first_line_nums = lines[0].split()
    assert len(first_line_nums) == 15
    # Second line has the 16th atom
    second_line_nums = lines[1].split()
    assert len(second_line_nums) == 1
    assert second_line_nums[0] == "16"


def test_protein_not_in_system(tmp_path: Path) -> None:
    """Protein absent from system raises RuntimeError."""
    protein = _FakeMol(3)
    other = _FakeMol(2)
    ligand = _FakeMol(2)
    system = _FakeSystem([other, ligand])  # protein not included

    with pytest.raises(RuntimeError, match="Protein"):
        write_index_from_system(system, protein, ligand, tmp_path / "test.ndx")


def test_ligand_not_in_system(tmp_path: Path) -> None:
    """Ligand absent from system raises RuntimeError."""
    protein = _FakeMol(3)
    ligand = _FakeMol(2)
    other = _FakeMol(2)
    system = _FakeSystem([protein, other])  # ligand not included

    with pytest.raises(RuntimeError, match="Ligand"):
        write_index_from_system(system, protein, ligand, tmp_path / "test.ndx")
