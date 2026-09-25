"""Unit tests for gromacs_index: atom selection and index-file writing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gbsa_pipeline.gromacs_index import (
    select_receptor_and_ligand_atoms_by_number,
    select_receptor_and_ligand_atoms_by_position,
    write_index,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeResName:
    def __init__(self, name: str) -> None:
        self._name = name

    def value(self) -> str:
        return self._name


class _FakeResidue:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> _FakeResName:
        return _FakeResName(self._name)


class _FakeMol:
    def __init__(self, n_atoms: int, number: int, resname: str = "SOL") -> None:
        self._n = n_atoms
        self._num = number
        self._resname = resname

    def atoms(self) -> range:
        return range(self._n)

    def number(self) -> int:
        return self._num

    def residues(self) -> list[_FakeResidue]:
        return [_FakeResidue(self._resname)]


class _FakeSystem:
    def __init__(self, molecules: list[_FakeMol]) -> None:
        self._mols = molecules

    def __iter__(self):
        return iter(self._mols)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_index(path: Path) -> str:
    return path.read_text()


# ---------------------------------------------------------------------------
# select_receptor_and_ligand_atoms_by_number -- [system] (soluble) convention
# ---------------------------------------------------------------------------


def test_select_by_number_two_molecule_system() -> None:
    """Protein at idx 0, ligand at idx 1 - correct 1-based atom numbers."""
    protein = _FakeMol(3, number=1)
    ligand = _FakeMol(2, number=2)
    system = _FakeSystem([protein, ligand])

    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms_by_number(system, protein, ligand)

    assert receptor_atoms == [1, 2, 3]
    assert ligand_atoms == [4, 5]


def test_select_by_number_three_molecule_system() -> None:
    """Protein + solvent + ligand - only protein and ligand atoms selected; offsets correct."""
    protein = _FakeMol(5, number=1)
    solvent = _FakeMol(10, number=2)
    ligand = _FakeMol(3, number=3)
    system = _FakeSystem([protein, solvent, ligand])

    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms_by_number(system, protein, ligand)

    assert receptor_atoms == [1, 2, 3, 4, 5]
    # 5 protein + 10 solvent + 1-based start
    assert ligand_atoms == [16, 17, 18]


def test_select_by_number_protein_not_in_system() -> None:
    """Protein absent from system - receptor selection comes back empty."""
    protein = _FakeMol(3, number=1)
    other = _FakeMol(2, number=2)
    ligand = _FakeMol(2, number=3)
    system = _FakeSystem([other, ligand])  # protein not included

    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms_by_number(system, protein, ligand)

    assert receptor_atoms == []
    assert ligand_atoms == [3, 4]


def test_select_by_number_ligand_not_in_system() -> None:
    """Ligand absent from system - ligand selection comes back empty."""
    protein = _FakeMol(3, number=1)
    ligand = _FakeMol(2, number=2)
    other = _FakeMol(2, number=3)
    system = _FakeSystem([protein, other])  # ligand not included

    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms_by_number(system, protein, ligand)

    assert receptor_atoms == [1, 2, 3]
    assert ligand_atoms == []


def test_select_by_number_raises_on_unrecognized_solvent_name() -> None:
    """Excluded molecule named like real crystallographic water ('HOH') is rejected.

    gmx_MMPBSA's own cleantop() does not strip "HOH" -- unlike "SOL", "WAT",
    or "TIP3P" -- so a system built from a raw crystal structure (common for
    CHARMM-GUI-style prebuilt membrane systems) would otherwise silently
    produce a Receptor/Ligand index that no longer matches the cleaned
    topology's atom count.
    """
    protein = _FakeMol(3, number=1)
    ligand = _FakeMol(2, number=2)
    water = _FakeMol(1, number=3, resname="HOH")
    system = _FakeSystem([protein, ligand, water])

    with pytest.raises(ValueError, match="HOH"):
        select_receptor_and_ligand_atoms_by_number(system, protein, ligand)


def test_select_by_number_accepts_recognized_solvent_name() -> None:
    """Excluded molecule named "SOL" (gmx_MMPBSA-recognized) passes validation."""
    protein = _FakeMol(3, number=1)
    ligand = _FakeMol(2, number=2)
    water = _FakeMol(1, number=3, resname="SOL")
    system = _FakeSystem([protein, ligand, water])

    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms_by_number(system, protein, ligand)

    assert receptor_atoms == [1, 2, 3]
    assert ligand_atoms == [4, 5]


# ---------------------------------------------------------------------------
# select_receptor_and_ligand_atoms_by_position -- [membrane] convention
# ---------------------------------------------------------------------------


def test_select_by_position_lipids_land_in_receptor() -> None:
    """Protein + multiple lipids + ligand -- lipids must join Receptor, not be dropped.

    gmx_MMPBSA's own topology cleaning strips only water/ions from the
    complex topology, never lipids -- if lipids were excluded here, the
    Receptor+Ligand selection would no longer match that cleaned topology.
    """
    protein = _FakeMol(3, number=1)
    lipid1 = _FakeMol(2, number=2)
    lipid2 = _FakeMol(2, number=3)
    ligand = _FakeMol(2, number=4)
    system = _FakeSystem([protein, lipid1, lipid2, ligand])

    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms_by_position(
        system, n_solute_molecules=3, ligand=ligand
    )

    assert receptor_atoms == [1, 2, 3, 4, 5, 6, 7]
    assert ligand_atoms == [8, 9]


def test_select_by_position_water_and_ions_excluded() -> None:
    """Water/ions placed after the ligand are excluded from both groups."""
    protein = _FakeMol(3, number=1)
    lipid = _FakeMol(2, number=2)
    ligand = _FakeMol(2, number=3)
    water = _FakeMol(3, number=4)
    ion = _FakeMol(1, number=5)
    system = _FakeSystem([protein, lipid, ligand, water, ion])

    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms_by_position(
        system, n_solute_molecules=2, ligand=ligand
    )

    assert receptor_atoms == [1, 2, 3, 4, 5]
    assert ligand_atoms == [6, 7]


def test_select_by_position_ligand_not_in_system() -> None:
    """Ligand absent from system - ligand selection comes back empty."""
    protein = _FakeMol(3, number=1)
    lipid = _FakeMol(2, number=2)
    ligand = _FakeMol(2, number=99)  # not in system
    system = _FakeSystem([protein, lipid])

    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms_by_position(
        system, n_solute_molecules=2, ligand=ligand
    )

    assert receptor_atoms == [1, 2, 3, 4, 5]
    assert ligand_atoms == []


def test_select_by_position_raises_on_unrecognized_solvent_name() -> None:
    """Excluded water named "HOH" (not stripped by gmx_MMPBSA's cleantop()) is rejected."""
    protein = _FakeMol(3, number=1)
    lipid = _FakeMol(2, number=2)
    ligand = _FakeMol(2, number=3)
    water = _FakeMol(1, number=4, resname="HOH")
    system = _FakeSystem([protein, lipid, ligand, water])

    with pytest.raises(ValueError, match="HOH"):
        select_receptor_and_ligand_atoms_by_position(system, n_solute_molecules=2, ligand=ligand)


# ---------------------------------------------------------------------------
# write_index
# ---------------------------------------------------------------------------


def test_write_index_writes_receptor_and_ligand_groups(tmp_path: Path) -> None:
    out = tmp_path / "test.ndx"
    write_index([1, 2, 3], [4, 5], out)

    content = _read_index(out)
    assert "[ Receptor ]" in content
    assert "[ Ligand ]" in content
    assert "1 2 3" in content
    assert "4 5" in content


def test_write_index_line_wrapping(tmp_path: Path) -> None:
    """16 receptor atoms - first line has 15 atoms, second has 1."""
    out = tmp_path / "test.ndx"
    write_index(list(range(1, 17)), [17], out)

    content = _read_index(out)
    lines = [ln for ln in content.splitlines() if ln and not ln.startswith("[")]
    first_line_nums = lines[0].split()
    assert len(first_line_nums) == 15
    second_line_nums = lines[1].split()
    assert len(second_line_nums) == 1
    assert second_line_nums[0] == "16"


def test_write_index_raises_when_receptor_empty(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Protein"):
        write_index([], [1, 2], tmp_path / "test.ndx")


def test_write_index_raises_when_ligand_empty(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Ligand"):
        write_index([1, 2, 3], [], tmp_path / "test.ndx")
