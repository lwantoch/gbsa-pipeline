"""Checking functions from parametrization module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import BioSimSpace as BSS
import pytest

from gbsa_pipeline.parametrization import (
    _collect_pdb_resnums,
    _pdb_sidechain_names_by_depth,
    _write_crystal_waters_pdb,
    export_gromacs_top_gro,
    load_protein_pdb,
    parameterise_ligand_gaff2,
    parameterise_protein_amber,
)


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


# ---------------------------------------------------------------------------
# _FakeParamResult helper
# ---------------------------------------------------------------------------


class _FakeParamResult:
    """Simulates BSS parameterisation result which exposes getMolecule()."""

    def __init__(self, mol: object) -> None:
        self._mol = mol

    def getMolecule(self) -> object:  # noqa: N802
        return self._mol


# ---------------------------------------------------------------------------
# parameterise_protein_amber tests
# ---------------------------------------------------------------------------


def test_parameterise_protein_amber_invalid_ff() -> None:
    """Unknown FF string raises ValueError without touching BSS."""
    with pytest.raises(ValueError, match="Unsupported protein FF"):
        parameterise_protein_amber(None, ff="unknown_ff")


@pytest.mark.parametrize(
    ("input_ff", "expected_attr"),
    [
        ("ff14SB", "ff14SB"),
        ("FF14SB", "ff14SB"),
        ("ff19SB", "ff19SB"),
        ("ff99SB", "ff99SB"),
    ],
)
def test_parameterise_protein_amber_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    input_ff: str,
    expected_attr: str,
) -> None:
    """Correct BSS.Parameters.<attr> is called and getMolecule() is unwrapped."""
    fake_mol = MagicMock()
    fake_result = _FakeParamResult(fake_mol)
    mock_fn = MagicMock(return_value=fake_result)

    monkeypatch.setattr(BSS.Parameters, expected_attr, mock_fn)

    protein_mock = MagicMock()
    result = parameterise_protein_amber(protein_mock, ff=input_ff)

    mock_fn.assert_called_once_with(protein_mock)
    assert result is fake_mol


# ---------------------------------------------------------------------------
# parameterise_ligand_gaff2 tests
# ---------------------------------------------------------------------------


def test_parameterise_ligand_gaff2_forwards_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """charge_method and net_charge are forwarded to BSS.Parameters.gaff2."""
    fake_mol = MagicMock()
    fake_result = _FakeParamResult(fake_mol)
    mock_gaff2 = MagicMock(return_value=fake_result)

    monkeypatch.setattr(BSS.Parameters, "gaff2", mock_gaff2)

    ligand_mock = MagicMock()
    result = parameterise_ligand_gaff2(ligand_mock, net_charge=-1, charge_method="RESP")

    mock_gaff2.assert_called_once_with(ligand_mock, net_charge=-1, charge_method="RESP")
    assert result is fake_mol


# ---------------------------------------------------------------------------
# export_gromacs_top_gro tests
# ---------------------------------------------------------------------------


def test_export_gromacs_top_gro(monkeypatch: pytest.MonkeyPatch) -> None:
    """BSS.IO.saveMolecules is called twice; returned paths have .gro/.top extensions."""
    calls: list[tuple[str, object, str]] = []

    def _fake_save(path: str, system: object, fileformat: str) -> None:
        calls.append((path, system, fileformat))

    monkeypatch.setattr(BSS.IO, "saveMolecules", _fake_save)

    system_mock = MagicMock()
    paths = export_gromacs_top_gro(system_mock, prefix="out/complex")

    assert len(calls) == 2
    assert calls[0][2] == "gro87"
    assert calls[1][2] == "grotop"

    assert len(paths) == 2
    suffixes = {p.suffix for p in paths}
    assert ".gro" in suffixes
    assert ".top" in suffixes


# ---------------------------------------------------------------------------
# _write_crystal_waters_pdb tests
# ---------------------------------------------------------------------------

_PROTEIN_WITH_WATERS = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
HETATM    3  O   HOH B   1      10.000  10.000  10.000  1.00  0.00           O
HETATM    4  O   HOH B   2      11.000  11.000  11.000  1.00  0.00           O
END
"""

_PROTEIN_NO_WATERS = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
HETATM    3  ZN  ZN  A 100       5.000   5.000   5.000  1.00  0.00          ZN
END
"""


def test_write_crystal_waters_pdb_extracts_waters(tmp_path: Path) -> None:
    """Only water residues are written; non-water records are excluded."""
    src = tmp_path / "protein.pdb"
    out = tmp_path / "waters.pdb"
    src.write_text(_PROTEIN_WITH_WATERS, encoding="utf-8")

    result = _write_crystal_waters_pdb(src, out)

    assert result == out
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "HOH" in content
    assert "ALA" not in content
    assert "ZN" not in content


def test_write_crystal_waters_pdb_returns_none_when_no_waters(tmp_path: Path) -> None:
    """Returns None and does not create a file when the PDB has no water residues."""
    src = tmp_path / "protein.pdb"
    out = tmp_path / "waters.pdb"
    src.write_text(_PROTEIN_NO_WATERS, encoding="utf-8")

    result = _write_crystal_waters_pdb(src, out)

    assert result is None
    assert not out.exists()


def test_write_crystal_waters_pdb_removes_stale_file(tmp_path: Path) -> None:
    """A previously written waters file is removed when the source has no waters."""
    src = tmp_path / "protein.pdb"
    out = tmp_path / "waters.pdb"
    src.write_text(_PROTEIN_NO_WATERS, encoding="utf-8")
    out.write_text("stale content", encoding="utf-8")

    _write_crystal_waters_pdb(src, out)

    assert not out.exists()


# ---------------------------------------------------------------------------
# _pdb_sidechain_names_by_depth tests
# ---------------------------------------------------------------------------

_GLU_PDB = """\
ATOM      1  N   GLU A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  GLU A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   GLU A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   GLU A   1       4.000   2.000   3.000  1.00  0.00           O
ATOM      5  CB  GLU A   1       2.000   3.000   3.000  1.00  0.00           C
ATOM      6  CG  GLU A   1       2.000   4.000   3.000  1.00  0.00           C
ATOM      7  CD  GLU A   1       2.000   5.000   3.000  1.00  0.00           C
ATOM      8  OE1 GLU A   1       1.000   6.000   3.000  1.00  0.00           O
ATOM      9  OE2 GLU A   1       3.000   6.000   3.000  1.00  0.00           O
END
"""


def test_pdb_sidechain_names_by_depth_excludes_backbone(tmp_path: Path) -> None:
    """Backbone atoms (N, CA, C, O) are not included in the result."""
    src = tmp_path / "glu.pdb"
    src.write_text(_GLU_PDB, encoding="utf-8")

    result = _pdb_sidechain_names_by_depth(src, "GLU")

    all_names = [name for names in result.values() for name in names]
    assert "N" not in all_names
    assert "CA" not in all_names
    assert "C" not in all_names
    assert "O" not in all_names


def test_pdb_sidechain_names_by_depth_correct_depths(tmp_path: Path) -> None:
    """Sidechain carbons are keyed by (element, greek-letter depth)."""
    src = tmp_path / "glu.pdb"
    src.write_text(_GLU_PDB, encoding="utf-8")

    result = _pdb_sidechain_names_by_depth(src, "GLU")

    # CB = beta depth 1, CG = gamma depth 2, CD = delta depth 3
    assert "CB" in result.get(("C", 1), [])
    assert "CG" in result.get(("C", 2), [])
    assert "CD" in result.get(("C", 3), [])
    # OE1 and OE2 = epsilon depth 4
    assert "OE1" in result.get(("O", 4), [])
    assert "OE2" in result.get(("O", 4), [])


# ---------------------------------------------------------------------------
# _collect_pdb_resnums tests
# ---------------------------------------------------------------------------

_MIXED_RESNUM_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   2       2.000   2.000   3.000  1.00  0.00           C
HETATM    3  ZN  ZN  A 100       5.000   5.000   5.000  1.00  0.00          ZN
HETATM    4  ZN  ZN  A 101       6.000   6.000   6.000  1.00  0.00          ZN
END
"""


def test_collect_pdb_resnums_separates_atom_hetatm(tmp_path: Path) -> None:
    """ATOM and HETATM residue numbers are collected into separate sets."""
    pdb = tmp_path / "mixed.pdb"
    pdb.write_text(_MIXED_RESNUM_PDB, encoding="utf-8")

    atom_resnums, hetatm_resnums = _collect_pdb_resnums(pdb)

    assert atom_resnums == {1, 2}
    assert hetatm_resnums == {100, 101}
