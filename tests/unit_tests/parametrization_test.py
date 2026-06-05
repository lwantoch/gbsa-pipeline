"""Checking functions from parametrization module."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import BioSimSpace as BSS
import pytest

from gbsa_pipeline.parametrization import (
    _collect_pdb_resnums,
    _pdb_sidechain_names_by_depth,
    _strip_mol2_dipeptide_caps,
    _strip_mol2_or_original,
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


# ---------------------------------------------------------------------------
# _strip_mol2_dipeptide_caps tests
# ---------------------------------------------------------------------------

# Minimal ACE-ALA-NME capped dipeptide in GAFF atom types.
# Three substructures in the SUBSTRUCTURE section cause pmd.load_file() to
# return ResidueTemplateContainer (not Structure) unless structure=True is
# passed — this is the root cause of the bug we fix in the implementation.
_ACE_ALA_NME_MOL2 = """\
@<TRIPOS>MOLECULE
ALA
   22    21     3     0     0
SMALL
RESP Charge


@<TRIPOS>ATOM
      1 C1       -2.0000   0.0000   0.0000 c3        1 ACE      0.1160
      2 C2       -1.0000   0.0000   0.0000 c         1 ACE      0.5970
      3 O3       -1.0000   1.0000   0.0000 o         1 ACE     -0.5680
      4 H4       -2.0000   1.0000   0.0000 h1        1 ACE      0.1010
      5 H5       -2.0000  -1.0000   0.0000 h1        1 ACE      0.1010
      6 H6       -3.0000   0.0000   0.0000 h1        1 ACE      0.1010
      7 N7        0.0000   0.0000   0.0000 n         2 ALA     -0.4160
      8 H8        0.0000   1.0000   0.0000 hn        2 ALA      0.2730
      9 C9        1.0000   0.0000   0.0000 c3        2 ALA      0.0340
     10 H10       1.0000   1.0000   0.0000 h1        2 ALA      0.0820
     11 C11       1.0000   0.0000   1.0000 c3        2 ALA     -0.1820
     12 H12       0.0000   0.0000   1.0000 hc        2 ALA      0.0600
     13 H13       1.0000   1.0000   1.0000 hc        2 ALA      0.0600
     14 H14       2.0000   0.0000   1.0000 hc        2 ALA      0.0600
     15 C15       2.0000   0.0000   0.0000 c         2 ALA      0.5970
     16 O16       2.0000   1.0000   0.0000 o         2 ALA     -0.5680
     17 N17       3.0000   0.0000   0.0000 n         3 NME     -0.4160
     18 H18       3.0000   1.0000   0.0000 hn        3 NME      0.2730
     19 C19       4.0000   0.0000   0.0000 c3        3 NME      0.1160
     20 H20       5.0000   0.0000   0.0000 hc        3 NME      0.0970
     21 H21       4.0000   1.0000   0.0000 hc        3 NME      0.0970
     22 H22       4.0000  -1.0000   0.0000 hc        3 NME      0.0970
@<TRIPOS>BOND
     1     1     2 1
     2     2     3 2
     3     1     4 1
     4     1     5 1
     5     1     6 1
     6     2     7 am
     7     7     8 1
     8     7     9 1
     9     9    10 1
    10     9    11 1
    11    11    12 1
    12    11    13 1
    13    11    14 1
    14     9    15 1
    15    15    16 2
    16    15    17 am
    17    17    18 1
    18    17    19 1
    19    19    20 1
    20    19    21 1
    21    19    22 1
@<TRIPOS>SUBSTRUCTURE
     1 ACE         1 TEMP              0 ****  ****    0 ROOT
     2 ALA         7 TEMP              0 ****  ****    0 ROOT
     3 NME        17 TEMP              0 ****  ****    0 ROOT
"""


def _mol2_atom_names(content: str) -> list[str]:
    """Return atom names from the @<TRIPOS>ATOM section of a mol2 string."""
    in_atom = False
    names: list[str] = []
    for line in content.splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom = False
            continue
        if in_atom and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                names.append(parts[1])
    return names


def test_strip_mol2_dipeptide_caps_removes_ace_nme(tmp_path: Path) -> None:
    """ACE and NME residues are stripped; only ALA atoms remain in the output."""
    mol2 = tmp_path / "ace_ala_nme.mol2"
    out = tmp_path / "stripped.mol2"
    mol2.write_text(_ACE_ALA_NME_MOL2, encoding="utf-8")

    result = _strip_mol2_dipeptide_caps(mol2, out)

    assert result == out
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    names = _mol2_atom_names(content)
    # ACE has 6 atoms, ALA has 10, NME has 6; only ALA should remain
    assert len(names) == 10
    assert "ACE" not in content
    assert "NME" not in content


def test_strip_mol2_dipeptide_caps_renames_backbone(tmp_path: Path) -> None:
    """Backbone atoms in the stripped mol2 carry AMBER ff14SB names N/H/CA/HA/CB/C/O."""
    mol2 = tmp_path / "ace_ala_nme.mol2"
    out = tmp_path / "stripped.mol2"
    mol2.write_text(_ACE_ALA_NME_MOL2, encoding="utf-8")

    _strip_mol2_dipeptide_caps(mol2, out)

    names = set(_mol2_atom_names(out.read_text(encoding="utf-8")))
    assert {"N", "H", "CA", "HA", "CB", "C", "O"}.issubset(names)


def test_strip_mol2_dipeptide_caps_no_ace_raises(tmp_path: Path) -> None:
    """A single-residue mol2 without ACE raises ValueError."""
    mol2 = pathlib.Path("tests/testdata/CM1.mol2")
    out = tmp_path / "stripped.mol2"

    with pytest.raises(ValueError, match="ACE"):
        _strip_mol2_dipeptide_caps(mol2, out)


# ---------------------------------------------------------------------------
# _strip_mol2_or_original tests
# ---------------------------------------------------------------------------


def test_strip_mol2_or_original_returns_stripped_path(tmp_path: Path) -> None:
    """On success, returns work_dir/<stem>_stripped.mol2 with caps removed."""
    mol2 = tmp_path / "ALA.mol2"
    mol2.write_text(_ACE_ALA_NME_MOL2, encoding="utf-8")

    result = _strip_mol2_or_original(mol2, tmp_path)

    assert result == tmp_path / "ALA_stripped.mol2"
    assert result.exists()
    names = set(_mol2_atom_names(result.read_text(encoding="utf-8")))
    assert {"N", "CA", "C", "O"}.issubset(names)


def test_strip_mol2_or_original_returns_original_on_failure(tmp_path: Path) -> None:
    """When stripping fails (no ACE cap), returns the original path unchanged."""
    mol2 = pathlib.Path("tests/testdata/CM1.mol2")

    result = _strip_mol2_or_original(mol2, tmp_path)

    assert result == mol2


def test_strip_mol2_or_original_warns_on_failure(tmp_path: Path) -> None:
    """A UserWarning is emitted when stripping fails."""
    mol2 = pathlib.Path("tests/testdata/CM1.mol2")

    with pytest.warns(UserWarning, match="Cap stripping skipped"):
        _strip_mol2_or_original(mol2, tmp_path)
