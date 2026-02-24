"""Checking functions from parametrization module."""

from __future__ import annotations

from unittest.mock import MagicMock

import BioSimSpace as BSS
import pytest

from gbsa_pipeline.parametrization import (
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
    assert calls[0][2] == "GRO"
    assert calls[1][2] == "TOP"

    assert len(paths) == 2
    suffixes = {p.suffix for p in paths}
    assert ".gro" in suffixes
    assert ".top" in suffixes
