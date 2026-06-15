"""Receptor PDB → PDBQT conversion checks for every working DEKOIS protein.

These tests protect ``convert_receptor_pdb_to_pdbqt`` against regressions on the
real receptors used in the GBSA screen. Two things are verified per protein:

1. The Meeko receptor-preparation path runs end to end and produces a non-empty
   PDBQT with ATOM records (the same failure class as the 3OLL case, where a
   receptor could not be converted at all).
2. The freshly converted PDBQT is byte-for-byte identical to the receptor PDBQT
   produced by the production docking run (``prepare_inputs`` →
   ``prepare/receptor.pdbqt``). The conversion is deterministic, so any drift —
   a Meeko upgrade, a code change in the prep path — is caught here.

Fixtures live in ``tests/testdata/pdbqt/receptors``:
  * ``<PDB>.pdb``                       — receptor input handed to docking
  * ``<PDB>_receptor_reference.pdbqt``  — the production ("normal calculation")
                                          receptor PDBQT to compare against
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from gbsa_pipeline.docking import convert_receptor_pdb_to_pdbqt

TESTDATA = Path(__file__).parents[1] / "testdata"
RECEPTOR_TESTDATA = TESTDATA / "pdbqt" / "receptors"

# Every protein currently running in the screen end to end.
WORKING_PROTEINS = ["1A5H", "1F0R", "2P1T", "2W8Y", "3FDN", "3L3M", "3LBK", "3K5E", "1UOU"]


def _atom_records(pdbqt_text: str) -> list[str]:
    """Return only ATOM/HETATM coordinate lines (the chemistry-bearing content).

    Comparing these lines ignores any future header/REMARK additions while still
    catching changes to atom typing, partial charges, ordering, or coordinates.
    """
    return [ln for ln in pdbqt_text.splitlines() if ln.startswith(("ATOM", "HETATM"))]


@pytest.mark.integration
@pytest.mark.parametrize("pdb_id", WORKING_PROTEINS)
def test_working_receptor_converts_to_pdbqt(pdb_id: str, tmp_path: Path) -> None:
    """Each working receptor converts to a valid, non-empty PDBQT via Meeko."""
    receptor_pdb = RECEPTOR_TESTDATA / f"{pdb_id}.pdb"
    if not receptor_pdb.exists():
        pytest.skip(f"missing receptor fixture: {receptor_pdb}")

    output = tmp_path / f"{pdb_id}.pdbqt"
    result = convert_receptor_pdb_to_pdbqt(receptor_pdb, output_path=output)

    assert result == output
    assert output.exists()
    atoms = _atom_records(output.read_text(encoding="utf-8"))
    assert atoms, f"{pdb_id}: Meeko produced a PDBQT with no ATOM records"


@pytest.mark.integration
@pytest.mark.parametrize("pdb_id", WORKING_PROTEINS)
def test_receptor_pdbqt_matches_production(pdb_id: str, tmp_path: Path) -> None:
    """Freshly converted receptor PDBQT matches the production docking output.

    Compares ATOM/HETATM records against the ``prepare/receptor.pdbqt`` produced
    by the screen's ``prepare_inputs`` stage. They must be identical because the
    conversion is deterministic and the production call used the same helper.
    """
    receptor_pdb = RECEPTOR_TESTDATA / f"{pdb_id}.pdb"
    reference_pdbqt = RECEPTOR_TESTDATA / f"{pdb_id}_receptor_reference.pdbqt"
    if not receptor_pdb.exists() or not reference_pdbqt.exists():
        pytest.skip(f"missing receptor/reference fixture for {pdb_id}")

    output = tmp_path / f"{pdb_id}.pdbqt"
    convert_receptor_pdb_to_pdbqt(receptor_pdb, output_path=output)

    produced = _atom_records(output.read_text(encoding="utf-8"))
    reference = _atom_records(reference_pdbqt.read_text(encoding="utf-8"))

    if produced != reference:
        diff = "\n".join(
            difflib.unified_diff(
                reference,
                produced,
                fromfile=f"{pdb_id}_production.pdbqt",
                tofile=f"{pdb_id}_test.pdbqt",
                lineterm="",
                n=1,
            )
        )
        pytest.fail(
            f"{pdb_id}: converted receptor PDBQT differs from production "
            f"({len(reference)} vs {len(produced)} atoms).\n{diff[:4000]}"
        )
