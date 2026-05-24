"""Integration test for the full GBSA calculation workflow.

This module tests the combined gromacs_index + mmbsa pipeline against
real production MD data.  Unit-style checks for MMPBSAConfig rendering
and gromacs_index atom counting live in tests/unit_tests/, not here.
Test data must be placed in tests/testdata/mmbsa/ before running;
the test is skipped automatically when the files are absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import BioSimSpace as BSS
import pytest

from gbsa_pipeline.gromacs_index import write_index_from_system
from gbsa_pipeline.mmbsa import MMPBSAConfig, run_gmx_mmpbsa_from_gromacs

TESTDATA = Path(__file__).resolve().parents[1] / "testdata" / "mmbsa"

# Production MD outputs to place in TESTDATA before running this test.
COMPLEX_GRO = TESTDATA / "complex.gro"
TRAJ_XTC = TESTDATA / "traj.xtc"
TOPOL_TOP = TESTDATA / "topol.top"


def _load_bss_system(gro: Path, top: Path) -> object:
    """Load a GROMACS system via BioSimSpace and return the BSS system object.

    BioSimSpace.IO.readMolecules is used for consistency with the rest of the
    pipeline.  The BSS system object is returned directly so that callers can
    both iterate over molecules (for write_index_from_system) and save to PDB
    format (required by gmx_MMPBSA for the -cs flag, which does not accept .gro).
    The caller is responsible for selecting the correct protein and ligand
    molecules based on the known molecule ordering of their production system.
    """
    return BSS.IO.readMolecules([str(gro), str(top)], make_whole=True)


def _save_pdb(bss_system: object, path: Path) -> Path:
    """Save a BSS system to PDB and return the written path.

    gmx_MMPBSA requires the complex structure file passed via -cs to be a
    .tpr or .pdb file; .gro is not accepted.  This helper converts the loaded
    BSS system to PDB into a temporary directory so the testdata files are not
    modified.  BSS.IO.saveMolecules writes the basename without extension and
    returns the list of written paths; we return the first (and only) one.
    """
    stem = str(path.with_suffix(""))
    written = BSS.IO.saveMolecules(stem, bss_system, "pdb")
    return Path(written[0])


@pytest.mark.integration
def test_gbsa_full_run(tmp_path: Path) -> None:
    """Run the full GBSA workflow against prebuilt production MD data.

    This test exercises the combined gromacs_index + mmbsa pipeline end to end:
    it loads the production complex, writes a GROMACS index file for the
    Receptor (molecule index 0) and Ligand (last molecule) groups, generates a
    GB-only gmx_MMPBSA input file, and calls gmx_MMPBSA against the production
    trajectory.  The assertion is intentionally coarse — returncode 0 and a
    non-empty output directory — because this is a smoke-level integration test
    for successful execution, not a scientific validation of the binding free
    energy result.  Place complex.gro, traj.xtc, and topol.top in
    tests/testdata/mmbsa/ before running; the test is skipped when any file is
    missing or when gmx_MMPBSA is not available in PATH.
    """
    for f in [COMPLEX_GRO, TRAJ_XTC, TOPOL_TOP]:
        if not f.exists():
            pytest.skip(f"missing testdata: {f}")

    if shutil.which("gmx_MMPBSA") is None:
        pytest.skip("gmx_MMPBSA not available in PATH")

    bss_system = _load_bss_system(COMPLEX_GRO, TOPOL_TOP)
    sire_system = bss_system._sire_object  # type: ignore[attr-defined]

    # Iterate directly over the raw sire system for molecule identification.
    molecules = list(sire_system)

    # Molecule ordering for this system: protein (idx 0, 6645 atoms),
    # ligand UNK (idx 1, 19 atoms), then water and ions.
    protein = molecules[0]
    ligand = molecules[1]

    index_file = tmp_path / "index.ndx"
    write_index_from_system(sire_system, protein, ligand, index_file)
    assert index_file.exists()

    # gmx_MMPBSA requires -cs to be .tpr or .pdb — convert from .gro.
    complex_pdb = _save_pdb(bss_system, tmp_path / "complex.pdb")

    input_file = tmp_path / "mmpbsa.in"
    MMPBSAConfig(pb=None).write(input_file)

    mmpbsa_dir = tmp_path / "mmpbsa_run"
    result = run_gmx_mmpbsa_from_gromacs(
        input_file=input_file,
        complex_structure=complex_pdb,
        trajectory=TRAJ_XTC,
        topology=TOPOL_TOP,
        index_file=index_file,
        receptor_group=0,
        ligand_group=1,
        output_dir=mmpbsa_dir,
    )

    assert result.returncode == 0, f"gmx_MMPBSA failed (rc={result.returncode}):\n{result.stderr[-3000:]}"
    assert any(mmpbsa_dir.iterdir())
