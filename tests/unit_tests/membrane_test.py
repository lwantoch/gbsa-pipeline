"""Unit tests for gbsa_pipeline.membrane.

estimate_membrane_geometry() is exercised against a real downloaded MemProtMD
system (tests/testdata/membrane/1py6/, see SOURCE.md there) rather than a
synthetic structure, so the numbers are meaningful DPPC bilayer values, not
just internally self-consistent test fixtures. fetch_memprotmd_system() needs
network access to the real MemProtMD server and is exercised only by the
integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gbsa_pipeline.membrane import estimate_membrane_geometry

TESTDATA = Path(__file__).resolve().parents[1] / "testdata" / "membrane" / "1py6"
STRUCTURE = TESTDATA / "atomistic-system.pdb"


def test_estimate_membrane_geometry_matches_known_dppc_bilayer() -> None:
    """mthick/mctrdz for 1py6's 209-DPPC bilayer match hand-checked values.

    n_phosphates == 209 matches the [molecules] "DPPC 209" count in topol.top.
    mthick (~39.4 A) is a normal phosphate-to-phosphate DPPC bilayer thickness;
    mctrdz falls inside the box's z extent (CRYST1 z = 97.344 A in this file).
    """
    geometry = estimate_membrane_geometry(STRUCTURE, lipid_resnames=["DPP"])

    assert geometry.n_phosphates == 209
    assert 35.0 < geometry.mthick < 45.0
    assert 0.0 < geometry.mctrdz < 97.344


def test_estimate_membrane_geometry_default_resnames_match_pdb_truncated_dppc() -> None:
    """The default lipid_resnames list includes "DPP" (PDB-truncated DPPC).

    PDB's 3-character resname field truncates "DPPC" to "DPP" — confirmed in
    this real file — so the default list must find it without the caller
    having to pass lipid_resnames= explicitly for the common case.
    """
    default_geometry = estimate_membrane_geometry(STRUCTURE)
    explicit_geometry = estimate_membrane_geometry(STRUCTURE, lipid_resnames=["DPP"])

    assert default_geometry == explicit_geometry


def test_estimate_membrane_geometry_pb_params_is_membrane_ready() -> None:
    """MembraneGeometry.pb_params() produces a valid memopt=1 PBParams.

    eneopt/ipb/nfocus/bcopt must all be set to what memopt=1 requires (see
    mmbsa.PBParams.__post_init__ -- ipb/nfocus/bcopt confirmed against real
    sander, which rejects each of their class defaults for a membrane
    calculation with its own explicit "PB Bomb" error), and mthick/mctrdz
    must come from the measurement, not gmx_MMPBSA's plain soluble-protein
    defaults.
    """
    geometry = estimate_membrane_geometry(STRUCTURE, lipid_resnames=["DPP"])

    pb = geometry.pb_params()

    assert pb.memopt == 1
    assert pb.eneopt == 1
    assert pb.ipb == 1
    assert pb.nfocus == 1
    assert pb.bcopt == 10
    assert pb.mthick == geometry.mthick
    assert pb.mctrdz == geometry.mctrdz


def test_estimate_membrane_geometry_pb_params_overrides_take_precedence() -> None:
    """Explicit pb_params() keyword overrides win over the measured defaults."""
    geometry = estimate_membrane_geometry(STRUCTURE, lipid_resnames=["DPP"])

    pb = geometry.pb_params(emem=7.0, mthick=36.0)

    assert pb.emem == 7.0
    assert pb.mthick == 36.0
    assert pb.mctrdz == geometry.mctrdz  # untouched override still uses the measurement


def test_estimate_membrane_geometry_wrong_resname_raises() -> None:
    """A resname that matches nothing produces a clear, actionable error."""
    with pytest.raises(ValueError, match="No phosphate atoms found"):
        estimate_membrane_geometry(STRUCTURE, lipid_resnames=["NOPE"])


def test_estimate_membrane_geometry_rejects_gro_files() -> None:
    """.gro is rejected with a clear conversion hint -- gemmi cannot parse it."""
    with pytest.raises(ValueError, match=r"\.gro"):
        estimate_membrane_geometry(TESTDATA / "not_a_real_file.gro")
