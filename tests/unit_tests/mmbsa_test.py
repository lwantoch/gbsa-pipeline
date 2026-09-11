"""Unit tests for mmbsa.MMPBSAConfig and rendering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gbsa_pipeline.mmbsa import GBParams, GeneralParams, MMPBSAConfig, PBParams

if TYPE_CHECKING:
    from pathlib import Path


def test_config_renders_general_namelist() -> None:
    """MMPBSAConfig.to_text() produces a &general block with expected keys.

    This test checks that the rendered output contains the correct Fortran
    namelist header and a representative key-value pair from GeneralParams.
    The ``startframe`` field is used as a canary because it is always written
    (non-None default) and has a distinctive numeric value.
    We are not checking the full field list here because that would duplicate
    the dataclass definition; the goal is to confirm the rendering pipeline
    works end to end.
    """
    config = MMPBSAConfig(
        general=GeneralParams(startframe=5),
        gb=None,
        pb=None,
    )
    text = config.to_text()

    assert "&general" in text
    assert "startframe" in text
    assert "5" in text
    assert "&gb" not in text
    assert "&pb" not in text


def test_config_gb_only_omits_pb() -> None:
    """Setting pb=None omits the &pb namelist from the rendered output.

    This is the expected path for a GB-only MMPBSA run and is the most common
    use case when PB is not needed.  The test also verifies that &gb is still
    present so the omission of &pb cannot be explained by a silent failure to
    render anything at all.
    We are currently not checking specific GB parameter values; that is covered
    by the rendering test.
    """
    config = MMPBSAConfig(pb=None)
    text = config.to_text()

    assert "&gb" in text
    assert "&pb" not in text


def test_config_pb_only_omits_gb() -> None:
    """Setting gb=None omits the &gb namelist from the rendered output.

    This exercises the PB-only path, which is less common but must work
    correctly for workflows that use only the Poisson-Boltzmann solvation model.
    The test mirrors test_config_gb_only_omits_pb and uses the same logic.
    We are not checking specific PB parameter values here.
    """
    config = MMPBSAConfig(gb=None)
    text = config.to_text()

    assert "&pb" in text
    assert "&gb" not in text


def test_extra_dict_overrides_explicit_field() -> None:
    """Values in ``extra`` override the corresponding explicit dataclass field.

    This verifies the escape-hatch mechanism that allows callers to pass
    any gmx_MMPBSA keyword without subclassing.  The ``startframe`` field is
    overridden via ``extra`` and the test checks that the rendered output
    contains the overriding value rather than the default.
    We are checking only that the override appears in the output; duplicated
    keys in the rendered text are not tested here because gmx_MMPBSA would
    reject them — the correct behaviour is that ``_as_kv`` produces only one
    entry per key.
    """
    config = MMPBSAConfig(
        general=GeneralParams(startframe=1, extra={"startframe": 99}),
        gb=None,
        pb=None,
    )
    text = config.to_text()

    assert "99" in text


def test_other_namelists_are_rendered() -> None:
    """Entries in ``other_namelists`` appear as additional namelist blocks.

    This verifies the open extension point for namelists not modelled as
    dataclasses (e.g. &decomp, &nmode).  The test uses a minimal dict with
    one key-value pair to confirm the rendering path without importing any
    optional namelist constants.
    We are checking only for block presence and key presence; full formatting
    is covered by the rendering helper tests.
    """
    config = MMPBSAConfig(
        gb=None,
        pb=None,
        other_namelists={"decomp": {"idecomp": 1}},
    )
    text = config.to_text()

    assert "&decomp" in text
    assert "idecomp" in text


def test_write_creates_file(tmp_path: Path) -> None:
    """MMPBSAConfig.write() creates the file and returns the resolved path.

    This test verifies the public file-writing interface used by the pipeline
    to generate the gmx_MMPBSA input file before calling the runner.  The
    returned path must equal the input path and the file must exist and be
    non-empty.
    We are not checking the file content here because that is covered by the
    to_text tests above.
    """
    config = MMPBSAConfig(gb=None, pb=None)
    out = tmp_path / "mmpbsa.in"

    returned = config.write(out)

    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_gb_with_membrane_pb_raises() -> None:
    """Requesting GB alongside an implicit-membrane PB config is rejected.

    gmx_MMPBSA silently ignores ``memopt`` for ``&gb`` — the GB models have no
    membrane term — so this combination would compute a GB energy that quietly
    ignores the membrane instead of failing loudly. MMPBSAConfig should reject
    it at construction time rather than let a misleading input file reach
    gmx_MMPBSA.
    """
    with pytest.raises(ValueError, match="membrane"):
        MMPBSAConfig(gb=GBParams(), pb=PBParams(memopt=1, eneopt=1))


def test_pb_only_membrane_config_is_accepted() -> None:
    """A membrane PB config with gb=None constructs and renders memopt.

    This is the supported path for membrane-protein MM/PBSA runs: PB only,
    with the implicit membrane enabled.
    """
    config = MMPBSAConfig(gb=None, pb=PBParams(memopt=1, eneopt=1))
    text = config.to_text()

    assert "&gb" not in text
    assert "memopt" in text


def test_membrane_with_default_eneopt_raises() -> None:
    """PBParams(memopt=1) with the default eneopt=2 is rejected.

    eneopt=2 (charge-view energies, PBParams' default) is unsupported for
    membrane systems per the gmx_MMPBSA docs; the membrane examples all set
    eneopt=1. Since PBParams defaults to eneopt=2, simply turning on memopt
    without also overriding eneopt is a likely mistake worth catching early.
    """
    with pytest.raises(ValueError, match="eneopt"):
        PBParams(memopt=1)


def test_membrane_emem_out_of_bounds_raises() -> None:
    """PBParams(memopt=1) requires indi <= emem < exdi.

    The membrane dielectric constant must sit between the solute and solvent
    dielectrics or gmx_MMPBSA errors; this checks the boundary is enforced
    before the input file is even written.
    """
    with pytest.raises(ValueError, match="emem"):
        PBParams(memopt=1, eneopt=1, emem=100.0)
