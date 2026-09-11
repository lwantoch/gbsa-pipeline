"""Unit tests for small pipeline.py stage-wiring helpers.

Mocks run_minimization/BSS process construction, matching the style of
tests/unit_tests/md_test.py: these tests check the local wiring between
pipeline stages and the md.py helpers, not GROMACS execution itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

from gbsa_pipeline import pipeline
from gbsa_pipeline.config import MinimizationConfig, RunConfig, SystemConfig

if TYPE_CHECKING:
    import pytest


def _config_with_minimization(**minimization_kwargs: object) -> RunConfig:
    return RunConfig(
        system=SystemConfig(protein=Path("/nonexistent/protein.pdb")),
        minimization=MinimizationConfig(**minimization_kwargs),  # type: ignore[arg-type]
    )


def test_stage_minimize_sd_sets_steep_integrator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_stage_minimize_sd always passes integrator="steep", regardless of config.

    run_minimization applies its params dict as a complete MDP overlay (see
    md._apply_gromacs_params_to_config), so any field not given here reverts
    to GromacsParams' class default -- including integrator, whose default is
    "md" (leapfrog), not "steep". Without this the SD minimization stage
    silently ran plain MD dynamics on the raw, unminimized starting structure
    instead of steepest-descent minimization.
    """
    config = _config_with_minimization(nsteps=123, emtol=45.0)
    system = Mock(name="system")
    stage_dir = tmp_path

    run_minimization = Mock(name="run_minimization", return_value=Mock(name="minimized_system"))
    monkeypatch.setattr(pipeline, "run_minimization", run_minimization)

    pipeline._stage_minimize_sd(config, system, stage_dir)

    run_minimization.assert_called_once()
    _, kwargs = run_minimization.call_args
    assert kwargs["params"]["integrator"] == "steep"
    assert kwargs["params"]["nsteps"] == 123
    assert kwargs["params"]["emtol"] == 45.0


def test_stage_minimize_sd_forwards_define(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_stage_minimize_sd forwards config.minimization.define, e.g. "-DFLEX_SPC".

    Needed for systems whose starting coordinates aren't precise enough for
    rigid SETTLE-constrained water (e.g. MemProtMD's CG2AT-backmapped
    output) -- see MinimizationConfig's docstring.
    """
    config = _config_with_minimization(define="-DFLEX_SPC")
    system = Mock(name="system")
    stage_dir = tmp_path

    run_minimization = Mock(name="run_minimization", return_value=Mock(name="minimized_system"))
    monkeypatch.setattr(pipeline, "run_minimization", run_minimization)

    pipeline._stage_minimize_sd(config, system, stage_dir)

    _, kwargs = run_minimization.call_args
    assert kwargs["params"]["define"] == "-DFLEX_SPC"


def test_stage_minimize_sd_define_defaults_to_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``define`` is still passed (as None) when unset -- matches GromacsParams' own default."""
    config = _config_with_minimization()
    system = Mock(name="system")
    stage_dir = tmp_path

    run_minimization = Mock(name="run_minimization", return_value=Mock(name="minimized_system"))
    monkeypatch.setattr(pipeline, "run_minimization", run_minimization)

    pipeline._stage_minimize_sd(config, system, stage_dir)

    _, kwargs = run_minimization.call_args
    assert kwargs["params"]["define"] is None
