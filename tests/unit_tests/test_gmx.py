"""Unit tests for the ``gmx`` subprocess wrapper.

grompp and mdrun are mocked at the :func:`subprocess.run` boundary so the
tests exercise :func:`gbsa_pipeline.gmx.run_stage`'s command assembly and
error handling without needing a real GROMACS install.  The handoff
section-6 contract is encoded as test cases: ``-maxwarn 0`` by default,
``grompp -t <cpt>`` on checkpoint handoff, ``-r <gro>`` on restrained
stages, SLURM-aware ``-ntomp``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from gbsa_pipeline import gmx
from gbsa_pipeline.gmx import GromppError, MdrunError, StageOutput, _ntomp, run_stage
from gbsa_pipeline.mdp import GromacsParams
from gbsa_pipeline.mdp_enum import Integrator


class _FakeResult:
    """Minimal subprocess.CompletedProcess substitute for monkeypatching."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# _ntomp
# ---------------------------------------------------------------------------


def test_ntomp_prefers_slurm_cpus_per_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """SLURM_CPUS_PER_TASK wins over OMP_NUM_THREADS when both are set."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    assert _ntomp() == 32


def test_ntomp_falls_back_to_omp_num_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """OMP_NUM_THREADS used when SLURM_CPUS_PER_TASK is absent."""
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert _ntomp() == 8


def test_ntomp_falls_back_to_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither SLURM nor OMP env set → cpu_count."""
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(gmx.os, "cpu_count", lambda: 7)
    assert _ntomp() == 7


# ---------------------------------------------------------------------------
# run_stage error paths
# ---------------------------------------------------------------------------


def _patch_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grompp_result: _FakeResult,
    mdrun_result: _FakeResult | None = None,
    captured: list[list[str]] | None = None,
) -> None:
    """Replace :func:`subprocess.run` with a two-call fake.

    First call = grompp; second call = mdrun.  ``captured`` collects the
    full argv for each call so tests can assert on command-line composition.
    """
    results = iter((grompp_result, mdrun_result or _FakeResult(returncode=0)))

    def _fake_run(cmd: list[str], **kwargs: Any) -> _FakeResult:
        if captured is not None:
            captured.append(list(cmd))
        return next(results)

    monkeypatch.setattr(gmx.subprocess, "run", _fake_run)


def test_run_stage_raises_grompp_error_on_nonzero_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Grompp exit 1 → GromppError with log-tail embedded."""
    _patch_subprocess(
        monkeypatch,
        grompp_result=_FakeResult(returncode=1, stderr="Fatal error: boom"),
    )
    mdp = GromacsParams(integrator=Integrator.STEEP, nsteps=10)
    with pytest.raises(GromppError, match="grompp returned 1"):
        run_stage("sd", mdp, tmp_path / "a.gro", tmp_path / "a.top", tmp_path / "work")


def test_run_stage_raises_mdrun_error_when_gro_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mdrun succeeds but doesn't write gromacs.gro → MdrunError."""
    _patch_subprocess(
        monkeypatch,
        grompp_result=_FakeResult(returncode=0),
        mdrun_result=_FakeResult(returncode=0),
    )
    mdp = GromacsParams(integrator=Integrator.STEEP, nsteps=10)
    with pytest.raises(MdrunError, match="final gro present: False"):
        run_stage("sd", mdp, tmp_path / "a.gro", tmp_path / "a.top", tmp_path / "work")


# ---------------------------------------------------------------------------
# run_stage happy path + command composition
# ---------------------------------------------------------------------------


def _patch_mdrun_writes_gro(
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
    captured: list[list[str]],
) -> None:
    """Fake subprocess that creates gromacs.gro on mdrun so run_stage succeeds."""
    results = iter(
        (
            _FakeResult(returncode=0),
            _FakeResult(returncode=0),
        )
    )

    def _fake_run(cmd: list[str], **kwargs: Any) -> _FakeResult:
        captured.append(list(cmd))
        if cmd[1] == "mdrun":
            (work_dir / "gromacs.gro").write_text("placeholder\n")
        return next(results)

    monkeypatch.setattr(gmx.subprocess, "run", _fake_run)


def test_run_stage_returns_stage_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Happy path: run_stage returns a StageOutput with every path under work_dir."""
    work = tmp_path / "stage"
    captured: list[list[str]] = []
    _patch_mdrun_writes_gro(monkeypatch, work, captured)

    mdp = GromacsParams(integrator=Integrator.STEEP, nsteps=10)
    out = run_stage("sd", mdp, tmp_path / "a.gro", tmp_path / "a.top", work)

    assert isinstance(out, StageOutput)
    assert out.gro == work / "gromacs.gro"
    assert out.cpt == work / "gromacs.cpt"
    assert out.log == work / "gromacs.log"
    assert out.tpr == work / "gromacs.tpr"
    assert out.work_dir == work


def test_run_stage_forwards_default_maxwarn_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default call → grompp sees -maxwarn 0 (opposite of BSS's hardcoded 9999)."""
    work = tmp_path / "stage"
    captured: list[list[str]] = []
    _patch_mdrun_writes_gro(monkeypatch, work, captured)
    run_stage(
        "sd",
        GromacsParams(integrator=Integrator.STEEP, nsteps=10),
        tmp_path / "a.gro",
        tmp_path / "a.top",
        work,
    )
    grompp_cmd = captured[0]
    assert "-maxwarn" in grompp_cmd
    assert grompp_cmd[grompp_cmd.index("-maxwarn") + 1] == "0"


def test_run_stage_forwards_cpt_as_grompp_t(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """cpt=<path> → grompp receives -t <path> (velocity handoff mechanism)."""
    work = tmp_path / "stage"
    captured: list[list[str]] = []
    _patch_mdrun_writes_gro(monkeypatch, work, captured)
    cpt = tmp_path / "prev.cpt"
    cpt.write_text("fake")
    run_stage(
        "npt",
        GromacsParams(integrator=Integrator.LEAP_FROG, nsteps=10),
        tmp_path / "a.gro",
        tmp_path / "a.top",
        work,
        cpt=cpt,
    )
    grompp_cmd = captured[0]
    assert "-t" in grompp_cmd
    assert grompp_cmd[grompp_cmd.index("-t") + 1] == str(cpt)


def test_run_stage_adds_r_flag_when_restrained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """restrained=True → grompp receives -r <gro> (restraint reference)."""
    work = tmp_path / "stage"
    captured: list[list[str]] = []
    _patch_mdrun_writes_gro(monkeypatch, work, captured)
    gro = tmp_path / "a.gro"
    run_stage(
        "heating",
        GromacsParams(integrator=Integrator.LEAP_FROG, nsteps=10),
        gro,
        tmp_path / "a.top",
        work,
        restrained=True,
    )
    grompp_cmd = captured[0]
    assert "-r" in grompp_cmd
    assert grompp_cmd[grompp_cmd.index("-r") + 1] == str(gro)


def test_run_stage_omits_r_flag_when_unrestrained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """restrained=False (default) → no -r flag."""
    work = tmp_path / "stage"
    captured: list[list[str]] = []
    _patch_mdrun_writes_gro(monkeypatch, work, captured)
    run_stage(
        "sd",
        GromacsParams(integrator=Integrator.STEEP, nsteps=10),
        tmp_path / "a.gro",
        tmp_path / "a.top",
        work,
    )
    assert "-r" not in captured[0]


def test_run_stage_writes_mdp_before_grompp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """gromacs.mdp exists on disk before grompp is called, with the resolved params."""
    work = tmp_path / "stage"
    captured: list[list[str]] = []
    seen_mdp_at_grompp: list[bool] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> _FakeResult:
        captured.append(list(cmd))
        if cmd[1] == "grompp":
            seen_mdp_at_grompp.append((work / "gromacs.mdp").exists())
        if cmd[1] == "mdrun":
            (work / "gromacs.gro").write_text("ok")
        return _FakeResult(returncode=0)

    monkeypatch.setattr(gmx.subprocess, "run", _fake_run)
    run_stage(
        "sd",
        GromacsParams(integrator=Integrator.STEEP, nsteps=123),
        tmp_path / "a.gro",
        tmp_path / "a.top",
        work,
    )
    assert seen_mdp_at_grompp == [True]
    mdp_text = (work / "gromacs.mdp").read_text()
    assert "nsteps = 123" in mdp_text
    assert "integrator = steep" in mdp_text
