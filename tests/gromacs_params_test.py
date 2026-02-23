from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from gbsa_pipeline.change_defaults import (
    GromacsCustom,
    GromacsParams,
    run_gro_custom,
)
from gbsa_pipeline.change_defaults_enum import CommMode, Integrator


def test_params_canonical_enum() -> None:
    mapping = GromacsParams(
        integrator=Integrator.LEAP_FROG,
        comm_mode=CommMode.ANGULAR,
        nsteps=100,
    ).to_mapping()

    assert mapping["integrator"] == "md"
    assert mapping["comm-mode"] == "Angular"
    assert mapping["nsteps"] == 100


def test_params_to_mdp_outputs_expected_lines() -> None:
    params = GromacsParams(nsteps=12, comm_mode=CommMode.ANGULAR, continuation=True)

    mdp_text = params.to_mdp()

    assert "nsteps = 12" in mdp_text
    assert "comm-mode = Angular" in mdp_text
    assert "continuation = yes" in mdp_text
    assert mdp_text.endswith("\n")


def test_params_defaults_are_applied_when_omitted() -> None:
    mapping = GromacsParams().to_mapping()

    assert mapping["integrator"] == "md"
    assert mapping["comm-mode"] == "Linear"
    assert mapping["nsteps"] == 500
    assert mapping["dt"] == 0.001
    assert mapping["pbc"] == "xyz"
    assert mapping["periodic-molecules"] is False
    assert mapping["rlist"] == 1.221


def test_params_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        GromacsParams(comm_mode=cast("Any", "invalid"))


def test_params_coerces_integral_float_to_int() -> None:
    params = GromacsParams(nsteps=cast("Any", 5.0))

    assert params.nsteps == 5


def test_params_non_integral_float_errors() -> None:
    with pytest.raises(ValidationError):
        GromacsParams(nsteps=cast("Any", 5.5))


def test_gromacs_custom_initialises_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyProtocol:
        def __init__(self, config: str | None = None) -> None:
            if config is None:
                self._config = []
            else:
                self._config = Path(config).read_text(encoding="utf-8").splitlines()
            self._parameters: dict[str, Any] = {}

        def setConfig(self, config: list[str]) -> None:  # noqa: N802
            self._config = list(config)

        def getConfig(self) -> list[str]:  # noqa: N802
            return list(self._config)

    monkeypatch.setattr(
        "gbsa_pipeline.change_defaults.BSS.Protocol.Custom",
        DummyProtocol,
    )

    proto = GromacsCustom()

    assert proto._parameters["nsteps"] == 500
    assert any("nsteps" in line and "500" in line for line in proto._config)


def test_run_gro_custom_applies_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyProtocol:
        def __init__(self, config: str | None = None) -> None:
            if config is None:
                self._config = []
            else:
                self._config = Path(config).read_text(encoding="utf-8").splitlines()
            self._parameters: dict[str, Any] = {}

        def setConfig(self, config: list[str]) -> None:  # noqa: N802
            self._config = list(config)

        def getConfig(self) -> list[str]:  # noqa: N802
            return list(self._config)

    class DummyProcess:
        def __init__(self, system: object, protocol: object) -> None:
            self._system = system
            self.protocol = protocol

        def start(self) -> None:
            return None

        def wait(self) -> None:
            return None

        def getSystem(self, *_args: Any, **_kwargs: Any) -> object:  # noqa: N802
            return self._system

    monkeypatch.setattr(
        "gbsa_pipeline.change_defaults.BSS.Protocol.Custom",
        DummyProtocol,
    )
    monkeypatch.setattr("gbsa_pipeline.change_defaults.BSS.Process.Gromacs", DummyProcess)

    system = object()
    customized, protocol = run_gro_custom(
        parameters=None,
        system=system,
        params={"nsteps": 20},
    )

    assert customized is system
    assert any("nsteps" in line and "20" in line for line in protocol._config)


def test_set_params_accepts_dataclass_and_kwargs() -> None:
    # set_params removed; ensure mapping is stored from params
    proto = GromacsCustom(params=GromacsParams(nsteps=25))

    assert proto._parameters["nsteps"] == 25
    assert proto._parameters["comm-mode"] == "Linear"
