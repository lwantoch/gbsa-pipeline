"""Unit tests for small BioSimSpace MD protocol helpers.

These tests verify the local wiring between the gbsa-pipeline MD helper
functions and the BioSimSpace protocol/process APIs. They deliberately mock
BioSimSpace process construction because unit tests should not launch GROMACS
or require a real molecular system. The assertions focus on the public helper
defaults, optional work-directory handling, process execution, and returned
system object. Detailed MD stability and physical correctness belong in
integration tests with real prepared systems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from gbsa_pipeline import md

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _assert_gromacs_process_call(
    *,
    gromacs_factory: Mock,
    system: Mock,
    protocol: Mock,
    ignore_warnings: bool = True,
    work_dir: Path | str | None = None,
) -> None:
    """Assert that a BioSimSpace GROMACS process was built for a system.

    The production helpers may pass the molecular system either positionally or
    as a ``system=`` keyword, depending on the local wrapper implementation. Both
    forms are valid BioSimSpace process construction patterns, so this helper
    checks the effective call rather than enforcing one spelling. The protocol
    is expected as a keyword because that is the stable BioSimSpace API shape
    used by the helpers. ``work_dir`` is asserted only when explicitly supplied,
    so tests still protect the optional-directory behavior without depending on
    unrelated keyword choices such as warning handling.
    """
    gromacs_factory.assert_called_once()

    args, kwargs = gromacs_factory.call_args

    if args:
        assert args[0] is system
        assert "system" not in kwargs
    else:
        assert kwargs["system"] is system

    assert kwargs["protocol"] is protocol
    assert kwargs.get("ignore_warnings") == ignore_warnings

    if work_dir is None:
        assert "work_dir" not in kwargs
    else:
        assert kwargs["work_dir"] == str(work_dir)


def test_run_minimization_builds_bss_process_and_returns_minimized_system(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that the minimization helper delegates to BioSimSpace correctly.

    The helper should create a BioSimSpace ``Minimisation`` protocol and pass it
    together with the input system to ``BSS.Process.Gromacs``. The optional
    ``work_dir`` should be converted to a string because BioSimSpace process
    constructors commonly expect file-system paths in that form. The current
    helper also passes ``ignore_warnings=True`` by default so staged smoke tests
    can proceed through BioSimSpace's GROMACS wrapper consistently. The returned
    object should be exactly the system returned by ``getSystem(block=True)``.
    """
    input_system = Mock(name="input_system")
    minimized_system = Mock(name="minimized_system")
    minimization_protocol = Mock(name="minimization_protocol")
    process = Mock(name="gromacs_process")
    process.getSystem.return_value = minimized_system

    minimisation_factory = Mock(return_value=minimization_protocol)
    gromacs_factory = Mock(return_value=process)

    monkeypatch.setattr(md.BSS.Protocol, "Minimisation", minimisation_factory)
    monkeypatch.setattr(md.BSS.Process, "Gromacs", gromacs_factory)

    result = md.run_minimization(
        system=input_system,
        work_dir=tmp_path,
    )

    minimisation_factory.assert_called_once_with()
    _assert_gromacs_process_call(
        gromacs_factory=gromacs_factory,
        system=input_system,
        protocol=minimization_protocol,
        ignore_warnings=True,
        work_dir=str(tmp_path),
    )
    process.start.assert_called_once_with()
    process.wait.assert_called_once_with(max_time=None)
    process.getSystem.assert_called_once_with(block=True)
    assert result == minimized_system


def test_run_minimization_omits_work_dir_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the minimization helper does not invent a work directory.

    The ``work_dir`` argument is optional because callers may prefer to let
    BioSimSpace create and manage its own process directory. When no path is
    provided, the helper should not pass ``work_dir=None`` explicitly because
    that can differ from omitting the keyword for third-party APIs. The helper
    still passes the explicit default warning policy so process construction is
    stable across helper functions. The returned minimized system is still taken
    from ``getSystem(block=True)``.
    """
    input_system = Mock(name="input_system")
    minimized_system = Mock(name="minimized_system")
    minimization_protocol = Mock(name="minimization_protocol")
    process = Mock(name="gromacs_process")
    process.getSystem.return_value = minimized_system

    minimisation_factory = Mock(return_value=minimization_protocol)
    gromacs_factory = Mock(return_value=process)

    monkeypatch.setattr(md.BSS.Protocol, "Minimisation", minimisation_factory)
    monkeypatch.setattr(md.BSS.Process, "Gromacs", gromacs_factory)

    result = md.run_minimization(system=input_system)

    minimisation_factory.assert_called_once_with()
    _assert_gromacs_process_call(
        gromacs_factory=gromacs_factory,
        system=input_system,
        protocol=minimization_protocol,
        ignore_warnings=True,
    )
    process.start.assert_called_once_with()
    process.wait.assert_called_once_with(max_time=None)
    process.getSystem.assert_called_once_with(block=True)
    assert result == minimized_system


def test_run_heating_builds_bss_process_and_returns_equilibrated_system(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that the heating helper delegates to BioSimSpace correctly.

    The helper should create a BioSimSpace ``Equilibration`` protocol with the
    requested runtime, a conservative 1 fs timestep, a 50 K start temperature, a
    300 K end temperature, and the current backbone restraint default. The
    optional ``work_dir`` should be converted to a string and passed only as a
    process keyword, matching the minimization helper behavior. The test mocks
    BioSimSpace because unit coverage should verify local protocol wiring, not
    run GROMACS. The returned object should be exactly the system returned by
    ``getSystem(block=True)``.
    """
    minimized_system = Mock(name="minimized_system")
    equilibrated_system = Mock(name="equilibrated_system")
    simulation_time = Mock(name="simulation_time")
    heating_protocol = Mock(name="heating_protocol")
    process = Mock(name="gromacs_process")
    process.getSystem.return_value = equilibrated_system
    process.getConfig.return_value = []

    equilibration_factory = Mock(return_value=heating_protocol)
    gromacs_factory = Mock(return_value=process)

    monkeypatch.setattr(md.BSS.Protocol, "Equilibration", equilibration_factory)
    monkeypatch.setattr(md.BSS.Process, "Gromacs", gromacs_factory)

    result = md.run_heating(
        simulation_time=simulation_time,
        minimized=minimized_system,
        work_dir=tmp_path,
    )

    equilibration_factory.assert_called_once_with(
        timestep=1 * md.BSS.Units.Time.femtosecond,
        runtime=simulation_time,
        temperature_start=50 * md.BSS.Units.Temperature.kelvin,
        temperature_end=300 * md.BSS.Units.Temperature.kelvin,
        restraint="backbone",
    )
    _assert_gromacs_process_call(
        gromacs_factory=gromacs_factory,
        system=minimized_system,
        protocol=heating_protocol,
        ignore_warnings=True,
        work_dir=str(tmp_path),
    )
    process.start.assert_called_once_with()
    process.wait.assert_called_once_with(max_time=None)
    process.getSystem.assert_called_once_with(block=True)
    assert result == equilibrated_system


def test_run_heating_omits_work_dir_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the heating helper does not invent a work directory.

    The ``work_dir`` argument is optional for the same reason as in the
    minimization helper: BioSimSpace can create its own process directory when
    the caller does not need a specific location. When no path is provided, the
    helper should omit the keyword entirely instead of passing ``work_dir=None``.
    The protocol defaults are still asserted because they define the staged MD
    helper contract. This test does not validate physical heating behavior,
    which belongs in an integration test with a real parametrized system.
    """
    minimized_system = Mock(name="minimized_system")
    equilibrated_system = Mock(name="equilibrated_system")
    simulation_time = Mock(name="simulation_time")
    heating_protocol = Mock(name="heating_protocol")
    process = Mock(name="gromacs_process")
    process.getSystem.return_value = equilibrated_system
    process.getConfig.return_value = []

    equilibration_factory = Mock(return_value=heating_protocol)
    gromacs_factory = Mock(return_value=process)

    monkeypatch.setattr(md.BSS.Protocol, "Equilibration", equilibration_factory)
    monkeypatch.setattr(md.BSS.Process, "Gromacs", gromacs_factory)

    result = md.run_heating(
        simulation_time=simulation_time,
        minimized=minimized_system,
    )

    equilibration_factory.assert_called_once_with(
        timestep=1 * md.BSS.Units.Time.femtosecond,
        runtime=simulation_time,
        temperature_start=50 * md.BSS.Units.Temperature.kelvin,
        temperature_end=300 * md.BSS.Units.Temperature.kelvin,
        restraint="backbone",
    )
    _assert_gromacs_process_call(
        gromacs_factory=gromacs_factory,
        system=minimized_system,
        protocol=heating_protocol,
        ignore_warnings=True,
    )
    process.start.assert_called_once_with()
    process.wait.assert_called_once_with(max_time=None)
    process.getSystem.assert_called_once_with(block=True)
    assert result == equilibrated_system


def test_run_npt_equilibration_builds_bss_process_and_returns_equilibrated_system(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that the NPT equilibration helper delegates to BioSimSpace correctly.

    The helper should create a BioSimSpace ``Equilibration`` protocol with the
    requested runtime, a conservative 1 fs timestep, a 300 K target temperature,
    a 1 atm pressure target, and the current conservative backbone restraint
    default. The optional ``work_dir`` should be converted to a string and
    passed only as a process keyword, matching the other MD helper behavior.
    The test mocks BioSimSpace because unit coverage should verify local
    protocol wiring, not pressure equilibration with a real GROMACS process.
    """
    heated_system = Mock(name="heated_system")
    equilibrated_system = Mock(name="equilibrated_system")
    simulation_time = Mock(name="simulation_time")
    equilibration_protocol = Mock(name="equilibration_protocol")
    process = Mock(name="gromacs_process")
    process.getSystem.return_value = equilibrated_system
    process.getConfig.return_value = []

    equilibration_factory = Mock(return_value=equilibration_protocol)
    gromacs_factory = Mock(return_value=process)

    monkeypatch.setattr(md.BSS.Protocol, "Equilibration", equilibration_factory)
    monkeypatch.setattr(md.BSS.Process, "Gromacs", gromacs_factory)

    result = md.run_npt_equilibration(
        simulation_time=simulation_time,
        heated=heated_system,
        work_dir=tmp_path,
    )

    equilibration_factory.assert_called_once_with(
        timestep=1 * md.BSS.Units.Time.femtosecond,
        runtime=simulation_time,
        temperature=300 * md.BSS.Units.Temperature.kelvin,
        pressure=1 * md.BSS.Units.Pressure.atm,
        restraint="backbone",
    )
    _assert_gromacs_process_call(
        gromacs_factory=gromacs_factory,
        system=heated_system,
        protocol=equilibration_protocol,
        ignore_warnings=True,
        work_dir=str(tmp_path),
    )
    process.start.assert_called_once_with()
    process.wait.assert_called_once_with(max_time=None)
    process.getSystem.assert_called_once_with(block=True)
    assert result == equilibrated_system


def test_run_npt_equilibration_omits_work_dir_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the NPT equilibration helper does not invent a work directory.

    The ``work_dir`` argument remains optional so the caller can either choose a
    predictable process directory or let BioSimSpace create one. When no path is
    provided, the helper should omit the keyword entirely instead of passing
    ``work_dir=None``. The unit test still checks the exact protocol defaults
    because those defaults are part of the helper contract. Real NPT stability
    remains the responsibility of integration tests with prepared systems.
    """
    heated_system = Mock(name="heated_system")
    equilibrated_system = Mock(name="equilibrated_system")
    simulation_time = Mock(name="simulation_time")
    equilibration_protocol = Mock(name="equilibration_protocol")
    process = Mock(name="gromacs_process")
    process.getSystem.return_value = equilibrated_system
    process.getConfig.return_value = []

    equilibration_factory = Mock(return_value=equilibration_protocol)
    gromacs_factory = Mock(return_value=process)

    monkeypatch.setattr(md.BSS.Protocol, "Equilibration", equilibration_factory)
    monkeypatch.setattr(md.BSS.Process, "Gromacs", gromacs_factory)

    result = md.run_npt_equilibration(
        simulation_time=simulation_time,
        heated=heated_system,
    )

    equilibration_factory.assert_called_once_with(
        timestep=1 * md.BSS.Units.Time.femtosecond,
        runtime=simulation_time,
        temperature=300 * md.BSS.Units.Temperature.kelvin,
        pressure=1 * md.BSS.Units.Pressure.atm,
        restraint="backbone",
    )
    _assert_gromacs_process_call(
        gromacs_factory=gromacs_factory,
        system=heated_system,
        protocol=equilibration_protocol,
        ignore_warnings=True,
    )
    process.start.assert_called_once_with()
    process.wait.assert_called_once_with(max_time=None)
    process.getSystem.assert_called_once_with(block=True)
    assert result == equilibrated_system


def test_run_production_builds_bss_process_and_returns_production_system(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that the production helper delegates to BioSimSpace correctly.

    The helper should create a BioSimSpace ``Production`` protocol with the
    requested runtime, a 300 K temperature, and 1 atm pressure. The optional
    ``work_dir`` should be converted to a string and passed only as a process
    keyword, matching the other MD helper behavior. The current process wrapper
    also passes ``ignore_warnings=True`` by default for consistency with the
    staged smoke-test helpers. The returned object should be exactly the system
    returned by ``getSystem(block=True)``.
    """
    equilibrated_system = Mock(name="equilibrated_system")
    production_system = Mock(name="production_system")
    simulation_time = Mock(name="simulation_time")
    production_protocol = Mock(name="production_protocol")
    process = Mock(name="gromacs_process")
    process.getSystem.return_value = production_system
    process.getConfig.return_value = []

    production_factory = Mock(return_value=production_protocol)
    gromacs_factory = Mock(return_value=process)

    monkeypatch.setattr(md.BSS.Protocol, "Production", production_factory)
    monkeypatch.setattr(md.BSS.Process, "Gromacs", gromacs_factory)

    result = md.run_production(
        simulation_time=simulation_time,
        equilibrated=equilibrated_system,
        work_dir=tmp_path,
    )

    production_factory.assert_called_once_with(
        runtime=simulation_time,
        temperature=300 * md.BSS.Units.Temperature.kelvin,
        pressure=1 * md.BSS.Units.Pressure.atm,
    )
    _assert_gromacs_process_call(
        gromacs_factory=gromacs_factory,
        system=equilibrated_system,
        protocol=production_protocol,
        ignore_warnings=True,
        work_dir=str(tmp_path),
    )
    process.start.assert_called_once_with()
    process.wait.assert_called_once_with(max_time=None)
    process.getSystem.assert_called_once_with(block=True)
    assert result == production_system


def test_run_production_omits_work_dir_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the production helper does not invent a work directory.

    The ``work_dir`` argument remains optional so the caller can either choose a
    predictable process directory or let BioSimSpace create one. When no path is
    provided, the helper should omit the keyword entirely instead of passing
    ``work_dir=None``. The process warning policy is still explicit because the
    helper always forwards the current default to the BioSimSpace GROMACS
    process. This test checks only local process construction and return
    behavior, keeping real production MD execution out of unit coverage.
    """
    equilibrated_system = Mock(name="equilibrated_system")
    production_system = Mock(name="production_system")
    simulation_time = Mock(name="simulation_time")
    production_protocol = Mock(name="production_protocol")
    process = Mock(name="gromacs_process")
    process.getSystem.return_value = production_system
    process.getConfig.return_value = []

    production_factory = Mock(return_value=production_protocol)
    gromacs_factory = Mock(return_value=process)

    monkeypatch.setattr(md.BSS.Protocol, "Production", production_factory)
    monkeypatch.setattr(md.BSS.Process, "Gromacs", gromacs_factory)

    result = md.run_production(
        simulation_time=simulation_time,
        equilibrated=equilibrated_system,
    )

    production_factory.assert_called_once_with(
        runtime=simulation_time,
        temperature=300 * md.BSS.Units.Temperature.kelvin,
        pressure=1 * md.BSS.Units.Pressure.atm,
    )
    _assert_gromacs_process_call(
        gromacs_factory=gromacs_factory,
        system=equilibrated_system,
        protocol=production_protocol,
        ignore_warnings=True,
    )
    process.start.assert_called_once_with()
    process.wait.assert_called_once_with(max_time=None)
    process.getSystem.assert_called_once_with(block=True)
    assert result == production_system
