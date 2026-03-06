"""Preparing and running a heating procedure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import BioSimSpace as BSS

if TYPE_CHECKING:
    from pathlib import Path

    import sire


def run_heating(simulation_time: BSS.Types.Time, minimized: sire.System, work_dir: Path | None = None) -> sire.System:
    """Function creates a protocol for NVT Heating and then proceeds with run."""
    heating_protocol = BSS.Protocol.Equilibration(
        runtime=simulation_time,
        temperature_start=0 * BSS.Units.Temperature.kelvin,
        temperature_end=300 * BSS.Units.Temperature.kelvin,
        restraint="backbone",
    )
    kwargs = {"work_dir": str(work_dir)} if work_dir else {}
    heating_process = BSS.Process.Gromacs(protocol=heating_protocol, system=minimized, **kwargs)

    heating_process.start()
    heating_process.wait()
    equilibrated = heating_process.getSystem(block=True)

    return equilibrated
