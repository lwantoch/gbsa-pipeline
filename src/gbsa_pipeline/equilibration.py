# Preparing minimization protocol and running

from __future__ import annotations

import BioSimSpace as BSS
import sire


def run_heating(simulation_time: BSS.Types.Time, minimized: sire.System) -> sire.System:

    heating_protocol = BSS.Protocol.Equilibration(
        runtime=simulation_time,
        temperature_start=0 * BSS.Units.Temperature.kelvin,
        temperature_end=300 * BSS.Units.Temperature.kelvin,
        restraint="backbone",
    )
    heating_process = BSS.Process.Gromacs(protocol=heating_protocol, system=minimized)

    heating_process.start()
    heating_process.wait()
    equilibrated = heating_process.getSystem(block=True)

    return equilibrated
