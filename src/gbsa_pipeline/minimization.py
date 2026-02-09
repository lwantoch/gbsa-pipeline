# Preparing minimization protocol and running

from __future__ import annotations

import BioSimSpace as BSS
import sire


def run_minimization(nsteps: int, system: sire.System, engine: str = "GROMACS"):

    protocol = BSS.Protocol.Minimisation(steps=nsteps)
    process = BSS.MD.run(
        system,
        protocol,
        engine=engine,
        gpu_support=False,
        auto_start=True,
        name="min",
        property_map={},
    )
    minimized = process.getSystem(block=True)

    return minimized
