"""NVT heating — legacy compatibility shim.

.. deprecated::
    Use :func:`gbsa_pipeline.md.run_heating` directly.
    This module remains to avoid breaking existing callers.  New code should
    call ``md.run_heating`` and pass explicit temperature and restraint
    arguments.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import BioSimSpace as BSS

if TYPE_CHECKING:
    from pathlib import Path

    import sire


def run_heating(
    simulation_time: BSS.Types.Time,
    minimized: sire.System,
    work_dir: Path | None = None,
) -> sire.System:
    """Run NVT heating from 0 K to 300 K with backbone restraints.

    .. deprecated::
        Use :func:`gbsa_pipeline.md.run_heating` with explicit
        ``temperature_start`` and ``temperature_end`` arguments instead.

    Args:
        simulation_time: Total heating runtime as a BSS ``Time`` quantity.
        minimized: Minimized BioSimSpace/Sire system.
        work_dir: Optional working directory for GROMACS files.

    Returns:
        Equilibrated BioSimSpace/Sire system.
    """
    warnings.warn(
        "gbsa_pipeline.equilibration.run_heating is deprecated. "
        "Use gbsa_pipeline.md.run_heating with explicit temperature_start and temperature_end instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from gbsa_pipeline.md import run_heating as _run_heating  # noqa: PLC0415

    return _run_heating(
        simulation_time,
        minimized,
        work_dir=work_dir,
        temperature_start=0 * BSS.Units.Temperature.kelvin,
        temperature_end=300 * BSS.Units.Temperature.kelvin,
        restraint="backbone",
    )
