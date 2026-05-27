"""Energy minimization — legacy compatibility shim.

.. deprecated::
    Use :func:`gbsa_pipeline.md.run_minimization` directly.
    This module remains to avoid breaking existing callers that pass ``nsteps``
    positionally.  New code should construct a
    :class:`~gbsa_pipeline.mdp.GromacsParams` with
    ``integrator=Integrator.STEEP`` and pass it to ``md.run_minimization``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import BioSimSpace as BSS

if TYPE_CHECKING:
    from pathlib import Path

    import sire


def run_minimization(
    nsteps: int,
    system: sire.System,
    work_dir: Path | None = None,
    *,
    ignore_warnings: bool = True,
    **kwargs: Any,
) -> BSS.System:
    """Run energy minimization via GROMACS.

    .. deprecated::
        Use :func:`gbsa_pipeline.md.run_minimization` with a
        :class:`~gbsa_pipeline.mdp.GromacsParams` instead.

    Args:
        nsteps: Maximum number of minimization steps.
        system: Solvated system to minimize.
        work_dir: Optional working directory for GROMACS files.
        ignore_warnings: Suppress BioSimSpace process warnings (default True).
        **kwargs: Extra keyword arguments forwarded to ``BSS.Process.Gromacs``.
            Deprecated — ``ignore_warnings`` is the only supported override.

    Returns:
        Minimized BioSimSpace system.
    """
    warnings.warn(
        "gbsa_pipeline.minimization.run_minimization is deprecated. "
        "Use gbsa_pipeline.md.run_minimization with GromacsParams(integrator=Integrator.STEEP, nsteps=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    protocol = BSS.Protocol.Minimisation(steps=nsteps)

    gromacs_kwargs: dict[str, Any] = {"ignore_warnings": ignore_warnings, **kwargs}
    if work_dir is not None:
        gromacs_kwargs["work_dir"] = str(work_dir)

    process = BSS.Process.Gromacs(system, protocol, name="min", **gromacs_kwargs)
    process.start()
    result = process.getSystem(block=True)

    if result is None:
        raise RuntimeError(
            f"Minimization finished without a readable output system. Check GROMACS logs in {work_dir!r}."
        )

    return result
