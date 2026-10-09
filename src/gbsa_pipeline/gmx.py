"""Thin subprocess wrapper around ``gmx grompp`` + ``gmx mdrun``.

Replaces :class:`BSS.Process.Gromacs` with explicit, debuggable subprocess
calls. The pipeline owns the MDP contents end-to-end via
:class:`~gbsa_pipeline.mdp.GromacsParams`; velocities carry across stages via
``grompp -t <cpt>``, which embeds the checkpoint velocities into the next TPR
directly. This replaces the ``mdrun -cpi`` workaround previously needed in
``md.py`` and is the only reliable mechanism when the integrator changes
between stages (``mdrun -cpi`` then aborts with ``"Cannot change integrator
during a checkpoint restart"``).

grompp is called with ``-maxwarn 0`` by default: every warning aborts the
stage.  This is the opposite of :class:`BSS.Process.Gromacs`, which hardcoded
``--maxwarn 9999`` and hid every grompp warning the user had no way to see.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

    from gbsa_pipeline.mdp import GromacsParams

logger = logging.getLogger(__name__)


class StageOutput(NamedTuple):
    """Files produced by a single ``grompp`` + ``mdrun`` invocation.

    All paths point inside ``work_dir`` and follow the GROMACS ``-deffnm gromacs``
    convention. The ``xtc`` and ``edr`` paths are returned even when the stage's
    MDP sets their output cadence to 0; callers that depend on them should check ``.exists()``.
    """

    gro: Path
    cpt: Path
    log: Path
    edr: Path
    xtc: Path
    tpr: Path
    work_dir: Path


class GromppError(RuntimeError):
    """``gmx grompp`` returned nonzero or exceeded ``maxwarn``."""


class MdrunError(RuntimeError):
    """``gmx mdrun`` returned nonzero or did not produce ``gromacs.gro``."""


def _tail(path: Path, max_lines: int = 60) -> str:
    """Return the last ``max_lines`` lines of a text file.

    Falls back to a placeholder when the file does not exist, so that
    diagnostic messages are safe to format even if grompp/mdrun crashed
    before writing anything.
    """
    if not path.exists():
        return f"<missing: {path}>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _ntomp() -> int:
    """OpenMP thread count: honour SLURM, then ``OMP_NUM_THREADS``, then ``os.cpu_count()``."""
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = os.environ.get(var)
        if value:
            return int(value)
    return os.cpu_count() or 1


def run_stage(
    name: str,
    mdp: GromacsParams,
    gro: Path,
    top: Path,
    work_dir: Path,
    *,
    cpt: Path | None = None,
    restrained: bool = False,
    gmx: str = "gmx",
    maxwarn: int = 0,
) -> StageOutput:
    """Run one ``grompp`` + ``mdrun`` stage and return the produced files.

    Parameters
    ----------
    name:
        Short label used in log messages (e.g. ``"nvt_restrained"``).
    mdp:
        Validated GROMACS parameters, rendered to ``gromacs.mdp`` via
        :meth:`GromacsParams.to_mdp`.
    gro, top:
        Input coordinates and topology.
    work_dir:
        Directory to write MDP, TPR, trajectory and log files.  Created if
        missing.  grompp and mdrun are both run with ``cwd=work_dir`` so
        GROMACS backup files (``#*#``) stay isolated per stage.
    cpt:
        If given, passed to ``grompp`` as ``-t``.  The checkpoint's velocity
        state is embedded into the generated TPR, which is the only reliable
        way to carry velocities between stages whose integrators differ.
    restrained:
        If True, ``grompp -r <gro>`` supplies restraint reference
        coordinates.  The input ``gro`` doubles as the reference, matching
        the project's existing convention.
    gmx:
        Name of the GROMACS executable on ``PATH``.  Default ``"gmx"``; set
        to ``"gmx_d"`` for a double-precision binary.
    maxwarn:
        Passed verbatim to grompp.  Default ``0`` — any warning aborts.

    Raises:
    ------
    GromppError
        ``grompp`` returned nonzero or exceeded ``maxwarn`` warnings.
    MdrunError
        ``mdrun`` returned nonzero or did not produce ``gromacs.gro``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    mdp_path = work_dir / "gromacs.mdp"
    tpr_path = work_dir / "gromacs.tpr"
    mdp_path.write_text(mdp.to_mdp(), encoding="utf-8")

    grompp_cmd = [
        gmx,
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(gro),
        "-p",
        str(top),
        "-o",
        str(tpr_path),
        "-po",
        str(work_dir / "gromacs.out.mdp"),
        "-maxwarn",
        str(maxwarn),
    ]
    if restrained:
        grompp_cmd.extend(["-r", str(gro)])
    if cpt is not None:
        grompp_cmd.extend(["-t", str(cpt)])

    logger.info("  [%s] grompp …", name)
    grompp = subprocess.run(grompp_cmd, cwd=work_dir, capture_output=True, text=True, check=False)  # noqa: S603
    (work_dir / "grompp.out").write_text(grompp.stdout, encoding="utf-8")
    (work_dir / "grompp.err").write_text(grompp.stderr, encoding="utf-8")
    if grompp.returncode != 0:
        raise GromppError(
            f"[{name}] grompp returned {grompp.returncode}\n--- grompp.err tail ---\n{_tail(work_dir / 'grompp.err')}"
        )

    ntomp = _ntomp()
    mdrun_cmd = [gmx, "mdrun", "-deffnm", "gromacs", "-ntmpi", "1", "-ntomp", str(ntomp)]
    logger.info("  [%s] mdrun -ntomp %d …", name, ntomp)
    mdrun = subprocess.run(mdrun_cmd, cwd=work_dir, capture_output=True, text=True, check=False)  # noqa: S603
    (work_dir / "mdrun.out").write_text(mdrun.stdout, encoding="utf-8")
    (work_dir / "mdrun.err").write_text(mdrun.stderr, encoding="utf-8")

    final_gro = work_dir / "gromacs.gro"
    if mdrun.returncode != 0 or not final_gro.exists():
        raise MdrunError(
            f"[{name}] mdrun returned {mdrun.returncode}; final gro present: {final_gro.exists()}\n"
            f"--- mdrun.err tail ---\n{_tail(work_dir / 'mdrun.err')}\n"
            f"--- gromacs.log tail ---\n{_tail(work_dir / 'gromacs.log')}"
        )

    return StageOutput(
        gro=final_gro,
        cpt=work_dir / "gromacs.cpt",
        log=work_dir / "gromacs.log",
        edr=work_dir / "gromacs.edr",
        xtc=work_dir / "gromacs.xtc",
        tpr=tpr_path,
        work_dir=work_dir,
    )
