"""Ensure ``AMBERHOME`` is set *before* the first BioSimSpace / Sire import.

BioSimSpace (via Sire) reads ``AMBERHOME`` at import time to locate ``tleap`` and
the AMBER force-field data. When the engine is launched through a bare
interpreter — e.g. ``<env>/bin/python action.py`` without activating the
conda/pixi env — ``PATH`` may include the env's ``bin`` (so ``antechamber`` /
``sqm`` resolve and ligand parametrization succeeds) while ``AMBERHOME`` stays
unset. The tleap-backed steps (``BSS.Parameters.ff14SB`` on crystal waters/ions)
then crash at the C level with no Python traceback.

Setting ``AMBERHOME`` here, before anything imports BioSimSpace, fixes it
regardless of how the process was launched. Setting it *after* Sire is imported
is too late: Sire has already cached its AMBER configuration, so a later
``os.environ['AMBERHOME'] = ...`` is silently ignored.

This module deliberately imports nothing beyond the standard library so it can
run before every other ``gbsa_pipeline`` import.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def guess_amberhome() -> str:
    """Best-effort AMBERHOME from the location of ``tleap`` (conda/pixi env prefix)."""
    tleap = shutil.which("tleap")
    if tleap:
        return str(Path(tleap).resolve().parent.parent)
    return os.environ.get("CONDA_PREFIX", "")


def ensure_amberhome() -> str | None:
    """Set ``AMBERHOME`` from the active env if unset. Returns the value in effect.

    No-op when ``AMBERHOME`` is already set. Only adopts a guessed prefix when it
    actually contains the AMBER leap data (``dat/leap``), so a wrong guess can
    never shadow a real install.
    """
    existing = os.environ.get("AMBERHOME")
    if existing:
        return existing
    home = guess_amberhome()
    if home and Path(home, "dat", "leap").is_dir():
        os.environ["AMBERHOME"] = home
        return home
    return None
