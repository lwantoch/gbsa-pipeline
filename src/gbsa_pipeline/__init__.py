"""GBSA pipeline package.

Optimize GBSA calculations for speed
"""

from __future__ import annotations

# Set AMBERHOME before any module below pulls in BioSimSpace/Sire (which read it
# at import time and cache it). Must run first — see gbsa_pipeline._amber_env.
from gbsa_pipeline._amber_env import ensure_amberhome as _ensure_amberhome

_ensure_amberhome()

from gbsa_pipeline.config import RunConfig
from gbsa_pipeline.frcmod_parametrization import AmberFFInput, AmberInput, build_amber_ff_xml, load_amber_complex
from gbsa_pipeline.parametrization import ParametrisedComplex, ParametrizationConfig, ParametrizationInput, parametrize
from gbsa_pipeline.pipeline import run_pipeline
from gbsa_pipeline.solvation_box import SolvatedComplex
from gbsa_pipeline.solvation_bss import solvate_bss
from gbsa_pipeline.solvation_openmm import solvate_openmm

__all__ = [
    "AmberFFInput",
    "AmberInput",
    "ParametrisedComplex",
    "ParametrizationConfig",
    "ParametrizationInput",
    "RunConfig",
    "SolvatedComplex",
    "build_amber_ff_xml",
    "load_amber_complex",
    "parametrize",
    "run_pipeline",
    "solvate_bss",
    "solvate_openmm",
]
