"""Functional pipeline runner — orchestrates all MD simulation stages."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import BioSimSpace as BSS

from gbsa_pipeline.md import run_heating, run_production
from gbsa_pipeline.parametrization import export_gromacs_top_gro, parametrize
from gbsa_pipeline.solvation_box import SolvationParams
from gbsa_pipeline.solvation_box import SolvatedComplex
from gbsa_pipeline.solvation_openmm import solvate_openmm

if TYPE_CHECKING:
    from pathlib import Path

    from gbsa_pipeline.config import RunConfig, SolvationConfig
    from gbsa_pipeline.parametrization import ParametrisedComplex

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def _run_stage(name: str, fn: Callable[[], _T]) -> _T:
    """Run a named pipeline stage with logging and elapsed-time reporting.

    The stage name is logged on entry and exit so long-running runs produce a
    readable trace. Elapsed time is always reported, even on failure, to help
    diagnose slow or hanging stages. Any exception is re-raised after logging so
    the caller can decide how to handle pipeline failures.
    """
    logger.info("  [%s] starting …", name)
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception("  [%s] failed after %.1f s", name, elapsed)
        raise
    elapsed = time.perf_counter() - t0
    logger.info("  [%s] completed in %.1f s", name, elapsed)
    return result


# ---------------------------------------------------------------------------
# Individual stage helpers — pure functions over validated inputs
# ---------------------------------------------------------------------------


def _stage_parametrize(config: RunConfig, stage_dir: Path) -> ParametrisedComplex:
    """Assign force field parameters to the protein-ligand complex."""
    logger.info(
        "  protein_ff=%s  ligand_ff=%s  charge_method=%s",
        config.forcefield.protein_ff,
        config.forcefield.ligand_ff,
        config.forcefield.charge_method,
    )
    return parametrize(config.to_parametrization_input(stage_dir))


def _stage_solvate(
    config: RunConfig,
    parametrized: ParametrisedComplex,
    stage_dir: Path,
) -> Any:
    """Solvate with OpenMM + ParmEd and return the loaded BSS system."""
    sol = config.solvation
    box_desc = f"padding={sol.padding} nm" if sol.padding is not None else f"box_size={sol.box_size} nm"
    logger.info(
        "  water_model=%s  box_shape=%s  %s  ion_conc=%s mol/L",
        sol.water_model,
        sol.box_shape,
        box_desc,
        sol.ion_concentration,
    )
    solvated: SolvatedComplex = solvate_openmm(
        parametrized=parametrized,
        params=_to_solvation_params(sol),
        output_gro=stage_dir / "solvated.gro",
        output_top=stage_dir / "solvated.top",
    )
    logger.info("  Saved → %s / %s", solvated.gro_file.name, solvated.top_file.name)

    logger.info("  Loading solvated system into BSS …")
    system = solvated.load_bss()
    logger.info("  Loaded %d molecules (%d atoms)", system.nMolecules(), system.nAtoms())
    return system


def _stage_minimize(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Run energy minimization via GROMACS using a BSS Minimisation protocol.

    BioSimSpace owns the base MDP (``integrator = steep``) so the minimization
    protocol is always correctly configured. ``nsteps`` and ``emtol`` from
    the run config are passed directly to ``BSS.Protocol.Minimisation``.
    """
    logger.info(
        "  nsteps=%d  emtol=%.1f kJ/mol/nm",
        config.minimization.nsteps,
        config.minimization.emtol,
    )
    protocol = BSS.Protocol.Minimisation(steps=config.minimization.nsteps)
    process = BSS.Process.Gromacs(
        system,
        protocol,
        name="min",
        ignore_warnings=True,
        work_dir=str(stage_dir),
    )
    process.start()
    result = process.getSystem(block=True)

    if result is None:
        raise RuntimeError(
            f"Minimization finished without a readable output system. Check GROMACS logs in {stage_dir}."
        )
    return result


def _stage_equilibrate(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Heat from 0 K to 300 K under NVT with backbone restraints."""
    equil_time = config.equilibration.simulation_time_ps * BSS.Units.Time.picosecond
    logger.info("  NVT heating 0→300 K over %.1f ps", config.equilibration.simulation_time_ps)
    return run_heating(
        equil_time,
        system,
        work_dir=stage_dir,
        temperature_start=0 * BSS.Units.Temperature.kelvin,
        temperature_end=300 * BSS.Units.Temperature.kelvin,
        restraint="backbone",
    )


def _stage_production(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Run production MD using ``config.md`` parameters."""
    logger.info(
        "  integrator=%s  nsteps=%d  dt=%s ps  tcoupl=%s  pcoupl=%s",
        config.md.integrator,
        config.md.nsteps,
        config.md.dt,
        config.md.tcoupl,
        config.md.pcoupl,
    )
    # Compute simulation time from nsteps * dt (GROMACS dt is in picoseconds).
    sim_time = config.md.nsteps * config.md.dt * BSS.Units.Time.picosecond
    return run_production(
        simulation_time=sim_time,
        equilibrated=system,
        work_dir=stage_dir,
        params=config.md,
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(config: RunConfig, output_dir: Path) -> None:
    """Run the full GBSA pipeline from a validated :class:`~gbsa_pipeline.config.RunConfig`.

    Stages (each writes output to a numbered subdirectory):

    1. **Parametrize** — assign force field parameters to protein + ligand.
    2. **Solvate** — add water box and counter-ions.
    3. **Minimize** — energy minimization.
    4. **Equilibrate** — NVT heating from 0 K to 300 K.
    5. **Production MD** — NpT simulation driven by ``[md]`` section params.

    Parameters
    ----------
    config:
        Validated run configuration (usually loaded via
        :meth:`~gbsa_pipeline.config.RunConfig.from_toml`).
    output_dir:
        Root directory for all output. Created if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_config(config, output_dir)

    # Stage 1: Parametrize
    logger.info("─── Stage 1/5: Parametrization ───")
    param_dir = output_dir / "01_parametrize"
    parametrized = _run_stage("parametrize", lambda: _stage_parametrize(config, param_dir))
    logger.info("  Done → %s, %s", parametrized.gro_file.name, parametrized.top_file.name)

    # Stage 2: Solvate
    logger.info("─── Stage 2/5: Solvation ───")
    sol_dir = output_dir / "02_solvated"
    system = _run_stage("solvation", lambda: _stage_solvate(config, parametrized, sol_dir))

    # Stage 3: Minimize
    logger.info("─── Stage 3/5: Minimization ───")
    min_dir = output_dir / "03_minimized"
    min_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("minimization", lambda: _stage_minimize(config, system, min_dir))
    logger.info("  Done. Saving …")
    export_gromacs_top_gro(system, str(min_dir / "minimized"))
    logger.info("  Saved → 03_minimized/minimized.gro / .top")

    # Stage 4: Equilibrate
    logger.info("─── Stage 4/5: Equilibration ───")
    equil_dir = output_dir / "04_equilibrated"
    equil_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("equilibration", lambda: _stage_equilibrate(config, system, equil_dir))
    logger.info("  Done. Saving …")
    export_gromacs_top_gro(system, str(equil_dir / "equilibrated"))
    logger.info("  Saved → 04_equilibrated/equilibrated.gro / .top")

    # Stage 5: Production MD
    logger.info("─── Stage 5/5: Production MD ───")
    prod_dir = output_dir / "05_production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("production_md", lambda: _stage_production(config, system, prod_dir))
    logger.info("  Done. Saving …")
    export_gromacs_top_gro(system, str(prod_dir / "production"))
    logger.info("  Saved → 05_production/production.gro / .top")

    logger.info("Pipeline complete. Output written to %s", output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_solvation_params(cfg: SolvationConfig) -> SolvationParams:
    """Map a :class:`~gbsa_pipeline.config.SolvationConfig` to a :class:`~gbsa_pipeline.solvation_box.SolvationParams`."""
    return SolvationParams(
        water_model=cfg.water_model,
        shape=cfg.box_shape,
        padding=cfg.padding,
        box_size=cfg.box_size,
        ion_concentration=cfg.ion_concentration,
        neutralize=cfg.neutralize,
    )


def _log_config(config: RunConfig, output_dir: Path) -> None:
    """Write a JSON snapshot of the resolved config to ``output_dir/run_config.json``."""
    config_path = output_dir / "run_config.json"
    config_path.write_text(config.model_dump_json(indent=2))
    logger.info("Config written to %s", config_path)
