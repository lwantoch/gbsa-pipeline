"""Functional pipeline runner — orchestrates all MD simulation stages."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import BioSimSpace as BSS

from gbsa_pipeline.md import (
    remove_clashing_solvent_waters,
    run_heating,
    run_minimization,
    run_npt_equilibration,
    run_production,
    run_solvent_relaxation,
)
from gbsa_pipeline.md_io import save_bss_system_to_gromacs
from gbsa_pipeline.parametrization import parametrize
from gbsa_pipeline.solvation_bss import solvate_bss
from gbsa_pipeline.solvation_box import SolvationParams

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
    """Run a named pipeline stage with logging and elapsed-time reporting."""
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
    """Solvate with BSS.Solvent (gmx solvate + gmx genion) and return loaded BSS system."""
    sol = config.solvation
    box_desc = f"padding={sol.padding} nm" if sol.padding is not None else f"box_size={sol.box_size} nm"
    logger.info(
        "  water_model=%s  box_shape=%s  %s  ion_conc=%s mol/L",
        sol.water_model,
        sol.box_shape,
        box_desc,
        sol.ion_concentration,
    )
    solvated = solvate_bss(
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


def _stage_minimize_sd(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Steepest-descent energy minimization."""
    logger.info("  nsteps=%d  emtol=%.1f kJ/mol/nm", config.minimization.nsteps, config.minimization.emtol)
    return run_minimization(
        system,
        work_dir=stage_dir,
        params={"nsteps": config.minimization.nsteps, "emtol": config.minimization.emtol},
    )


def _stage_minimize_cg(system: Any, stage_dir: Path) -> Any:
    """Conjugate-gradient energy minimization."""
    return run_minimization(system, work_dir=stage_dir, params={"integrator": "cg"})


def _stage_nvt_restrained(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Water clash removal → short solvent relax → NVT heating 50→300 K with backbone restraints."""
    logger.info("  NVT heating over %.1f ps", config.equilibration.simulation_time_ps)

    system = remove_clashing_solvent_waters(system, work_dir=stage_dir / "water_cleanup")
    system = run_solvent_relaxation(system, work_dir=stage_dir / "solvent_relax")

    equil_time = config.equilibration.simulation_time_ps * BSS.Units.Time.picosecond
    return run_heating(
        equil_time,
        system,
        work_dir=stage_dir,
        temperature_start=50 * BSS.Units.Temperature.kelvin,
        temperature_end=300 * BSS.Units.Temperature.kelvin,
        restraint="backbone",
    )


def _stage_npt_restrained(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """NPT equilibration with backbone restraints."""
    logger.info("  %.1f ps  restraint=backbone", config.npt_equilibration.simulation_time_ps)
    npt_time = config.npt_equilibration.simulation_time_ps * BSS.Units.Time.picosecond
    return run_npt_equilibration(npt_time, system, work_dir=stage_dir, restraint="backbone")


def _stage_npt(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """NPT equilibration without restraints."""
    logger.info("  %.1f ps  restraint=none", config.npt_equilibration.simulation_time_ps)
    npt_time = config.npt_equilibration.simulation_time_ps * BSS.Units.Time.picosecond
    return run_npt_equilibration(npt_time, system, work_dir=stage_dir, restraint=None)


def _stage_production(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Production MD."""
    sim_time = config.md.nsteps * config.md.dt * BSS.Units.Time.picosecond
    logger.info(
        "  nsteps=%d  dt=%s ps  sim_time=%.1f ps  tcoupl=%s  pcoupl=%s",
        config.md.nsteps,
        config.md.dt,
        config.md.nsteps * config.md.dt,
        config.md.tcoupl,
        config.md.pcoupl,
    )
    return run_production(sim_time, system, work_dir=stage_dir, params=config.md)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(config: RunConfig, output_dir: Path) -> None:
    """Run the full GBSA pipeline from a validated :class:`~gbsa_pipeline.config.RunConfig`.

    Stages (each writes output to a numbered subdirectory):

    1. **Parametrize** — assign force field parameters to protein + ligand.
    2. **Solvate** — add water box and counter-ions via BSS.Solvent.
    3. **SD Minimization** — steepest-descent energy minimization.
    4. **CG Minimization** — conjugate-gradient energy minimization.
    5. **NVT Restrained** — water cleanup, solvent relax, NVT heating 50→300 K.
    6. **NPT Restrained** — NPT equilibration with backbone restraints.
    7. **NPT** — NPT equilibration without restraints.
    8. **Production MD** — NpT simulation driven by ``[md]`` section params.

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
    logger.info("─── Stage 1/8: Parametrization ───")
    param_dir = output_dir / "01_parametrize"
    parametrized = _run_stage("parametrize", lambda: _stage_parametrize(config, param_dir))
    logger.info("  Done → %s, %s", parametrized.gro_file.name, parametrized.top_file.name)

    # Stage 2: Solvate
    logger.info("─── Stage 2/8: Solvation ───")
    sol_dir = output_dir / "02_solvated"
    system = _run_stage("solvation", lambda: _stage_solvate(config, parametrized, sol_dir))

    # Stage 3: SD minimization
    logger.info("─── Stage 3/8: SD Minimization ───")
    sd_dir = output_dir / "03_sd"
    sd_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("sd_minimization", lambda: _stage_minimize_sd(config, system, sd_dir))
    save_bss_system_to_gromacs(system, sd_dir / "system")
    logger.info("  Saved → 03_sd/system.gro / .top")

    # Stage 4: CG minimization
    logger.info("─── Stage 4/8: CG Minimization ───")
    cg_dir = output_dir / "04_cg"
    cg_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("cg_minimization", lambda: _stage_minimize_cg(system, cg_dir))
    save_bss_system_to_gromacs(system, cg_dir / "system")
    logger.info("  Saved → 04_cg/system.gro / .top")

    # Stage 5: NVT restrained heating
    logger.info("─── Stage 5/8: NVT Restrained Heating ───")
    nvt_dir = output_dir / "05_nvt_res"
    nvt_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("nvt_restrained", lambda: _stage_nvt_restrained(config, system, nvt_dir))
    save_bss_system_to_gromacs(system, nvt_dir / "system")
    logger.info("  Saved → 05_nvt_res/system.gro / .top")

    # Stage 6: NPT restrained equilibration
    logger.info("─── Stage 6/8: NPT Restrained Equilibration ───")
    npt_res_dir = output_dir / "06_npt_res"
    npt_res_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("npt_restrained", lambda: _stage_npt_restrained(config, system, npt_res_dir))
    save_bss_system_to_gromacs(system, npt_res_dir / "system")
    logger.info("  Saved → 06_npt_res/system.gro / .top")

    # Stage 7: NPT unrestrained equilibration
    logger.info("─── Stage 7/8: NPT Equilibration ───")
    npt_dir = output_dir / "07_npt"
    npt_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("npt", lambda: _stage_npt(config, system, npt_dir))
    save_bss_system_to_gromacs(system, npt_dir / "system")
    logger.info("  Saved → 07_npt/system.gro / .top")

    # Stage 8: Production MD
    logger.info("─── Stage 8/8: Production MD ───")
    prod_dir = output_dir / "08_production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("production_md", lambda: _stage_production(config, system, prod_dir))
    save_bss_system_to_gromacs(system, prod_dir / "system")
    logger.info("  Saved → 08_production/system.gro / .top")

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
