"""Functional pipeline runner — orchestrates all MD simulation stages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import BioSimSpace as BSS

from gbsa_pipeline.change_defaults import run_gro_custom
from gbsa_pipeline.equilibration import run_heating
from gbsa_pipeline.minimization import run_minimization
from gbsa_pipeline.parametrization import export_gromacs_top_gro, parametrize
from gbsa_pipeline.solvation_box import SolvationParams
from gbsa_pipeline.solvation_openmm import SolvatedComplex, solvate_openmm

if TYPE_CHECKING:
    from pathlib import Path

    from gbsa_pipeline.config import RunConfig, SolvationConfig

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _run_stage(name: str, fn: Callable[[], _T]) -> _T:
    logger.info("  Starting %s …", name)
    try:
        result = fn()
    except Exception:
        logger.exception("Stage '%s' failed:", name)
        raise
    logger.info("  %s completed.", name)
    return result


def _save_bss_stage(system: Any, output_path: Path) -> None:
    """Write a BSS System to ``{output_path}.gro`` and ``{output_path}.top``."""
    BSS.IO.saveMolecules(str(output_path), system, fileformat=["GRO", "TOP"])


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
    logger.info(
        "  protein_ff=%s  ligand_ff=%s  charge_method=%s",
        config.forcefield.protein_ff,
        config.forcefield.ligand_ff,
        config.forcefield.charge_method,
    )
    param_dir = output_dir / "01_parametrize"
    parametrized = _run_stage("parametrize", lambda: parametrize(config.to_parametrization_input(param_dir)))
    logger.info("  Done → %s, %s", parametrized.gro_file.name, parametrized.top_file.name)

    # Stage 2: Solvate with OpenMM + ParmEd (bypasses BSS IO)
    logger.info("─── Stage 2/5: Solvation ───")
    sol = config.solvation
    box_desc = f"padding={sol.padding} nm" if sol.padding is not None else f"box_size={sol.box_size} nm"
    logger.info(
        "  water_model=%s  box_shape=%s  %s  ion_conc=%s mol/L",
        sol.water_model,
        sol.box_shape,
        box_desc,
        sol.ion_concentration,
    )
    solvation_params = _to_solvation_params(sol)
    sol_dir = output_dir / "02_solvated"
    solvated: SolvatedComplex = _run_stage(
        "solvation",
        lambda: solvate_openmm(
            parametrized=parametrized,
            params=solvation_params,
            output_gro=sol_dir / "solvated.gro",
            output_top=sol_dir / "solvated.top",
        ),
    )
    logger.info("  Saved → %s / %s", solvated.gro_file.name, solvated.top_file.name)

    logger.info("  Loading solvated system into BSS …")
    system = solvated.load_bss()
    logger.info("  Loaded %d molecules (%d atoms)", system.nMolecules(), system.nAtoms())

    # Stage 3: Minimize
    logger.info("─── Stage 3/5: Minimization ───")
    logger.info(
        "  nsteps=%d  emtol=%.1f kJ/mol/nm",
        config.minimization.nsteps,
        config.minimization.emtol,
    )
    min_dir = output_dir / "03_minimized"
    min_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage(
        "minimization",
        lambda: run_minimization(nsteps=config.minimization.nsteps, system=system, work_dir=min_dir),
    )
    logger.info("  Done. Saving …")
    export_gromacs_top_gro(system, str(min_dir / "minimized"))
    logger.info("  Saved → 03_minimized.gro / .top")

    # Stage 4: Equilibrate
    logger.info("─── Stage 4/5: Equilibration ───")
    logger.info("  NVT heating 0→300 K over %.1f ps", config.equilibration.simulation_time_ps)
    equil_time = config.equilibration.simulation_time_ps * BSS.Units.Time.picosecond
    equil_dir = output_dir / "04_equilibrated"
    equil_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage("equilibration", lambda: run_heating(equil_time, system, work_dir=equil_dir))
    logger.info("  Done. Saving …")
    export_gromacs_top_gro(system, str(equil_dir / "equilibrated"))
    logger.info("  Saved → 04_equilibrated.gro / .top")

    # Stage 5: Production MD
    logger.info("─── Stage 5/5: Production MD ───")
    logger.info(
        "  integrator=%s  nsteps=%d  dt=%s ps  tcoupl=%s  pcoupl=%s",
        config.md.integrator,
        config.md.nsteps,
        config.md.dt,
        config.md.tcoupl,
        config.md.pcoupl,
    )
    prod_dir = output_dir / "05_production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    system, _ = _run_stage(
        "production_md", lambda: run_gro_custom(parameters=config.md, system=system, work_dir=prod_dir)
    )
    logger.info("  Done. Saving …")
    export_gromacs_top_gro(system, str(prod_dir / "production"))
    logger.info("  Saved → 05_production.gro / .top")

    logger.info("Pipeline complete. Output written to %s", output_dir)


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
