"""Functional pipeline runner — orchestrates all MD simulation stages."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import BioSimSpace as BSS

from gbsa_pipeline.md import (
    npt_barostat_overrides,
    remove_clashing_solvent_waters,
    run_heating,
    run_minimization,
    run_npt_equilibration,
    run_production,
    run_solvent_relaxation,
)
from gbsa_pipeline.md_io import save_bss_system_to_gromacs
from gbsa_pipeline.membrane import canonicalize_gromacs_system
from gbsa_pipeline.parametrization import parametrize
from gbsa_pipeline.solvation_bss import solvate_bss

if TYPE_CHECKING:
    from pathlib import Path

    from gbsa_pipeline.config import RunConfig
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


def _run_md_stage(
    title: str,
    name: str,
    label: str,
    output_dir: Path,
    fn: Callable[[Path], Any],
) -> Any:
    """Run one MD stage: log banner, mkdir, run, save gro/top, return system."""
    logger.info("─── %s ───", title)
    stage_dir = output_dir / label
    stage_dir.mkdir(parents=True, exist_ok=True)
    system = _run_stage(name, lambda: fn(stage_dir))
    save_bss_system_to_gromacs(system, stage_dir / "system")
    logger.info("  Saved → %s/system.gro / .top", label)
    return system


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
        "  water_model=%s  shape=%s  %s  ion_conc=%s mol/L",
        sol.water_model,
        sol.shape,
        box_desc,
        sol.ion_concentration,
    )
    solvated = solvate_bss(
        parametrized=parametrized,
        params=sol,
        output_gro=stage_dir / "solvated.gro",
        output_top=stage_dir / "solvated.top",
    )
    logger.info("  Saved → %s / %s", solvated.gro_file.name, solvated.top_file.name)

    logger.info("  Loading solvated system into BSS …")
    system = solvated.load_bss()
    logger.info("  Loaded %d molecules (%d atoms)", system.nMolecules(), system.nAtoms())
    return system


def _stage_load_membrane_system(config: RunConfig, stage_dir: Path) -> Any:
    """Load a pre-built, already-solvated membrane system (e.g. from MemProtMD).

    Used instead of _stage_parametrize + _stage_solvate when
    config.membrane_system is set: structure/topology are already a complete
    protein-in-bilayer-in-water system, so there is nothing left to
    parametrize or add solvent to (see gbsa_pipeline.membrane and
    config.MembraneSystemConfig). make_whole=True rebuilds molecules split
    across periodic boundaries, matching how BSS-loaded systems are handled
    elsewhere in this pipeline (e.g. the mmbsa integration test's loader).

    The structure is first run through canonicalize_gromacs_system(), which
    fixes two legacy-GROMACS issues that make BioSimSpace/Sire's loader fail
    outright (confirmed against MemProtMD's atomistic-system.pdb: a
    hydrogen-naming convention mismatch, and a >9999-residue PDB
    numbering wrap) even though plain gmx grompp tolerates both — see that
    function's docstring. This needs a real GROMACS install; it is not
    optional for the membrane_system path.

    Loading a system fixed this way still doesn't guarantee later MD stages
    will succeed: BioSimSpace/Sire's own topology writer has a separate,
    unrelated data-loss bug (drops [nonbond_params]/[pairtypes] override
    tables on re-serialization) that can break minimization for force fields
    that rely on them, such as MemProtMD's default GROMOS+OPLS-lipid output —
    see the gbsa_pipeline.membrane module docstring for the full
    investigation. That is a BioSimSpace/Sire limitation, not something this
    function (or this pipeline) can fix.
    """
    membrane = config.membrane_system
    if membrane is None:  # pragma: no cover — guarded by RunConfig's validator
        raise ValueError("_stage_load_membrane_system requires config.membrane_system to be set.")
    canonical_gro = canonicalize_gromacs_system(membrane.structure, membrane.topology, stage_dir / "_canonicalize")
    logger.info(
        "  structure=%s (canonicalized → %s)  topology=%s", membrane.structure, canonical_gro.name, membrane.topology
    )
    system = BSS.IO.readMolecules([str(canonical_gro), str(membrane.topology)], make_whole=True)
    logger.info("  Loaded %d molecules (%d atoms)", system.nMolecules(), system.nAtoms())
    return system


def _stage_minimize_sd(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Steepest-descent energy minimization.

    ``params`` is applied as a complete MDP overlay (see
    ``_apply_gromacs_params_to_config`` in md.py), so any field not given
    here reverts to GromacsParams' class default -- including ``integrator``,
    whose default is ``"md"`` (leapfrog), not ``"steep"``. Without an explicit
    ``integrator: "steep"`` this stage silently ran plain MD dynamics on the
    raw, unminimized starting structure instead of steepest-descent
    minimization, which is exactly the kind of input that can blow up to a
    NaN potential energy at step 0 -- confirmed while debugging the membrane
    path, but the bug applies to every run, not only membrane ones.
    """
    logger.info(
        "  nsteps=%d  emtol=%.1f kJ/mol/nm  define=%s",
        config.minimization.nsteps,
        config.minimization.emtol,
        config.minimization.define or "(none)",
    )
    return run_minimization(
        system,
        work_dir=stage_dir,
        params={
            "integrator": "steep",
            "nsteps": config.minimization.nsteps,
            "emtol": config.minimization.emtol,
            "define": config.minimization.define,
        },
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


def _stage_npt(config: RunConfig, system: Any, stage_dir: Path, *, restraint: str | None = None) -> Any:
    """NPT equilibration, optionally with backbone restraints.

    Uses the same barostat (pcoupltype, ref_p, compressibility, tau_p) as the
    [md] production section — see npt_barostat_overrides() — so a membrane
    system configured with pcoupltype = semiisotropic there gets consistent
    (not isotropic) pressure coupling during equilibration too.
    """
    logger.info(
        "  %.1f ps  restraint=%s  pcoupltype=%s",
        config.npt_equilibration.simulation_time_ps,
        restraint or "none",
        config.md.pcoupltype,
    )
    npt_time = config.npt_equilibration.simulation_time_ps * BSS.Units.Time.picosecond
    return run_npt_equilibration(
        npt_time,
        system,
        work_dir=stage_dir,
        restraint=restraint,
        params=npt_barostat_overrides(config.md),
    )


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

    When ``config.membrane_system`` is set instead of ``config.system``,
    stages 1-2 are replaced by a single load of the already-complete,
    already-solvated system (see :class:`~gbsa_pipeline.config.MembraneSystemConfig`).

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

    if config.membrane_system is not None:
        logger.info("─── Stage 1-2/8: Loading pre-built membrane system ───")
        sol_dir = output_dir / "02_solvated"
        system = _run_stage("load_membrane_system", lambda: _stage_load_membrane_system(config, sol_dir))
    else:
        # Stage 1: Parametrize
        logger.info("─── Stage 1/8: Parametrization ───")
        param_dir = output_dir / "01_parametrize"
        parametrized = _run_stage("parametrize", lambda: _stage_parametrize(config, param_dir))
        logger.info("  Done → %s, %s", parametrized.gro_file.name, parametrized.top_file.name)

        # Stage 2: Solvate
        logger.info("─── Stage 2/8: Solvation ───")
        sol_dir = output_dir / "02_solvated"
        system = _run_stage("solvation", lambda: _stage_solvate(config, parametrized, sol_dir))

    system = _run_md_stage(
        "Stage 3/8: SD Minimization",
        "sd_minimization",
        "03_sd",
        output_dir,
        lambda d: _stage_minimize_sd(config, system, d),
    )
    system = _run_md_stage(
        "Stage 4/8: CG Minimization", "cg_minimization", "04_cg", output_dir, lambda d: _stage_minimize_cg(system, d)
    )
    system = _run_md_stage(
        "Stage 5/8: NVT Restrained Heating",
        "nvt_restrained",
        "05_nvt_res",
        output_dir,
        lambda d: _stage_nvt_restrained(config, system, d),
    )
    system = _run_md_stage(
        "Stage 6/8: NPT Restrained Equilibration",
        "npt_restrained",
        "06_npt_res",
        output_dir,
        lambda d: _stage_npt(config, system, d, restraint="backbone"),
    )
    system = _run_md_stage(
        "Stage 7/8: NPT Equilibration", "npt", "07_npt", output_dir, lambda d: _stage_npt(config, system, d)
    )
    system = _run_md_stage(
        "Stage 8/8: Production MD",
        "production_md",
        "08_production",
        output_dir,
        lambda d: _stage_production(config, system, d),
    )

    logger.info("Pipeline complete. Output written to %s", output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_config(config: RunConfig, output_dir: Path) -> None:
    """Write a JSON snapshot of the resolved config to ``output_dir/run_config.json``."""
    config_path = output_dir / "run_config.json"
    config_path.write_text(config.model_dump_json(indent=2))
    logger.info("Config written to %s", config_path)
