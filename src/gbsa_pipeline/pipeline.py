"""Functional pipeline runner — orchestrates all MD simulation stages."""

from __future__ import annotations

import logging
import shutil
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import BioSimSpace as BSS
import MDAnalysis as mda

from gbsa_pipeline.config import MembraneConfig
from gbsa_pipeline.gromacs_index import select_receptor_and_ligand_atoms, write_index
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
from gbsa_pipeline.membrane import (
    estimate_membrane_geometry,
    extract_protein_ligand_system,
    is_protein_molecule,
    lipid_headgroup_restraint_atoms,
)
from gbsa_pipeline.mmbsa import MMPBSAConfig, run_gmx_mmpbsa_from_gromacs
from gbsa_pipeline.parametrization import parameterise_ligand_gaff2, parametrize
from gbsa_pipeline.solvation_box import solvate_membrane
from gbsa_pipeline.solvation_bss import solvate_bss

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Sequence
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


def _stage_parametrize_membrane(config: RunConfig, stage_dir: Path) -> Any:
    """Load a pre-built [membrane] system and merge with parametrised ligand."""
    system_config = config.system
    if system_config.gro_file is None or system_config.top_file is None:
        raise ValueError("system.gro_file and system.top_file must be set for a membrane system.")
    if system_config.ligand is None:
        raise ValueError("system.ligand must be set to run the membrane parametrization stage.")

    logger.info(
        "gro_file=%s  top_file=%s ligand=%s net_charge=%s",
        system_config.gro_file.name,
        system_config.top_file.name,
        system_config.ligand.name,
        system_config.net_charge,
    )
    system = BSS.IO.readMolecules(
        [str(system_config.gro_file), str(system_config.top_file)],
        make_whole=True,
    )
    ligand_mol = BSS.IO.readMolecules(str(system_config.ligand)).getMolecules()[0]
    ligand = parameterise_ligand_gaff2(
        ligand_mol,
        net_charge=system_config.net_charge,
        work_dir=stage_dir,
    )
    system.addMolecules(ligand)
    return system


def _stage_solvate_membrane(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Solvate a membrane system, or pass it through unchanged if already solvated."""
    membrane = config.membrane or MembraneConfig()
    if not config.solvation.solvate:
        logger.info("solvate=False - system is already solvated, skipping.")
        return system

    logger.info(
        "z_padding_nm=%.2f water_model=%s.",
        membrane.z_padding_nm,
        config.solvation.water_model,
    )
    return solvate_membrane(
        system=system,
        params=config.solvation,
        z_padding_nm=membrane.z_padding_nm,
        lipid_resnames=sorted(membrane.lipid_resnames),
        work_dir=stage_dir,
    )


def _stage_minimize_sd(config: RunConfig, system: Any, stage_dir: Path) -> Any:
    """Steepest-descent energy minimization."""
    logger.info(
        "  nsteps=%d  emtol=%.1f kJ/mol/nm",
        config.minimization.nsteps,
        config.minimization.emtol,
    )
    return run_minimization(
        system,
        work_dir=stage_dir,
        params={
            "integrator": "steep",
            "nsteps": config.minimization.nsteps,
            "emtol": config.minimization.emtol,
        },
    )


def _stage_minimize_cg(system: Any, stage_dir: Path) -> Any:
    """Conjugate-gradient energy minimization."""
    return run_minimization(system, work_dir=stage_dir, params={"integrator": "cg"})


def _membrane_aware_restraint_selection(config: RunConfig, system: Any) -> str | list[int]:
    """Backbone-only restraint for soluble systems; backbone + lipid headgroups for membranes.

    BSS's "backbone" keyword has no concept of a membrane, so a membrane run
    that only restrains the protein backbone lets lipid headgroups drift or
    flip-flop during early NVT/NPT equilibration (CHARMM-GUI's standard
    protocol restrains both together for exactly this reason). Combining them
    requires an explicit atom-index list -- BSS accepts a restraint keyword or
    an index list, not both -- so "backbone" is resolved to indices first via
    ``system.getRestraintAtoms`` and unioned with the lipid phosphate indices.
    """
    if not config.system.membrane:
        return "backbone"

    membrane = config.membrane or MembraneConfig()
    backbone = system.getRestraintAtoms("backbone")
    lipids = lipid_headgroup_restraint_atoms(system, sorted(membrane.lipid_resnames))
    return sorted(set(backbone) | set(lipids))


def _stage_nvt_restrained(
    config: RunConfig,
    system: Any,
    stage_dir: Path,
    *,
    restraint: str | Sequence[int] = "backbone",
) -> Any:
    """Water clash removal → short solvent relax → NVT heating 50→300 K with restraints."""
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
        restraint=restraint,
    )


def _stage_npt(
    config: RunConfig,
    system: Any,
    stage_dir: Path,
    *,
    restraint: str | Sequence[int] | None = None,
    checkpoint_path: Path | None = None,
) -> Any:
    """NPT equilibration, optionally with backbone restraints.

    Uses the same barostat as the [md] section so a memprot configured with
    pcouple = semiisotropic gets consistent, not isotropic, values during
    equilibration.
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
        checkpoint_path=checkpoint_path,
    )


def _stage_production(
    config: RunConfig,
    system: Any,
    stage_dir: Path,
    *,
    checkpoint_path: Path | None = None,
) -> Any:
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
    return run_production(
        sim_time,
        system,
        work_dir=stage_dir,
        params=config.md,
        checkpoint_path=checkpoint_path,
    )


def _stage_mmbsa(
    config: RunConfig,
    system: Any,
    production_dir: Path,
    stage_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Run gmx_MMPBSA on the production trajectory.

    ``production_dir`` is Stage 8's own output directory: BSS.Process.Gromacs
    writes its raw GROMACS files there as ``gromacs.tpr``/``gromacs.xtc``/
    ``gromacs.top`` -- gmx_MMPBSA needs these, not the ``system.gro``/``.top``
    snapshot ``_run_md_stage`` separately re-exports.

    The ligand molecule itself is still identified positionally (protein
    first, ligand second for a ``[system]`` run -- ``parametrize()``'s own
    convention; ligand at ``n_protein_molecules`` for a ``[membrane]`` run,
    per :func:`extract_protein_ligand_system`), but the Receptor/Ligand atom
    *index* selection passed to gmx_MMPBSA is then computed from the ligand's
    moleculetype name against the actual topology file gmx_MMPBSA itself will
    load, via :func:`~gbsa_pipeline.gromacs_index.select_receptor_and_ligand_atoms`
    -- the same GROMACS moleculetype identity gmx_MMPBSA's own topology
    cleaning uses, so no molecule-ordering assumption is needed for that step.

    For a ``[membrane]`` run, gmx_MMPBSA instead runs against a *reduced*
    protein+ligand-only system from :func:`extract_protein_ligand_system`
    (see its docstring for why) -- membrane geometry is computed separately
    from the original unstripped structure, so nothing is lost.
    """
    index_file = stage_dir / "index.ndx"

    if config.system.membrane:
        prebuilt = BSS.IO.readMolecules(
            [str(config.system.gro_file), str(config.system.top_file)],
            make_whole=True,
        )
        membrane_cfg = config.membrane or MembraneConfig()
        lipid_resnames = set(membrane_cfg.lipid_resnames)

        n_solute_molecules = prebuilt.nMolecules()
        prebuilt_mols = prebuilt.getMolecules()
        n_protein_molecules = sum(1 for mol in prebuilt_mols if is_protein_molecule(mol))

        universe = mda.Universe(str(production_dir / "system.gro"))
        geometry = estimate_membrane_geometry(universe, lipid_resnames=sorted(lipid_resnames))

        reduced_system = extract_protein_ligand_system(system, n_solute_molecules, n_protein_molecules)
        complex_prefix = stage_dir / "complex"
        _, top_file = save_bss_system_to_gromacs(reduced_system, complex_prefix)
        BSS.IO.saveMolecules(str(complex_prefix), reduced_system, fileformat="pdb")
        structure_pdb = complex_prefix.with_suffix(".pdb")
        trajectory_pdb = stage_dir / "complex_traj.pdb"
        shutil.copy(structure_pdb, trajectory_pdb)

        reduced_sire = reduced_system._sire_object
        ligand_mol = list(reduced_sire)[n_protein_molecules]
        ligand_moltype = ligand_mol.residues()[0].name().value()
        receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms(top_file, ligand_moltype)
        write_index(receptor_atoms, ligand_atoms, index_file)

        mmpbsa_config = MMPBSAConfig(gb=None, pb=geometry.pb_params())
        input_file = mmpbsa_config.write(stage_dir / "mmpbsa.in")

        return run_gmx_mmpbsa_from_gromacs(
            input_file=input_file,
            complex_structure=structure_pdb,
            trajectory=trajectory_pdb,
            topology=top_file,
            index_file=index_file,
            receptor_group=0,
            ligand_group=1,
            output_dir=stage_dir,
        )

    production_sire = system._sire_object
    ligand_mol = list(production_sire)[1]
    ligand_moltype = ligand_mol.residues()[0].name().value()
    receptor_atoms, ligand_atoms = select_receptor_and_ligand_atoms(production_dir / "gromacs.top", ligand_moltype)
    write_index(receptor_atoms, ligand_atoms, index_file)
    mmpbsa_config = MMPBSAConfig()
    input_file = mmpbsa_config.write(stage_dir / "mmpbsa.in")

    return run_gmx_mmpbsa_from_gromacs(
        input_file=input_file,
        complex_structure=production_dir / "gromacs.tpr",
        trajectory=production_dir / "gromacs.xtc",
        topology=production_dir / "gromacs.top",
        index_file=index_file,
        receptor_group=0,
        ligand_group=1,
        output_dir=stage_dir,
    )


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
    9. **GBSA** — gmx_MMPBSA on the production trajectory.

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

    if config.system.membrane:
        # Stage 1: Parametrize ligand + merge into pre-built membrane system
        logger.info("─── Stage 1/9: Ligand parametrization + membrane merge ───")
        param_dir = output_dir / "01_parametrize"
        param_dir.mkdir(parents=True, exist_ok=True)
        system = _run_stage(
            "parametrize_membrane",
            lambda: _stage_parametrize_membrane(config, param_dir),
        )

        # Stage 2: Solvate (membrane-aware, or skip if already solvated)
        logger.info("─── Stage 2/9: Membrane solvation ───")
        sol_dir = output_dir / "02_solvated"
        sol_dir.mkdir(parents=True, exist_ok=True)
        system = _run_stage(
            "solvate_membrane",
            lambda: _stage_solvate_membrane(config, system, sol_dir),
        )
    else:
        # Stage 1: Parametrize
        logger.info("─── Stage 1/9: Parametrization ───")
        param_dir = output_dir / "01_parametrize"
        parametrized = _run_stage("parametrize", lambda: _stage_parametrize(config, param_dir))
        logger.info("  Done → %s, %s", parametrized.gro_file.name, parametrized.top_file.name)

        # Stage 2: Solvate
        logger.info("─── Stage 2/9: Solvation ───")
        sol_dir = output_dir / "02_solvated"
        system = _run_stage("solvation", lambda: _stage_solvate(config, parametrized, sol_dir))

    system = _run_md_stage(
        "Stage 3/9: SD Minimization",
        "sd_minimization",
        "03_sd",
        output_dir,
        lambda d: _stage_minimize_sd(config, system, d),
    )
    system = _run_md_stage(
        "Stage 4/9: CG Minimization",
        "cg_minimization",
        "04_cg",
        output_dir,
        lambda d: _stage_minimize_cg(system, d),
    )
    restraint = _membrane_aware_restraint_selection(config, system)
    system = _run_md_stage(
        "Stage 5/9: NVT Restrained Heating",
        "nvt_restrained",
        "05_nvt_res",
        output_dir,
        lambda d: _stage_nvt_restrained(config, system, d, restraint=restraint),
    )
    system = _run_md_stage(
        "Stage 6/9: NPT Restrained Equilibration",
        "npt_restrained",
        "06_npt_res",
        output_dir,
        lambda d: _stage_npt(
            config,
            system,
            d,
            restraint=restraint,
            checkpoint_path=output_dir / "05_nvt_res" / "gromacs.cpt",
        ),
    )
    system = _run_md_stage(
        "Stage 7/9: NPT Equilibration",
        "npt",
        "07_npt",
        output_dir,
        lambda d: _stage_npt(
            config,
            system,
            d,
            checkpoint_path=output_dir / "06_npt_res" / "gromacs.cpt",
        ),
    )
    system = _run_md_stage(
        "Stage 8/9: Production MD",
        "production_md",
        "08_production",
        output_dir,
        lambda d: _stage_production(
            config,
            system,
            d,
            checkpoint_path=output_dir / "07_npt" / "gromacs.cpt",
        ),
    )
    logger.info("─── Stage 9/9: GBSA (gmx_MMPBSA) ───")
    mmbsa_dir = output_dir / "09_mmbsa"
    mmbsa_dir.mkdir(parents=True, exist_ok=True)
    _run_stage(
        "mmbsa",
        lambda: _stage_mmbsa(config, system, output_dir / "08_production", mmbsa_dir),
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
