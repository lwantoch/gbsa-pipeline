"""Integration tests for module-level workflow connectivity.

This module exercises a full inspectable workflow across docking, ligand pose
export, parametrization, OpenMM solvation, and staged BioSimSpace/GROMACS MD
helpers. Runtime artefacts are written to pytest temporary directories so test
runs do not leave generated files under the repository tree. The assertions are
still smoke-level checks because detailed physical validation belongs in smaller
module-specific tests or curated manual benchmark runs. The purpose here is to
verify that the public module interfaces can still be connected end to end.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import BioSimSpace as BSS
import pytest

from gbsa_pipeline.docking import (
    DockingBox,
    DockingManifest,
    DockingRequest,
    VinaEngine,
    convert_receptor_pdb_to_pdbqt,
    dock_with_and_without_crystal_waters,
    export_pdbqt_to_sdf,
    load_first_sdf_molecule,
    prepare_ligand_with_meeko,
)
from gbsa_pipeline.gromacs_index import write_index_from_system
from gbsa_pipeline.md import (
    run_heating,
    run_minimization,
    run_npt_equilibration,
    run_production,
)
from gbsa_pipeline.mmbsa import MMPBSAConfig, run_gmx_mmpbsa_from_gromacs
from gbsa_pipeline.parametrization import (
    ParametrizationInput,
    parametrize,
)
from gbsa_pipeline.solvation_bss import solvate_parametrized_complex

TESTDATA = Path(__file__).parents[1] / "testdata"
DOCKING_TESTDATA = TESTDATA / "docking"
DOCKLIGAND_SDF = DOCKING_TESTDATA / "dockligand.sdf"
DOCKPROTEIN_PDB = DOCKING_TESTDATA / "dockprotein.pdb"
PARAMETRIZE_PROTEIN_PDB = DOCKING_TESTDATA / "dockprotein_crystal_waters.pdb"

MEEKO_RECEPTOR_BINARY = "mk_prepare_receptor.py"

DOCKPROTEIN_BOX = DockingBox(
    center=(-2.215, -0.043, 25.120),
    size=(17.0, 17.0, 17.0),
)

SD_PARAMS: dict[str, Any] = {
    "integrator": "steep",
    "nsteps": 100000,
    "emtol": 100.0,
    "emstep": 0.001,
    "cutoff_scheme": "Verlet",
    "nstlist": 20,
    "pbc": "xyz",
    "rlist": 1.26,
    "coulombtype": "PME",
    "rcoulomb": 1.2,
    "fourierspacing": 0.16,
    "pme_order": 4,
    "ewald_rtol": 1e-5,
    "vdwtype": "Cut-off",
    "vdw_modifier": "Force-switch",
    "rvdw_switch": 1.0,
    "rvdw": 1.2,
    "constraints": "none",
    "tcoupl": "no",
    "pcoupl": "no",
    "gen_vel": "no",
    "nstlog": 500,
    "nstenergy": 500,
    "nstxout_compressed": 0,
}

CG_PARAMS: dict[str, Any] = {
    "integrator": "cg",
    "nsteps": 50000,
    "emtol": 50.0,
    "emstep": 0.001,
    "nstcgsteep": 50,
    "cutoff_scheme": "Verlet",
    "nstlist": 20,
    "pbc": "xyz",
    "rlist": 1.221,
    "coulombtype": "PME",
    "rcoulomb": 1.2,
    "fourierspacing": 0.16,
    "pme_order": 4,
    "ewald_rtol": 1e-5,
    "vdwtype": "Cut-off",
    "vdw_modifier": "Force-switch",
    "rvdw_switch": 1.0,
    "rvdw": 1.2,
    "constraints": "none",
    "tcoupl": "no",
    "pcoupl": "no",
    "gen_vel": "no",
    "nstlog": 500,
    "nstenergy": 500,
    "nstxout_compressed": 0,
}

SOLVENT_RELAX_PARAMS: dict[str, Any] = {
    "integrator": "sd",
    "dt": 0.001,
    "nsteps": 2000,
    "tcoupl": "no",
    "mts": "no",
    "lincs_warnangle": 90.0,
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "lincs_order": 4,
    "gen_vel": "yes",
    "gen_temp": 300.0,
    "define": "-DPOSRES",
    "nstlog": 200,
    "nstenergy": 200,
    "nstcalcenergy": 100,
    "nstxout_compressed": 0,
}

HEATING_PARAMS: dict[str, Any] = {
    "integrator": "md",
    "dt": 0.001,
    "nsteps": 100000,
    "cutoff_scheme": "Verlet",
    "nstlist": 20,
    "pbc": "xyz",
    "rlist": 1.221,
    "coulombtype": "PME",
    "rcoulomb": 1.2,
    "fourierspacing": 0.16,
    "pme_order": 4,
    "ewald_rtol": 1e-5,
    "vdwtype": "Cut-off",
    "vdw_modifier": "Force-switch",
    "rvdw_switch": 1.0,
    "rvdw": 1.2,
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "lincs_order": 4,
    "lincs_warnangle": 90.0,
    "continuation": "no",
    "gen_vel": "yes",
    "gen_temp": 50.0,
    "tcoupl": "V-rescale",
    "tc_grps": "System",
    "tau_t": 0.1,
    "ref_t": 300.0,
    "pcoupl": "no",
    "annealing": "single",
    "annealing_npoints": 6,
    "annealing_time": "0 20 40 60 80 100",
    "annealing_temp": "50 100 150 200 250 300",
    "define": "-DPOSRES",
    "mts": "no",
    "nstlog": 500,
    "nstenergy": 500,
    "nstcalcenergy": 100,
    "nstxout_compressed": 1000,
}

NPT_RESTRAINED_PARAMS: dict[str, Any] = {
    "integrator": "md",
    "dt": 0.002,
    # Recommended: 100-200 ps (Lemkul 2024, DOI:10.1021/acs.jpcb.4c04901;
    # gmx_MMPBSA ST example); halved here for test speed.
    "nsteps": 50000,
    "cutoff_scheme": "Verlet",
    "nstlist": 20,
    "pbc": "xyz",
    "rlist": 1.221,
    "coulombtype": "PME",
    "rcoulomb": 1.2,
    "fourierspacing": 0.16,
    "pme_order": 4,
    "ewald_rtol": 1e-5,
    "vdwtype": "Cut-off",
    "vdw_modifier": "Force-switch",
    "rvdw_switch": 1.0,
    "rvdw": 1.2,
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "lincs_order": 4,
    "lincs_warnangle": 90.0,  # GROMACS manual: 90° recommended for poorly-minimised/strained starting structures
    # continuation and gen_vel are NOT set here: run_npt_equilibration receives
    # checkpoint_path=nvt_restrained_dir/"gromacs.cpt", which makes md.py inject
    # continuation=yes and gen_vel=no automatically so GROMACS reads velocities
    # from the NVT checkpoint rather than regenerating them.
    "tcoupl": "V-rescale",
    "tc_grps": "System",
    "tau_t": 0.1,
    "ref_t": 300.0,
    "pcoupl": "C-rescale",
    "pcoupltype": "isotropic",
    "tau_p": 5.0,
    "ref_p": 1.0,
    "compressibility": 4.5e-5,
    "refcoord_scaling": "com",
    "define": "-DPOSRES",
    "nstlog": 500,
    "nstenergy": 500,
    "nstcalcenergy": 100,
    "nstxout_compressed": 1000,
}

NPT_PARAMS: dict[str, Any] = {
    "integrator": "md",
    "dt": 0.002,
    # Recommended: 200 ps; shorter runs can leave the system under-equilibrated and
    # cause production instabilities (observed in pytest-37). Halved for test speed.
    "nsteps": 50000,
    "cutoff_scheme": "Verlet",
    "nstlist": 20,
    "pbc": "xyz",
    "rlist": 1.221,
    "coulombtype": "PME",
    "rcoulomb": 1.2,
    "fourierspacing": 0.16,
    "pme_order": 4,
    "ewald_rtol": 1e-5,
    "vdwtype": "Cut-off",
    "vdw_modifier": "Force-switch",
    "rvdw_switch": 1.0,
    "rvdw": 1.2,
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "lincs_order": 4,
    "lincs_warnangle": 90.0,  # GROMACS manual: 90° recommended for poorly-minimised/strained starting structures
    # continuation and gen_vel are NOT set here: checkpoint_path handles them.
    "tcoupl": "V-rescale",
    "tc_grps": "System",
    "tau_t": 0.1,
    "ref_t": 300.0,
    "pcoupl": "C-rescale",
    "pcoupltype": "isotropic",
    "tau_p": 5.0,
    "ref_p": 1.0,
    "compressibility": 4.5e-5,
    "define": "",
    "nstlog": 500,
    "nstenergy": 500,
    "nstcalcenergy": 100,
    "nstxout_compressed": 1000,
}

PRODUCTION_PARAMS: dict[str, Any] = {
    "integrator": "md",
    "dt": 0.002,
    # Recommended: 200+ ps for GBSA averaging (Genheden & Ryde 2010,
    # DOI:10.1002/jcc.21366; gmx_MMPBSA docs recommend >=10-100 frames;
    # 200 ps at nstxout_compressed=200 gives 100 frames). Halved for test speed.
    "nsteps": 50000,
    "cutoff_scheme": "Verlet",
    "nstlist": 20,
    "pbc": "xyz",
    "rlist": 1.221,
    "coulombtype": "PME",
    "rcoulomb": 1.2,
    "fourierspacing": 0.16,
    "pme_order": 4,
    "ewald_rtol": 1e-5,
    "vdwtype": "Cut-off",
    "vdw_modifier": "Force-switch",
    "rvdw_switch": 1.0,
    "rvdw": 1.2,
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "lincs_order": 4,
    "lincs_warnangle": 90.0,  # GROMACS manual: 90° recommended for poorly-minimised/strained starting structures
    # continuation and gen_vel are NOT set here: run_production receives
    # checkpoint_path=npt_unrestrained_dir/"gromacs.cpt", which makes md.py inject
    # continuation=yes and gen_vel=no so GROMACS reads velocities from the NPT
    # checkpoint.  Observed failure without this: SETTLE crash at step 568 when
    # gen_vel=yes drew fresh Maxwell-Boltzmann velocities onto residual bad contacts.
    "tcoupl": "V-rescale",
    "tc_grps": "System",
    "tau_t": 0.1,
    "ref_t": 300.0,
    "pcoupl": "C-rescale",
    "pcoupltype": "isotropic",
    "tau_p": 5.0,
    "ref_p": 1.0,
    "compressibility": 4.5e-5,
    "define": "",
    "nstlog": 500,
    "nstenergy": 500,
    "nstcalcenergy": 100,
    # 100 frames: 100 000 steps / nstxout_compressed=1000 = 100 frames.
    "nstxout_compressed": 1000,
}


@pytest.fixture
def module_run_dir(tmp_path: Path) -> Path:
    """Return a disposable output directory for one module workflow test.

    The workflow writes many intermediate files from docking, parametrization,
    solvation, and MD stages. Using ``tmp_path`` keeps those artefacts out of the
    repository tree and avoids accidental commits of generated outputs. The
    dedicated subdirectory keeps all products from this test under one inspectable
    location during the pytest run. Pytest removes the directory according to its
    normal temporary-directory policy.
    """
    run_dir = tmp_path / "module_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.mark.integration
def test_prepare_inputs_run_docking_parametrize_and_solvate_keeps_outputs(
    module_run_dir: Path,
) -> None:
    """Run docking, solvation, minimization, equilibration, production, and GBSA.

    This test is a workflow smoke test, not a detailed unit test of the docking,
    parametrization, solvation, or MD helper modules. The ligand starts from SDF,
    the receptor starts from PDB, and the docking output is exported back to SDF
    before it is passed into the parametrization entry point. The parametrized
    complex is solvated with retained crystallographic waters restored before
    bulk solvent placement, then loaded into BioSimSpace. The MD bridge runs the
    exact six stage parameter blocks: SD, CG, restrained NVT heating, restrained
    NPT, unrestrained NPT, and production. The final stage runs gmx_MMPBSA in
    GB-only mode using the GAFF-parametrized topology and production trajectory
    produced by the preceding stages; because the ligand was fully parametrized
    upstream, sander receives valid force field parameters and the energy
    calculation completes without overflow. All output files land in the
    pytest-managed tmp directory printed at the start of the test.
    """
    print(f"\n[module_run_dir] {module_run_dir}", flush=True)  # noqa: T201
    if shutil.which("vina") is None:
        pytest.skip("vina not available in PATH")

    if shutil.which(MEEKO_RECEPTOR_BINARY) is None:
        pytest.skip(f"{MEEKO_RECEPTOR_BINARY} not available in PATH")

    if not DOCKLIGAND_SDF.exists():
        pytest.skip(f"missing ligand test file: {DOCKLIGAND_SDF}")

    if not DOCKPROTEIN_PDB.exists():
        pytest.skip(f"missing receptor test file: {DOCKPROTEIN_PDB}")

    ligand_pdbqt = module_run_dir / "dockligand.pdbqt"
    receptor_pdbqt = module_run_dir / "dockprotein.pdbqt"
    docked_sdf = module_run_dir / "dockligand_vina_out.sdf"

    docking_dir = module_run_dir / "docking"
    parametrization_dir = module_run_dir / "parametrization"
    solvation_dir = module_run_dir / "solvation"
    sd_minimization_dir = module_run_dir / "sd"
    cg_minimization_dir = module_run_dir / "cg"
    nvt_restrained_dir = module_run_dir / "nvt_res"
    npt_restrained_dir = module_run_dir / "npt_res"
    npt_unrestrained_dir = module_run_dir / "npt"
    production_dir = module_run_dir / "production"

    for _d in (
        docking_dir,
        parametrization_dir,
        solvation_dir,
        sd_minimization_dir,
        cg_minimization_dir,
        nvt_restrained_dir,
        npt_restrained_dir,
        npt_unrestrained_dir,
        production_dir,
    ):
        _d.mkdir(parents=True, exist_ok=True)

    ligand_molecule = load_first_sdf_molecule(DOCKLIGAND_SDF, remove_hs=False)

    prepared_ligand = prepare_ligand_with_meeko(
        ligand_molecule,
        ligand_pdbqt,
        name="DOCKLIG",
    )
    prepared_receptor = convert_receptor_pdb_to_pdbqt(
        DOCKPROTEIN_PDB,
        output_path=receptor_pdbqt,
        mk_prepare_receptor_binary=MEEKO_RECEPTOR_BINARY,
    )

    assert prepared_ligand == ligand_pdbqt
    assert prepared_receptor == receptor_pdbqt
    assert ligand_pdbqt.exists()
    assert receptor_pdbqt.exists()

    engine = VinaEngine(
        binary="vina",
        mk_prepare_receptor_binary=MEEKO_RECEPTOR_BINARY,
    )
    # Use PDB receptor so dock_with_and_without_crystal_waters can merge crystal
    # waters into a receptor+waters PDB for the second docking run.
    request = DockingRequest(
        receptor=DOCKPROTEIN_PDB,
        ligands=[ligand_pdbqt],
        box=DOCKPROTEIN_BOX,
        workdir=docking_dir,
    )

    manifest: DockingManifest = dock_with_and_without_crystal_waters(
        engine,
        request,
        crystal_waters_pdb=PARAMETRIZE_PROTEIN_PDB,
        ligand_sdf=DOCKLIGAND_SDF,
        work_dir=docking_dir,
    )

    assert manifest.without_waters.engine == "vina"
    assert len(manifest.without_waters.poses) == 1
    assert manifest.score_without is not None
    assert isinstance(manifest.retained_water_ids, list)

    # Prefer the with-water pose: crystal waters in the binding site enforce
    # the correct geometry even when the score difference is within Vina noise.
    assert manifest.with_waters is not None, "with-water docking produced no result"
    assert manifest.with_waters.engine == "vina"
    best_pose = manifest.with_waters.poses[0]
    assert best_pose.pose_path.exists()
    assert best_pose.metadata["returncode"] == 0
    assert best_pose.score is not None
    assert manifest.score_with is not None

    docking_output = best_pose.pose_path

    exported_sdf = export_pdbqt_to_sdf(
        docking_output,
        docked_sdf,
        template_mol=ligand_molecule,
        add_hydrogens_after_template=True,
    )

    assert exported_sdf == docked_sdf
    assert docked_sdf.exists()

    parametrized = parametrize(
        ParametrizationInput(
            protein_pdb=PARAMETRIZE_PROTEIN_PDB,
            ligand_sdf=docked_sdf,
            work_dir=parametrization_dir,
        )
    )

    crystal_waters_pdb = parametrization_dir / "crystal_waters.pdb"
    if parametrized.crystal_waters_pdb is not None:
        assert crystal_waters_pdb.exists()
        assert crystal_waters_pdb.read_text(encoding="utf-8").strip()
        assert parametrized.crystal_waters_pdb == crystal_waters_pdb
    assert parametrized.gro_file == parametrization_dir / "complex.gro"
    assert parametrized.top_file == parametrization_dir / "complex.top"
    assert parametrized.gro_file.exists()
    assert parametrized.top_file.exists()

    # ==========================================================================
    # SOLVATION — BSS.Solvent (gmx solvate) over solvate_openmm
    # ==========================================================================
    # We use solvation_bss.solvate_parametrized_complex instead of
    # solvation_openmm.solvate_openmm because the OpenMM + ParmEd path
    # produces explicit O-H harmonic bond springs in the water topology
    # (ParmEd rigidWater=False), which have a force constant ~727 000x
    # larger than the TIP3P O-O LJ well depth.  Every minimization step
    # corrects water geometry instead of resolving LJ clashes, causing
    # LJ (SR) to increase from +112 000 to +175 000 kJ/mol across SD + CG,
    # and LINCS to fail early in NVT heating.  BSS.Solvent wraps gmx solvate,
    # which writes a SETTLE-constrained [ settles ] topology from the start.
    # See solvation_bss.py module docstring for the full analysis with measured
    # LJ (SR) values and references.
    # ==========================================================================

    os.environ["GMX_MAXCONSTRWARN"] = "-1"

    bss_system = solvate_parametrized_complex(
        parametrized,
        shell_nm=1.0,
        work_dir=solvation_dir,
    )

    assert bss_system is not None

    sd_minimized = run_minimization(
        bss_system,
        work_dir=sd_minimization_dir,
        params=SD_PARAMS,
    )

    assert sd_minimized is not None
    assert any(sd_minimization_dir.iterdir())

    cg_minimized = run_minimization(
        sd_minimized,
        work_dir=cg_minimization_dir,
        params=CG_PARAMS,
    )

    assert cg_minimized is not None
    assert any(cg_minimization_dir.iterdir())

    # BSS protocol runtime must match HEATING_PARAMS nsteps x dt to prevent BSS
    # from timing out before the MDP run finishes and passing a truncated
    # checkpoint to the next stage (observed in pytest-37 with the 12 nm box).
    # Checkpoint continuity: NVT writes gromacs.cpt, which NPT restrained reads
    # via grompp -t.
    nvt_restrained = run_heating(
        100 * BSS.Units.Time.picosecond,
        cg_minimized,
        work_dir=nvt_restrained_dir,
        params=HEATING_PARAMS,
        temperature_start=50 * BSS.Units.Temperature.kelvin,
        temperature_end=300 * BSS.Units.Temperature.kelvin,
        restraint="backbone",
    )

    assert nvt_restrained is not None
    assert any(nvt_restrained_dir.iterdir())

    nvt_res_cpt = nvt_restrained_dir / "gromacs.cpt"
    npt_restrained = run_npt_equilibration(
        100 * BSS.Units.Time.picosecond,
        nvt_restrained,
        work_dir=npt_restrained_dir,
        params=NPT_RESTRAINED_PARAMS,
        restraint="backbone",
        checkpoint_path=nvt_res_cpt if nvt_res_cpt.exists() else None,
    )

    assert npt_restrained is not None
    assert any(npt_restrained_dir.iterdir())

    npt_res_cpt = npt_restrained_dir / "gromacs.cpt"
    npt_unrestrained = run_npt_equilibration(
        100 * BSS.Units.Time.picosecond,
        npt_restrained,
        work_dir=npt_unrestrained_dir,
        params=NPT_PARAMS,
        restraint=None,
        checkpoint_path=npt_res_cpt if npt_res_cpt.exists() else None,
    )

    assert npt_unrestrained is not None
    assert any(npt_unrestrained_dir.iterdir())

    npt_cpt = npt_unrestrained_dir / "gromacs.cpt"
    production = run_production(
        100 * BSS.Units.Time.picosecond,
        npt_unrestrained,
        work_dir=production_dir,
        params=PRODUCTION_PARAMS,
        checkpoint_path=npt_cpt if npt_cpt.exists() else None,
    )

    assert production is not None
    assert any(production_dir.iterdir())

    # ── GBSA ─────────────────────────────────────────────────────────────────
    # gmx_MMPBSA is optional — skip gracefully if not installed.
    if shutil.which("gmx_MMPBSA") is None:
        pytest.skip("gmx_MMPBSA not available in PATH — skipping GBSA stage")

    gbsa_dir = module_run_dir / "gbsa"
    gbsa_dir.mkdir(parents=True, exist_ok=True)

    # BSS writes gromacs.tpr, gromacs.xtc, and gromacs.top into production_dir.
    # The .tpr is required by gmx_MMPBSA for -cs (it rejects .gro).
    complex_tpr = production_dir / "gromacs.tpr"
    trajectory_xtc = production_dir / "gromacs.xtc"
    topology_top = production_dir / "gromacs.top"

    # Build the GROMACS index file from the production sire system.
    # Molecule ordering after parametrization + solvation:
    #   index 0 → protein, index 1 → GAFF ligand, remainder → water and ions.
    production_sire = production._sire_object
    production_molecules = list(production_sire)
    protein_mol = production_molecules[0]
    ligand_mol = production_molecules[1]

    index_file = gbsa_dir / "index.ndx"
    write_index_from_system(production_sire, protein_mol, ligand_mol, index_file)
    assert index_file.exists()

    mmpbsa_input = gbsa_dir / "mmpbsa.in"
    # saltcon=0.15 M: physiological ionic strength; recommended by gmx_MMPBSA docs
    # and Genheden & Ryde 2010 (DOI:10.1002/jcc.21366).
    # igb=5 (GB-OBC2): most widely recommended GB model for protein-ligand systems
    # per gmx_MMPBSA documentation and the 2018 AMBER tutorial.
    from gbsa_pipeline.mmbsa import GBParams  # noqa: PLC0415

    MMPBSAConfig(pb=None, gb=GBParams(igb=5, saltcon=0.15)).write(mmpbsa_input)

    mmpbsa_result = run_gmx_mmpbsa_from_gromacs(
        input_file=mmpbsa_input,
        complex_structure=complex_tpr,
        trajectory=trajectory_xtc,
        topology=topology_top,
        index_file=index_file,
        receptor_group=0,
        ligand_group=1,
        output_dir=gbsa_dir / "run",
    )

    assert mmpbsa_result.returncode == 0, (
        f"gmx_MMPBSA failed (rc={mmpbsa_result.returncode}):\n{mmpbsa_result.stderr[-3000:]}"
    )
    assert any((gbsa_dir / "run").iterdir())
