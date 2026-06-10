"""Pipeline integration tests for 1F0R and 3K5E.

Each test runs the complete data-preparation workflow:
docking with crystal-water selection → ligand pose export →
force-field parametrization → solvation.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from gbsa_pipeline.docking import (
    DockingBox,
    DockingRequest,
    VinaEngine,
    dock_with_and_without_crystal_waters,
    export_pdbqt_to_sdf,
    load_first_sdf_molecule,
    prepare_ligand_with_meeko,
)
from gbsa_pipeline.parametrization import ParametrizationConfig, ParametrizationInput, parametrize
from gbsa_pipeline.solvation_box import BoxShape, SolvationParams
from gbsa_pipeline.solvation_bss import solvate_bss

TESTDATA = Path(__file__).parents[1] / "testdata" / "challenging_data"


def _run_pipeline(
    tmp_path: Path,
    protein_pdb: Path,
    crystal_waters_pdb: Path,
    ligand_sdf: Path,
    box: DockingBox,
    extra_ff_files: tuple[Path, ...] = (),
) -> None:
    dock_dir = tmp_path / "docking"
    param_dir = tmp_path / "parametrize"
    solv_dir = tmp_path / "solvate"
    for d in (dock_dir, param_dir, solv_dir):
        d.mkdir(parents=True, exist_ok=True)

    ligand_mol = load_first_sdf_molecule(ligand_sdf, remove_hs=False)
    ligand_pdbqt = dock_dir / "ligand.pdbqt"
    prepare_ligand_with_meeko(ligand_mol, ligand_pdbqt, name="LIG")
    assert ligand_pdbqt.exists()

    engine = VinaEngine()
    request = DockingRequest(
        receptor=protein_pdb,
        ligands=[ligand_pdbqt],
        box=box,
        workdir=dock_dir,
        parameters={"exhaustiveness": 4, "num_modes": 3},
    )
    manifest = dock_with_and_without_crystal_waters(
        engine=engine,
        request=request,
        crystal_waters_pdb=crystal_waters_pdb,
        ligand_sdf=ligand_sdf,
        work_dir=dock_dir,
    )

    assert manifest.without_waters is not None
    assert manifest.without_waters.poses
    assert manifest.score_without is not None

    best = (
        manifest.with_waters
        if manifest.score_with is not None
        and (manifest.score_without is None or manifest.score_with < manifest.score_without)
        else manifest.without_waters
    )
    assert best is not None
    assert best.poses

    docked_sdf = dock_dir / "docked_ligand.sdf"
    export_pdbqt_to_sdf(
        best.poses[0].pose_path,
        docked_sdf,
        template_mol=ligand_mol,
        add_hydrogens_after_template=True,
    )
    assert docked_sdf.exists()

    parametrized = parametrize(
        ParametrizationInput(
            protein_pdb=protein_pdb,
            ligand_sdf=docked_sdf,
            config=ParametrizationConfig(extra_ff_files=extra_ff_files),
            work_dir=param_dir,
        )
    )
    assert parametrized.gro_file.exists()
    assert parametrized.top_file.exists()

    os.environ.setdefault("GMX_MAXCONSTRWARN", "-1")
    solvated = solvate_bss(
        parametrized,
        params=SolvationParams(
            water_model="tip3p",
            shape=BoxShape.TRUNCATED_OCTAHEDRON,
            padding=1.0,
        ),
        output_gro=solv_dir / "solvated.gro",
        output_top=solv_dir / "solvated.top",
    )
    assert solvated.gro_file.exists()
    assert solvated.top_file.exists()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_system_1f0r_docking_parametrize_solvate(tmp_path: Path) -> None:
    """1F0R: dock with crystal-water selection, parametrize, solvate."""
    if shutil.which("vina") is None:
        pytest.skip("vina not found in PATH")

    _run_pipeline(
        tmp_path,
        protein_pdb=TESTDATA / "1F0R" / "1F0R.pdb",
        crystal_waters_pdb=TESTDATA / "1F0R" / "1F0R_WAT.pdb",
        ligand_sdf=TESTDATA / "1F0R" / "ligands" / "ligand01.sdf",
        box=DockingBox(center=(23.1, 13.3, 25.2), size=(35.0, 35.0, 35.0)),
    )


@pytest.mark.integration
def test_system_3k5e_docking_parametrize_solvate(tmp_path: Path) -> None:
    """3K5E: dock with crystal-water selection, parametrize, solvate."""
    if shutil.which("vina") is None:
        pytest.skip("vina not found in PATH")

    _run_pipeline(
        tmp_path,
        protein_pdb=TESTDATA / "3K5E" / "3K5E.pdb",
        crystal_waters_pdb=TESTDATA / "3K5E" / "3K5E_WAT.pdb",
        ligand_sdf=TESTDATA / "3K5E" / "ligands" / "ligand01.sdf",
        box=DockingBox(center=(20.6, 18.7, 19.7), size=(30.0, 30.0, 30.0)),
    )
