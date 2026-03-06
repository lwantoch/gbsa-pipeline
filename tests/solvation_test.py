from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import BioSimSpace as BSS
import parmed as pmd
import pytest

from gbsa_pipeline.parametrization import (
    export_gromacs_top_gro,
    load_protein_pdb,
    parameterise_protein_amber,
)
from gbsa_pipeline.solvation_box import (
    BoxShape,
    SolvationParams,
    WaterModel,
    box_parameters,
    run_solvation,
)
from gbsa_pipeline.solvation_openmm import solvate_openmm


@pytest.mark.integration
def test_solvation() -> None:
    # Load raw protein
    mol = load_protein_pdb("tests/testdata/test1.pdb")

    # Parameterise -> returns Molecule
    mol_param = parameterise_protein_amber(mol, ff="ff14SB")

    # Create System explicitly
    system = BSS._SireWrappers.System(mol_param)
    n_atoms_before = system.nAtoms()

    # Solvate (assumed in-place)
    solvated = run_solvation(
        system=system,
        params=SolvationParams(
            water_model=WaterModel.TIP3P, box_size=8, ion_concentration=0.0
        ),
    )

    # Check that solvent was added
    assert solvated.nAtoms() > n_atoms_before


def test_box_shape_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_cubic(length: object) -> tuple[str, str]:
        called["shape"] = "cubic"
        return "box", "angles"

    monkeypatch.setattr(BSS.Box, "cubic", fake_cubic)

    params = SolvationParams(shape=BoxShape.CUBIC, box_size=5.0)
    box, angles = params.box()

    assert called["shape"] == "cubic"
    assert box == "box"
    assert angles == "angles"


def test_padding_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_solvate(model: object, **kwargs: object) -> str:
        captured["model"] = model
        captured.update(kwargs)
        return "solvated"

    monkeypatch.setattr(BSS.Solvent, "solvate", fake_solvate)

    params = SolvationParams(padding=1.5, box_size=20.0)
    dummy_system = cast("BSS._SireWrappers.System", object())
    out = run_solvation(system=dummy_system, params=params)

    assert out == "solvated"
    assert "shell" in captured
    padding = cast("Any", captured["shell"])
    padding_nm = padding / BSS.Units.Length.nanometer
    assert pytest.approx(padding_nm) == 1.5
    assert "box" not in captured


def test_run_solvation_box_vs_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_solvate(model: object, **kwargs: object) -> str:
        captured["model"] = model
        captured.update(kwargs)
        return "solvated"

    monkeypatch.setattr(BSS.Solvent, "solvate", fake_solvate)

    # Box-size path
    params_box = SolvationParams(box_size=5.0, water_model=WaterModel.SPC)
    out_box = run_solvation(
        system=cast("BSS._SireWrappers.System", object()), params=params_box
    )

    assert out_box == "solvated"
    assert captured["model"] == WaterModel.SPC.value
    assert "box" in captured
    assert "angles" in captured
    assert "shell" not in captured

    # Shell path overrides box
    captured.clear()
    params_shell = SolvationParams(
        padding=1.0, box_size=8.0, water_model=WaterModel.TIP4P
    )
    out_shell = run_solvation(
        system=cast("BSS._SireWrappers.System", object()), params=params_shell
    )

    assert out_shell == "solvated"
    assert captured["model"] == WaterModel.TIP4P.value
    assert "shell" in captured
    assert "box" not in captured
    assert "angles" not in captured


def test_water_model_validation() -> None:
    with pytest.raises(ValueError):
        SolvationParams(water_model="unsupported").solvent_builder()


def test_box_parameters_validation() -> None:
    with pytest.raises(ValueError):
        box_parameters(5.0, shape="invalid")


WATER_RESNAMES = {"SOL", "WAT", "HOH", "TIP3", "TIP3P"}


@pytest.mark.integration
def test_openmm_preparametrized(tmp_path: Path) -> None:
    testdata = (
        Path(__file__).resolve().parent / "testdata" / "solvation" / "complex.pickle"
    )
    with testdata.open("rb") as f:
        complex_obj = pickle.load(f)

    out_gro = tmp_path / "solvated.gro"
    out_top = tmp_path / "solvated.top"

    solvated = solvate_openmm(
        complex_obj,
        params=SolvationParams(ion_concentration=0.15, neutralize=True),
        output_gro=out_gro,
        output_top=out_top,
    )

    # pic = rass / "solvated.pickle"
    # with pic.open(mode="wb") as f:
    #    pickle.dump(solvated, f)

    assert solvated is not None
    assert out_gro.exists()
    assert out_top.exists()

    structure = pmd.load_file(str(out_top), xyz=str(out_gro))

    water_residues = [res for res in structure.residues if res.name in WATER_RESNAMES]

    assert len(water_residues) > 0, "No water molecules found in solvated system"


def test_bss_solvation() -> None:
    testdata = Path(__file__).resolve().parent / "testdata" / "solvation"

@pytest.mark.integration
    system = BSS.IO.readMolecules([str(testdata / "complex.gro"), str(testdata / "complex.top")])
    n_atoms_before = system.nAtoms()

    system = BSS.IO.readMolecules(
        [str(testdata / "complex.gro"), str(testdata / "complex.top")]
    )
        system=system,
        params=SolvationParams(
            water_model=WaterModel.TIP3P,
            padding=2,
            ion_concentration=0.1,
            neutralize=False,
        ),
    )

    export_gromacs_top_gro(solvated, "tests/testdata/minimization/solvated")

    # Check that solvent was added
    assert solvated.nAtoms() > n_atoms_before
    assert solvated.charge() == 0
