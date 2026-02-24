from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from gbsa_pipeline.docking import (
    DockingBox,
    DockingRequest,
)
from gbsa_pipeline.solvation_box import (
    BoxShape,
    SolvationParams,
    WaterModel,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_docking_request_input_round_trip(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    receptor.write_text("")
    ligand.write_text("")

    request = DockingRequest(
        receptor=receptor,
        ligands=[ligand],
        box=DockingBox(center=(0.0, 1.0, 2.0), size=(10.0, 10.0, 10.0)),
        seed=7,
        workdir=tmp_path,
        parameters={"cpu": 2},
    )

    assert request.receptor == receptor
    assert list(request.ligands) == [ligand]
    assert request.box.center == (0.0, 1.0, 2.0)
    assert request.parameters["cpu"] == 2
    assert request.workdir == tmp_path


def test_docking_request_input_validates_paths(tmp_path: Path) -> None:
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("")

    with pytest.raises(ValidationError):
        DockingRequest(
            receptor=tmp_path / "missing.pdbqt",
            ligands=[ligand],
            box=DockingBox(center=(0, 0, 0), size=(1, 1, 1)),
        )


def test_solvation_params_input_round_trip() -> None:
    params = SolvationParams(
        water_model="TIP4P",
        padding=1.0,
        box_size=None,
        ion_concentration=0.0,
        is_neutral=False,
        shape="cubic",
    )

    assert params.water_model == WaterModel.TIP4P
    assert params.padding == 1.0
    assert params.box_size is None
    assert params.is_neutral is False
    assert params.shape == BoxShape.CUBIC


def test_solvation_params_input_requires_box_or_padding() -> None:
    with pytest.raises(ValidationError):
        SolvationParams(box_size=None, padding=None)

    with pytest.raises(ValidationError):
        SolvationParams(box_size=-1.0)
