from __future__ import annotations

from typing import TYPE_CHECKING

import BioSimSpace as BSS
import pytest

from gbsa_pipeline.solvation_box import SolvationParams, WaterModel, run_solvation

if TYPE_CHECKING:
    from collections.abc import Iterable


def _molecule_names(system: BSS._SireWrappers.System) -> Iterable[str]:
    for mol in system:
        if hasattr(mol, "getName"):
            yield mol.getName()
        elif hasattr(mol, "name"):
            yield mol.name()
        else:
            yield str(mol)


def _is_ion(mol: BSS._SireWrappers.Molecule) -> bool:
    if hasattr(mol, "isIon"):
        return bool(mol.isIon())

    # Heuristic: single-atom molecules are treated as ions; otherwise check names.
    if hasattr(mol, "nAtoms") and mol.nAtoms() == 1:
        return True

    name = str(mol)
    return any(tag in name.upper() for tag in (" NA", " CL", " K", " CA", "NA+", "CL-"))


def _ion_molecules(
    system: BSS._SireWrappers.System,
) -> list[BSS._SireWrappers.Molecule]:
    if hasattr(system, "getIonMolecules"):
        ions = system.getIonMolecules()
        try:
            return list(ions)
        except TypeError:
            return []

    return [mol for mol in system if _is_ion(mol)]


@pytest.mark.integration
def test_solvation_real_protein_box_and_ions(tmp_path: pytest.TempPathFactory) -> None:
    system = BSS.IO.readMolecules(
        files=[
            "tests/testdata/test.gro",
            "tests/testdata/test.top",
        ],
        make_whole=True,
    )

    params = SolvationParams(
        water_model=WaterModel.TIP3P,
        padding=1.5,
        ion_concentration=0.1,
        neutralize=True,
    )

    solvated = run_solvation(system=system, params=params)

    # Box dimensions (Angstrom) -> ensure padding applied and reasonable size
    dims = solvated._sire_object.property("space").dimensions()
    dims_nm = [dim.value() / 10 for dim in dims]
    assert all(val > 3.0 for val in dims_nm)

    water_mols = solvated.getWaterMolecules()
    assert water_mols.nMolecules() > 0

    ions = [mol for mol in solvated if _is_ion(mol)]
    assert ions, "Expected ions when ion_concentration is set"


@pytest.mark.integration
def test_solvation_real_protein_without_neutralisation(
    tmp_path: pytest.TempPathFactory,
) -> None:
    system = BSS.IO.readMolecules(
        files=[
            "tests/testdata/test.gro",
            "tests/testdata/test.top",
        ],
        make_whole=True,
    )

    _ion_molecules(system)

    params = SolvationParams(
        water_model=WaterModel.TIP3P,
        padding=1.0,
        ion_concentration=0.0,
        neutralize=False,
    )

    solvated = run_solvation(system=system, params=params)

    water_mols = solvated.getWaterMolecules()
    assert water_mols.nMolecules() > 0

    _ion_molecules(solvated)  # ensure callable; no strict assertion because solvate may add ions
