# /home/grheco/repositorios/gbsa-pipeline/tests/integration_tests/docking_integration_test.py

"""Integration tests for the lightweight docking adapter layer.

This module keeps only integration-level checks.
Unit-style checks such as command construction and simple Meeko output
generation should live in tests/unit_tests/, not here.
The goal here is to verify external-tool workflows end to end.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rdkit.Chem import rdMolAlign

from gbsa_pipeline.docking import (
    DockingBox,
    DockingRequest,
    VinaEngine,
    convert_receptor_pdb_to_pdbqt,
    export_pdbqt_to_sdf,
    load_first_sdf_molecule,
    molecule_centroid,
    prepare_ligand_with_meeko,
    remove_hydrogens_copy,
)

if TYPE_CHECKING:
    from rdkit import Chem


TESTDATA = Path(__file__).parents[1] / "testdata"
DOCKING_TESTDATA = TESTDATA / "docking"
DOCKLIGAND_SDF = DOCKING_TESTDATA / "dockligand.sdf"
DOCKPROTEIN_PDB = DOCKING_TESTDATA / "dockprotein.pdb"

DOCKPROTEIN_BOX = DockingBox(
    center=(10.115, 39.148, 53.112),
    size=(10.0, 10.0, 10.0),
)


def _assert_basic_ligand_pdbqt_content(path: Path) -> None:
    """Check that a prepared ligand file contains basic PDBQT sections.

    This helper exists because Meeko output details can vary slightly between
    versions, while downstream docking only needs a syntactically useful PDBQT
    file at this test level.
    The `path` parameter is required because the test should verify the actual
    file written by the public preparation helper, not an in-memory string.
    We currently check for a root section and at least one atom-like record,
    accepting either ATOM or HETATM because both are valid PDB-style records that
    can appear in ligand PDBQT output.
    """
    content = Path(path).read_text(encoding="utf-8")

    assert "ROOT" in content
    assert "ENDROOT" in content
    assert "ATOM" in content or "HETATM" in content


def _assert_basic_receptor_pdbqt_content(path: Path) -> None:
    """Check that a prepared receptor file contains basic PDBQT content.

    This helper is intentionally separate from ligand PDBQT checks because
    receptor PDBQT files do not necessarily contain ROOT and ENDROOT sections.
    The `path` parameter points to the receptor file produced by Meeko's receptor
    preparation command, so the test verifies the actual file consumed by Vina.
    We currently check only for a file with atom-like records because detailed
    receptor chemistry policy belongs to upstream receptor preparation, not this
    docking adapter test.
    """
    content = Path(path).read_text(encoding="utf-8")

    assert "ATOM" in content or "HETATM" in content


def _bond_order_sum(molecule: Chem.Mol) -> float:
    """Return the sum of bond orders over the heavy-atom graph.

    This is a deliberately simple chemistry summary used to compare whether a
    reconstructed ligand moved closer to the template chemistry than the raw
    export did.
    The `molecule` parameter is needed because we compare raw export, rebuilt
    export, and template against the same scalar metric.
    We are currently checking coarse bond-order restoration, not a full graph
    isomorphism proof of chemical correctness.
    """
    heavy = remove_hydrogens_copy(molecule)
    return sum(bond.GetBondTypeAsDouble() for bond in heavy.GetBonds())


def _aromatic_bond_count(molecule: Chem.Mol) -> int:
    """Count aromatic bonds on the heavy-atom graph.

    This helper complements `_bond_order_sum()` because aromatic perception is
    one of the places where raw export and reconstructed chemistry may differ.
    The `molecule` parameter is required because we compare template, raw, and
    rebuilt forms under the same aromaticity summary.
    We are currently checking whether reconstruction restores at least this
    basic aromatic feature set without disturbing the coordinates.
    """
    heavy = remove_hydrogens_copy(molecule)
    return sum(1 for bond in heavy.GetBonds() if bond.GetIsAromatic())


def test_meeko_smiles_to_pdbqt(tmp_path: Path) -> None:
    """Check that SMILES input can be prepared to a basic PDBQT file.

    This is still useful as an integration-level sanity check because Meeko and
    RDKit behavior can differ across environments even for trivial examples.
    The `tmp_path` parameter is required because the prepared PDBQT file is only
    a runtime artifact used to confirm that the preparation pipeline runs.
    We are currently checking that the output file exists and contains the
    expected PDBQT-style sections needed by later docking stages.
    """
    output = tmp_path / "ligand.pdbqt"

    path = prepare_ligand_with_meeko("CCO", output, name="ETH")

    assert path == output
    assert output.exists()
    _assert_basic_ligand_pdbqt_content(output)


def test_meeko_rdkit_mol_to_pdbqt(tmp_path: Path) -> None:
    """Check that an RDKit molecule input can be prepared to PDBQT.

    This test protects the current Meeko integration path used by the real
    docking workflow, where ligands commonly arrive from SDF files as RDKit
    molecules rather than as SMILES strings.
    The `tmp_path` parameter is required because the prepared PDBQT file is only
    a disposable runtime artifact used to verify the conversion.
    We are currently checking the public helper behavior, especially that it can
    satisfy Meeko's explicit-hydrogen and 3D-coordinate requirements before
    calling the Meeko preparation API.
    """
    if not DOCKLIGAND_SDF.exists():
        pytest.skip(f"missing ligand test file: {DOCKLIGAND_SDF}")

    output = tmp_path / "dockligand.pdbqt"
    molecule = load_first_sdf_molecule(DOCKLIGAND_SDF, remove_hs=False)

    path = prepare_ligand_with_meeko(molecule, output, name="DOCKLIG")

    assert path == output
    assert output.exists()
    _assert_basic_ligand_pdbqt_content(output)


@pytest.mark.integration
def test_meeko_receptor_pdb_to_pdbqt(tmp_path: Path) -> None:
    """Check that a receptor PDB can be prepared to receptor PDBQT with Meeko.

    This test protects the receptor-preparation path used by the Vina adapter
    when a request provides a PDB receptor instead of a prebuilt PDBQT receptor.
    The `tmp_path` parameter is required because Meeko writes runtime artifacts
    and the test should not modify repository fixtures.
    We currently check only that Meeko runs through the public helper and writes
    a basic receptor PDBQT file, because receptor protonation and cleanup are
    expected to happen upstream.
    """
    if not DOCKPROTEIN_PDB.exists():
        pytest.skip(f"missing receptor test file: {DOCKPROTEIN_PDB}")

    output = tmp_path / "dockprotein.pdbqt"

    path = convert_receptor_pdb_to_pdbqt(
        DOCKPROTEIN_PDB,
        output_path=output,
    )

    assert path == output
    assert output.exists()
    _assert_basic_receptor_pdbqt_content(output)


def test_vina_build_command(tmp_path: Path) -> None:
    """Check that the Vina adapter builds a command with the expected flags.

    This test keeps the command-construction contract explicit because small
    changes in argument ordering or omission can silently break integration
    runs even before Vina itself is called.
    The `tmp_path` parameter is required because we create disposable receptor,
    ligand, and output paths only to inspect the constructed command.
    We are currently checking that seed, CPU, mode count, exhaustiveness, and
    energy range are forwarded into the final Vina command line.
    """
    engine = VinaEngine(binary="vina")

    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    output = tmp_path / "out.pdbqt"

    receptor.write_text("", encoding="utf-8")
    ligand.write_text("", encoding="utf-8")

    cmd = engine._build_command(
        receptor=receptor,
        ligand=ligand,
        output=output,
        box=DOCKPROTEIN_BOX,
        seed=42,
        num_modes=5,
        exhaustiveness=3,
        energy_range=4.5,
        extra_flags={"--cpu": 2},
    )

    assert cmd[:2] == ["vina", "--receptor"]
    assert "--seed" in cmd
    assert "42" in cmd
    assert "--cpu" in cmd
    assert "2" in cmd
    assert "--num_modes" in cmd
    assert "5" in cmd
    assert "--exhaustiveness" in cmd
    assert "3" in cmd
    assert "--energy_range" in cmd
    assert "4.5" in cmd


@pytest.mark.integration
def test_pdbqt_to_sdf_roundtrip_preserves_pose(tmp_path: Path) -> None:
    """Check that raw PDBQT-to-SDF export preserves the ligand pose reasonably.

    This test isolates export behavior without docking so that failures can be
    attributed to Meeko preparation or Meeko-library roundtrip handling rather
    than to Vina sampling.
    The outputs are written into `tmp_path` so the integration test remains
    self-contained and does not modify repository fixtures.
    We are currently checking that a ligand prepared to PDBQT and exported back
    to SDF stays close in heavy-atom geometry to the original template ligand.

    Reference:
    RDKit documents `CalcRMS()` as an in-place RMS measure that is useful for
    docking-pose comparisons and does not pre-align the probe molecule:
    https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html
    """
    if not DOCKLIGAND_SDF.exists():
        pytest.skip(f"missing ligand test file: {DOCKLIGAND_SDF}")

    roundtrip_pdbqt = tmp_path / "dockligand_roundtrip.pdbqt"
    roundtrip_raw_sdf = tmp_path / "dockligand_roundtrip_raw.sdf"

    input_molecule = load_first_sdf_molecule(DOCKLIGAND_SDF, remove_hs=False)

    prepare_ligand_with_meeko(input_molecule, roundtrip_pdbqt, name="DOCKLIG")
    export_pdbqt_to_sdf(roundtrip_pdbqt, roundtrip_raw_sdf)

    assert roundtrip_pdbqt.exists()
    assert roundtrip_raw_sdf.exists()
    _assert_basic_ligand_pdbqt_content(roundtrip_pdbqt)

    exported_molecule = load_first_sdf_molecule(roundtrip_raw_sdf)

    input_heavy = remove_hydrogens_copy(input_molecule)
    exported_heavy = remove_hydrogens_copy(exported_molecule)

    assert input_heavy.GetNumAtoms() > 0
    assert exported_heavy.GetNumAtoms() > 0
    assert input_heavy.GetNumAtoms() == exported_heavy.GetNumAtoms()

    heavy_atom_rmsd = rdMolAlign.CalcRMS(exported_heavy, input_heavy)

    assert heavy_atom_rmsd < 1.0


@pytest.mark.integration
def test_pdbqt_to_sdf_template_reconstruction_restores_chemistry_without_pose_drift(
    tmp_path: Path,
) -> None:
    """Check isolated chemistry reconstruction on a non-docked roundtrip ligand.

    This test is intentionally narrower than the end-to-end docking test and
    exists so chemistry-rebuild failures can be debugged without involving Vina.
    The outputs are written into `tmp_path` so the integration test remains
    self-contained and does not modify repository fixtures.
    We are currently checking two things at once: the rebuilt molecule should
    remain where the raw exported molecule is, and its chemistry should match
    the original template at least as well as the raw export does.

    Reference:
    RDKit `AssignBondOrdersFromTemplate()` is the core mechanism used for
    restoring bond orders from a trusted template molecule:
    https://www.rdkit.org/docs/source/rdkit.Chem.AllChem.html
    """
    if not DOCKLIGAND_SDF.exists():
        pytest.skip(f"missing ligand test file: {DOCKLIGAND_SDF}")

    roundtrip_pdbqt = tmp_path / "dockligand_roundtrip.pdbqt"
    roundtrip_raw_sdf = tmp_path / "dockligand_roundtrip_raw.sdf"
    roundtrip_rebuilt_sdf = tmp_path / "dockligand_roundtrip_rebuilt.sdf"

    template_molecule = load_first_sdf_molecule(DOCKLIGAND_SDF, remove_hs=False)

    prepare_ligand_with_meeko(template_molecule, roundtrip_pdbqt, name="DOCKLIG")
    _assert_basic_ligand_pdbqt_content(roundtrip_pdbqt)

    raw_export_path = export_pdbqt_to_sdf(
        roundtrip_pdbqt,
        roundtrip_raw_sdf,
    )
    rebuilt_export_path = export_pdbqt_to_sdf(
        roundtrip_pdbqt,
        roundtrip_rebuilt_sdf,
        template_mol=template_molecule,
        add_hydrogens_after_template=True,
    )

    assert raw_export_path.exists()
    assert rebuilt_export_path.exists()

    raw_molecule = load_first_sdf_molecule(raw_export_path)
    rebuilt_molecule = load_first_sdf_molecule(rebuilt_export_path)

    raw_heavy = remove_hydrogens_copy(raw_molecule)
    rebuilt_heavy = remove_hydrogens_copy(rebuilt_molecule)
    template_heavy = remove_hydrogens_copy(template_molecule)

    assert raw_heavy.GetNumAtoms() > 0
    assert rebuilt_heavy.GetNumAtoms() > 0
    assert template_heavy.GetNumAtoms() > 0

    assert raw_heavy.GetNumAtoms() == rebuilt_heavy.GetNumAtoms()
    assert rebuilt_heavy.GetNumAtoms() == template_heavy.GetNumAtoms()

    raw_centroid = molecule_centroid(raw_heavy)
    rebuilt_centroid = molecule_centroid(rebuilt_heavy)
    centroid_shift = raw_centroid.Distance(rebuilt_centroid)

    raw_to_rebuilt_rmsd = rdMolAlign.CalcRMS(rebuilt_heavy, raw_heavy)

    assert centroid_shift < 0.5
    assert raw_to_rebuilt_rmsd < 1.0

    raw_bond_order_delta = abs(_bond_order_sum(raw_molecule) - _bond_order_sum(template_molecule))
    rebuilt_bond_order_delta = abs(_bond_order_sum(rebuilt_molecule) - _bond_order_sum(template_molecule))

    raw_aromatic_delta = abs(_aromatic_bond_count(raw_molecule) - _aromatic_bond_count(template_molecule))
    rebuilt_aromatic_delta = abs(_aromatic_bond_count(rebuilt_molecule) - _aromatic_bond_count(template_molecule))

    assert rebuilt_bond_order_delta <= raw_bond_order_delta
    assert rebuilt_aromatic_delta <= raw_aromatic_delta
    assert rebuilt_bond_order_delta == 0.0
    assert rebuilt_aromatic_delta == 0


@pytest.mark.integration
def test_vina_binary_smoke(tmp_path: Path) -> None:
    """Check that the external Vina workflow runs and returns one pose.

    This is intentionally just a smoke test and should not be interpreted as a
    scientific validation of the docking setup or box choice.
    The outputs are written into `tmp_path` so the integration test remains
    self-contained and does not modify repository fixtures.
    We are currently checking only that the adapter can prepare input, prepare
    the receptor with Meeko, call Vina, and produce one readable pose file with
    a parsed score.

    Reference:
    The Vina basic workflow requires a receptor, ligand, box center, and box
    size for a standard docking run:
    https://autodock-vina.readthedocs.io/en/latest/docking_basic.html
    """
    if shutil.which("vina") is None:
        pytest.skip("vina not available in PATH")

    if not DOCKPROTEIN_PDB.exists():
        pytest.skip(f"missing receptor test file: {DOCKPROTEIN_PDB}")

    if not DOCKLIGAND_SDF.exists():
        pytest.skip(f"missing ligand test file: {DOCKLIGAND_SDF}")

    docking_input_pdbqt = tmp_path / "dockligand_for_docking.pdbqt"
    docking_output_pdbqt = tmp_path / "dockligand_for_docking_vina_out.pdbqt"

    engine = VinaEngine(binary="vina")

    prepare_ligand_with_meeko(
        load_first_sdf_molecule(DOCKLIGAND_SDF, remove_hs=False),
        docking_input_pdbqt,
        name="DOCKLIG",
    )
    _assert_basic_ligand_pdbqt_content(docking_input_pdbqt)

    request = DockingRequest(
        receptor=DOCKPROTEIN_PDB,
        ligands=[docking_input_pdbqt],
        box=DOCKPROTEIN_BOX,
        workdir=tmp_path,
    )

    result = engine.dock(request=request)

    assert result.engine == "vina"
    assert len(result.poses) == 1
    assert result.poses[0].pose_path == docking_output_pdbqt
    assert result.poses[0].pose_path.exists()
    assert result.poses[0].metadata["returncode"] == 0
    assert result.poses[0].score is not None


@pytest.mark.integration
def test_docked_pose_reconstruction_restores_chemistry_and_keeps_docked_position(
    tmp_path: Path,
) -> None:
    """Run the full ligand-prep, docking, export, and reconstruction workflow.

    This is the real end-to-end test for the current docking segment and it is
    the one that matters most scientifically for the reconstruction logic.
    The generated docking artifacts are written into `tmp_path` so the workflow
    stays isolated while still exercising the full external-tool path.
    We are currently checking the core rule of the pipeline: the final rebuilt
    ligand must inherit chemistry from the original template while inheriting
    coordinates from the docked pose, not from the pre-docking template.

    Reference:
    RDKit `CalcRMS()` is used here specifically because its documentation notes
    that it computes RMS in place and is useful for docking-pose comparisons:
    https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html
    """
    if shutil.which("vina") is None:
        pytest.skip("vina not available in PATH")

    if not DOCKPROTEIN_PDB.exists():
        pytest.skip(f"missing receptor test file: {DOCKPROTEIN_PDB}")

    if not DOCKLIGAND_SDF.exists():
        pytest.skip(f"missing ligand test file: {DOCKLIGAND_SDF}")

    docking_input_pdbqt = tmp_path / "dockligand_for_docking.pdbqt"
    docking_output_raw_sdf = tmp_path / "dockligand_for_docking_vina_out_raw.sdf"
    docking_output_rebuilt_sdf = tmp_path / "dockligand_for_docking_vina_out_rebuilt.sdf"
    docking_output_pdbqt = tmp_path / "dockligand_for_docking_vina_out.pdbqt"

    template_molecule = load_first_sdf_molecule(DOCKLIGAND_SDF, remove_hs=False)

    prepare_ligand_with_meeko(
        template_molecule,
        docking_input_pdbqt,
        name="DOCKLIG",
    )
    _assert_basic_ligand_pdbqt_content(docking_input_pdbqt)

    engine = VinaEngine(binary="vina")
    request = DockingRequest(
        receptor=DOCKPROTEIN_PDB,
        ligands=[docking_input_pdbqt],
        box=DOCKPROTEIN_BOX,
        workdir=tmp_path,
    )

    result = engine.dock(request=request)

    assert result.engine == "vina"
    assert len(result.poses) == 1
    assert result.poses[0].pose_path == docking_output_pdbqt
    assert result.poses[0].pose_path.exists()
    assert result.poses[0].metadata["returncode"] == 0

    raw_docked_sdf = export_pdbqt_to_sdf(
        docking_output_pdbqt,
        docking_output_raw_sdf,
    )
    rebuilt_docked_sdf = export_pdbqt_to_sdf(
        docking_output_pdbqt,
        docking_output_rebuilt_sdf,
        template_mol=template_molecule,
        add_hydrogens_after_template=True,
    )

    assert raw_docked_sdf.exists()
    assert rebuilt_docked_sdf.exists()

    raw_docked_molecule = load_first_sdf_molecule(raw_docked_sdf)
    rebuilt_docked_molecule = load_first_sdf_molecule(rebuilt_docked_sdf)

    raw_docked_heavy = remove_hydrogens_copy(raw_docked_molecule)
    rebuilt_docked_heavy = remove_hydrogens_copy(rebuilt_docked_molecule)
    template_heavy = remove_hydrogens_copy(template_molecule)

    assert raw_docked_heavy.GetNumAtoms() > 0
    assert rebuilt_docked_heavy.GetNumAtoms() > 0
    assert template_heavy.GetNumAtoms() > 0

    assert raw_docked_heavy.GetNumAtoms() == rebuilt_docked_heavy.GetNumAtoms()
    assert rebuilt_docked_heavy.GetNumAtoms() == template_heavy.GetNumAtoms()

    raw_centroid = molecule_centroid(raw_docked_heavy)
    rebuilt_centroid = molecule_centroid(rebuilt_docked_heavy)
    centroid_shift = raw_centroid.Distance(rebuilt_centroid)

    raw_to_rebuilt_rmsd = rdMolAlign.CalcRMS(
        rebuilt_docked_heavy,
        raw_docked_heavy,
    )

    assert centroid_shift < 0.5
    assert raw_to_rebuilt_rmsd < 1.0

    rebuilt_bond_order_delta = abs(_bond_order_sum(rebuilt_docked_molecule) - _bond_order_sum(template_molecule))
    rebuilt_aromatic_delta = abs(
        _aromatic_bond_count(rebuilt_docked_molecule) - _aromatic_bond_count(template_molecule)
    )

    assert rebuilt_bond_order_delta == 0.0
    assert rebuilt_aromatic_delta == 0
