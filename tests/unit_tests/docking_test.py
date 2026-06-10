"""Unit tests for the lightweight docking adapter layer.

This module keeps only small, local checks that do not require external tools.
The goal is to verify stable contracts for helper behavior such as ligand
preparation output shape and Vina command construction.
Anything that requires real Vina execution, mk_export.py, or filesystem-heavy
workflow chaining belongs in the integration test module instead.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gbsa_pipeline.docking import DockingBox, VinaEngine, prepare_ligand_with_meeko
from gbsa_pipeline.docking._crystal_waters import (
    _detect_clashes,
    _find_water_bridges,
    _in_docking_box,
    _iter_residue_coords,
    _read_pdb_like,
    _write_chain_as_pdb,
    select_docking_crystal_waters,
    validate_docked_pose,
)
from gbsa_pipeline.docking._receptor_prep import _merge_sdfs_into_pdb, _strip_hetatm, merge_pdb_structures

TESTDATA = Path(__file__).parents[1] / "testdata"


def test_meeko_smiles_to_pdbqt(tmp_path: Path) -> None:
    """Check that a simple SMILES string is converted into a PDBQT file.

    This is a unit-level contract test for `prepare_ligand_with_meeko()` and
    does not try to prove docking correctness or chemical realism beyond basic
    output generation.
    The `tmp_path` parameter is required because the function writes a PDBQT
    file, and unit tests should keep such artifacts isolated from repository
    fixtures and from other tests.
    We are currently checking three things: the returned path matches the
    requested output path, the file is actually created, and the contents look
    like a PDBQT-style ligand file by containing expected record sections.
    """
    output = tmp_path / "ligand.pdbqt"

    path = prepare_ligand_with_meeko("CCO", output, name="ETH")

    assert path == output
    assert output.exists()

    content = output.read_text(encoding="utf-8")
    assert "ROOT" in content
    assert "ATOM" in content


def test_vina_build_command(tmp_path: Path) -> None:
    """Check that the Vina engine builds the expected command-line arguments.

    This is a unit test for `_build_command()` and exists so argument forwarding
    can be validated without invoking the real Vina binary.
    The `tmp_path` parameter is required because the command is assembled from
    concrete receptor, ligand, and output paths, even though the files are only
    placeholders for this local contract check.
    We are currently checking that the box, random seed, mode count,
    exhaustiveness, energy range, and extra flags are all encoded into the
    generated command list in a way the downstream subprocess call can use.
    """
    engine = VinaEngine(binary="vina")
    box = DockingBox(center=(-3.245, 29.915, 53.639), size=(10.0, 10.0, 10.0))

    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    output = tmp_path / "out.pdbqt"

    receptor.write_text("", encoding="utf-8")
    ligand.write_text("", encoding="utf-8")

    cmd = engine._build_command(
        receptor=receptor,
        ligand=ligand,
        output=output,
        box=box,
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


# ---------------------------------------------------------------------------
# _strip_hetatm tests
# ---------------------------------------------------------------------------

_MIXED_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
HETATM    3  ZN  ZN  A 100       5.000   5.000   5.000  1.00  0.00          ZN
HETATM    4  O   HOH B   1      10.000  10.000  10.000  1.00  0.00           O
END
"""

_ATOM_ONLY_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
END
"""


def test_strip_hetatm_removes_hetatm_records(tmp_path: Path) -> None:
    """HETATM records (metals, waters) are removed; ATOM records are preserved."""
    src = tmp_path / "receptor.pdb"
    dest = tmp_path / "stripped.pdb"
    src.write_text(_MIXED_PDB, encoding="utf-8")

    result = _strip_hetatm(src, dest)

    assert result == dest
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")
    assert "HETATM" not in content
    assert content.count("ATOM") >= 2


def test_strip_hetatm_no_hetatm_is_unchanged(tmp_path: Path) -> None:
    """A PDB with no HETATM records is written unchanged (ATOM records preserved)."""
    src = tmp_path / "receptor.pdb"
    dest = tmp_path / "stripped.pdb"
    src.write_text(_ATOM_ONLY_PDB, encoding="utf-8")

    _strip_hetatm(src, dest)

    content = dest.read_text(encoding="utf-8")
    assert content.count("ATOM") == 2
    assert "HETATM" not in content


# ---------------------------------------------------------------------------
# merge_pdb_structures tests
# ---------------------------------------------------------------------------

_PROTEIN_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
END
"""

_WATERS_PDB = """\
HETATM    1  O   HOH W   1      10.000  10.000  10.000  1.00  0.00           O
HETATM    2  O   HOH W   2      11.000  11.000  11.000  1.00  0.00           O
END
"""


def test_merge_pdb_structures_merges_both(tmp_path: Path) -> None:
    """Merged PDB contains records from both input files."""
    protein = tmp_path / "protein.pdb"
    waters = tmp_path / "waters.pdb"
    out = tmp_path / "merged.pdb"
    protein.write_text(_PROTEIN_PDB, encoding="utf-8")
    waters.write_text(_WATERS_PDB, encoding="utf-8")

    result = merge_pdb_structures(protein, waters, out)

    assert result == out
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.count("ALA") >= 1
    assert content.count("HOH") >= 2


def test_merge_pdb_structures_creates_parent(tmp_path: Path) -> None:
    """Output parent directory is created if it does not exist."""
    protein = tmp_path / "protein.pdb"
    waters = tmp_path / "waters.pdb"
    out = tmp_path / "subdir" / "merged.pdb"
    protein.write_text(_PROTEIN_PDB, encoding="utf-8")
    waters.write_text(_WATERS_PDB, encoding="utf-8")

    merge_pdb_structures(protein, waters, out)

    assert out.exists()


# ---------------------------------------------------------------------------
# _merge_sdfs_into_pdb tests
# ---------------------------------------------------------------------------


def _write_ethanol_sdf(path: Path) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore[attr-defined]
    with Chem.SDWriter(str(path)) as w:
        w.write(mol)


def test_merge_sdfs_into_pdb_includes_cofactor(tmp_path: Path) -> None:
    """Output contains protein ATOM records and cofactor HETATM records."""
    protein = tmp_path / "protein.pdb"
    sdf = tmp_path / "cofactor.sdf"
    out = tmp_path / "merged.pdb"
    protein.write_text(_PROTEIN_PDB, encoding="utf-8")
    _write_ethanol_sdf(sdf)

    result = _merge_sdfs_into_pdb(protein, [sdf], out)

    assert result == out
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "ALA" in content
    assert "HETATM" in content


def test_merge_sdfs_into_pdb_multiple_sdfs(tmp_path: Path) -> None:
    """All chains from multiple SDFs are appended to the protein."""
    protein = tmp_path / "protein.pdb"
    sdf1 = tmp_path / "cof1.sdf"
    sdf2 = tmp_path / "cof2.sdf"
    out = tmp_path / "merged.pdb"
    protein.write_text(_PROTEIN_PDB, encoding="utf-8")
    _write_ethanol_sdf(sdf1)
    _write_ethanol_sdf(sdf2)

    _merge_sdfs_into_pdb(protein, [sdf1, sdf2], out)

    content = out.read_text(encoding="utf-8")
    assert content.count("HETATM") >= 2


def test_merge_sdfs_into_pdb_invalid_sdf_raises(tmp_path: Path) -> None:
    """An unreadable SDF raises ValueError."""
    protein = tmp_path / "protein.pdb"
    bad_sdf = tmp_path / "bad.sdf"
    out = tmp_path / "merged.pdb"
    protein.write_text(_PROTEIN_PDB, encoding="utf-8")
    bad_sdf.write_text("not a valid sdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read cofactor SDF"):
        _merge_sdfs_into_pdb(protein, [bad_sdf], out)


# ---------------------------------------------------------------------------
# _read_pdb_like tests
# ---------------------------------------------------------------------------

_SINGLE_MODEL_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
END
"""


def test_read_pdb_like_single_model(tmp_path: Path) -> None:
    """A plain PDB is read without error."""
    pdb = tmp_path / "single.pdb"
    pdb.write_text(_SINGLE_MODEL_PDB, encoding="utf-8")
    st = _read_pdb_like(pdb)
    assert len(st) >= 1


def test_read_pdb_like_multi_model_pdbqt_first_model_accessible() -> None:
    """A real Vina PDBQT with multiple MODEL/TORSDOF blocks is read without error.

    Vina PDBQT files contain TORSDOF inside MODEL blocks; gemmi's PDB parser
    resets its model counter on TORSDOF, causing it to raise on the second MODEL
    unless _read_pdb_like truncates to the first ENDMDL before parsing.
    """
    pdbqt = TESTDATA / "vina_multi_pose_out.pdbqt"
    st = _read_pdb_like(pdbqt)
    assert len(st) >= 1


# ---------------------------------------------------------------------------
# _iter_residue_coords tests
# ---------------------------------------------------------------------------

_MULTI_WATER_PDB = """\
HETATM    1  O   HOH W   1      10.000  10.000  10.000  1.00  0.00           O
HETATM    2  H1  HOH W   1      10.957  10.000  10.000  1.00  0.00           H
HETATM    3  O   HOH W   2      11.000  11.000  11.000  1.00  0.00           O
HETATM    4  H1  HOH W   2      11.957  11.000  11.000  1.00  0.00           H
END
"""

_HYDROGEN_ONLY_PDB = """\
ATOM      1  H   ALA A   1       1.000   2.000   3.000  1.00  0.00           H
END
"""


def test_iter_residue_coords_yields_one_per_residue(tmp_path: Path) -> None:
    """Each residue contributes exactly one (res_id, xyz) pair."""
    pdb = tmp_path / "waters.pdb"
    pdb.write_text(_MULTI_WATER_PDB, encoding="utf-8")

    results = list(_iter_residue_coords(pdb))

    assert len(results) == 2


def test_iter_residue_coords_correct_coords(tmp_path: Path) -> None:
    """Yielded coordinates match the first heavy atom of each residue."""
    pdb = tmp_path / "waters.pdb"
    pdb.write_text(_MULTI_WATER_PDB, encoding="utf-8")

    results = list(_iter_residue_coords(pdb))

    res_ids = [r[0] for r in results]
    assert res_ids == ["1", "2"]
    xyz0 = results[0][1]
    assert abs(xyz0[0] - 10.0) < 0.01
    assert abs(xyz0[1] - 10.0) < 0.01
    assert abs(xyz0[2] - 10.0) < 0.01


def test_iter_residue_coords_skips_hydrogen_only_residues(tmp_path: Path) -> None:
    """Residues with only hydrogen atoms are skipped."""
    pdb = tmp_path / "honly.pdb"
    pdb.write_text(_HYDROGEN_ONLY_PDB, encoding="utf-8")

    results = list(_iter_residue_coords(pdb))

    assert results == []


def test_iter_residue_coords_empty_file(tmp_path: Path) -> None:
    """An empty PDB yields nothing."""
    pdb = tmp_path / "empty.pdb"
    pdb.write_text("END\n", encoding="utf-8")

    results = list(_iter_residue_coords(pdb))

    assert results == []


# ---------------------------------------------------------------------------
# _in_docking_box tests
# ---------------------------------------------------------------------------

_BOX = DockingBox(center=(10.0, 10.0, 10.0), size=(4.0, 6.0, 8.0))


def test_in_docking_box_center_is_inside() -> None:
    """The box center is inside the box."""
    assert _in_docking_box(np.array([10.0, 10.0, 10.0]), _BOX)


def test_in_docking_box_corner_is_inside() -> None:
    """A point on the boundary is inside (inclusive)."""
    assert _in_docking_box(np.array([8.0, 7.0, 6.0]), _BOX)


def test_in_docking_box_outside_x() -> None:
    """A point just beyond the x boundary is outside."""
    assert not _in_docking_box(np.array([12.1, 10.0, 10.0]), _BOX)


def test_in_docking_box_outside_y() -> None:
    """A point just beyond the y boundary is outside."""
    assert not _in_docking_box(np.array([10.0, 13.1, 10.0]), _BOX)


def test_in_docking_box_outside_z() -> None:
    """A point just beyond the z boundary is outside."""
    assert not _in_docking_box(np.array([10.0, 10.0, 14.1]), _BOX)


# ---------------------------------------------------------------------------
# _write_chain_as_pdb tests
# ---------------------------------------------------------------------------


def _make_water_chain(chain_name: str = "W") -> gemmi.Chain:
    chain = gemmi.Chain(chain_name)
    res = gemmi.Residue()
    res.name = "HOH"
    res.seqid = gemmi.SeqId("1")
    atom = gemmi.Atom()
    atom.name = "O"
    atom.pos = gemmi.Position(1.0, 2.0, 3.0)
    atom.element = gemmi.Element("O")
    res.add_atom(atom)
    chain.add_residue(res)
    return chain


def test_write_chain_as_pdb_creates_file(tmp_path: Path) -> None:
    """Output file is created."""
    out = tmp_path / "chain.pdb"
    _write_chain_as_pdb(_make_water_chain(), out)
    assert out.exists()


def test_write_chain_as_pdb_creates_parent_dir(tmp_path: Path) -> None:
    """Parent directory is created if absent."""
    out = tmp_path / "sub" / "chain.pdb"
    _write_chain_as_pdb(_make_water_chain(), out)
    assert out.exists()


def test_write_chain_as_pdb_contains_residue(tmp_path: Path) -> None:
    """Written PDB contains the residue added to the chain."""
    out = tmp_path / "chain.pdb"
    _write_chain_as_pdb(_make_water_chain(), out)
    content = out.read_text(encoding="utf-8")
    assert "HOH" in content


# ---------------------------------------------------------------------------
# select_docking_crystal_waters tests
# ---------------------------------------------------------------------------

_ONE_WATER_AT_10 = """\
HETATM    1  O   HOH W   1      10.000  10.000  10.000  1.00  0.00           O
END
"""
_TWO_WATERS_ONE_FAR = """\
HETATM    1  O   HOH W   1      10.000  10.000  10.000  1.00  0.00           O
HETATM    2  O   HOH W   2      20.000  20.000  20.000  1.00  0.00           O
END
"""
_LIGAND_AT_10 = """\
HETATM    1  C1  LIG L   1      10.000  10.000  10.000  1.00  0.00           C
END
"""
_RECEPTOR_FAR = """\
ATOM      1  CA  ALA A   1      50.000  50.000  50.000  1.00  0.00           C
END
"""
_RECEPTOR_CLASH = """\
ATOM      1  CA  ALA A   1      10.000  10.000  11.000  1.00  0.00           C
END
"""
_BOX_AT_10 = DockingBox(center=(10.0, 10.0, 10.0), size=(6.0, 6.0, 6.0))


def test_select_crystal_waters_missing_file_returns_none(tmp_path: Path) -> None:
    """Missing crystal waters file returns (None, [])."""
    path, ids = select_docking_crystal_waters(
        tmp_path / "missing.pdb",
        _BOX_AT_10,
        tmp_path / "lig.pdb",
        tmp_path / "rec.pdb",
        tmp_path / "out.pdb",
    )
    assert path is None
    assert ids == []


def test_select_crystal_waters_retains_passing_water(tmp_path: Path) -> None:
    """Water inside the box and near the ligand is retained."""
    crystal = tmp_path / "waters.pdb"
    ligand = tmp_path / "ligand.pdb"
    receptor = tmp_path / "receptor.pdb"
    out = tmp_path / "out.pdb"
    crystal.write_text(_ONE_WATER_AT_10, encoding="utf-8")
    ligand.write_text(_LIGAND_AT_10, encoding="utf-8")
    receptor.write_text(_RECEPTOR_FAR, encoding="utf-8")

    path, ids = select_docking_crystal_waters(crystal, _BOX_AT_10, ligand, receptor, out)

    assert path == out
    assert ids == ["1"]


def test_select_crystal_waters_excludes_water_outside_box(tmp_path: Path) -> None:
    """Water outside the docking box is rejected."""
    crystal = tmp_path / "waters.pdb"
    ligand = tmp_path / "ligand.pdb"
    receptor = tmp_path / "receptor.pdb"
    out = tmp_path / "out.pdb"
    crystal.write_text(_TWO_WATERS_ONE_FAR, encoding="utf-8")
    ligand.write_text(_LIGAND_AT_10, encoding="utf-8")
    receptor.write_text(_RECEPTOR_FAR, encoding="utf-8")

    _, ids = select_docking_crystal_waters(crystal, _BOX_AT_10, ligand, receptor, out)

    assert "2" not in ids


def test_select_crystal_waters_excludes_receptor_clash(tmp_path: Path) -> None:
    """Water within receptor clash distance is rejected."""
    crystal = tmp_path / "waters.pdb"
    ligand = tmp_path / "ligand.pdb"
    receptor = tmp_path / "receptor.pdb"
    out = tmp_path / "out.pdb"
    crystal.write_text(_ONE_WATER_AT_10, encoding="utf-8")
    ligand.write_text(_LIGAND_AT_10, encoding="utf-8")
    receptor.write_text(_RECEPTOR_CLASH, encoding="utf-8")

    path, ids = select_docking_crystal_waters(
        crystal,
        _BOX_AT_10,
        ligand,
        receptor,
        out,
        receptor_clash_cutoff_angstrom=2.2,
    )

    assert path is None
    assert ids == []


# ---------------------------------------------------------------------------
# validate_docked_pose tests
# ---------------------------------------------------------------------------

_POSE_AT_50 = """\
HETATM    1  C1  LIG L   1      50.000  50.000  50.000  1.00  0.00           C
END
"""
_POSE_AT_10 = """\
HETATM    1  C1  LIG L   1      10.000  10.000  10.000  1.00  0.00           C
END
"""
_RECEPTOR_AT_10 = """\
ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00  0.00           C
END
"""


def test_validate_docked_pose_no_clashes(tmp_path: Path) -> None:
    """Ligand far from receptor produces no clashes."""
    pose = tmp_path / "pose.pdb"
    receptor = tmp_path / "receptor.pdb"
    pose.write_text(_POSE_AT_50, encoding="utf-8")
    receptor.write_text(_RECEPTOR_AT_10, encoding="utf-8")

    result = validate_docked_pose(pose, receptor)

    assert result.ligand_protein_clashes == []


def test_validate_docked_pose_detects_clash(tmp_path: Path) -> None:
    """Ligand atom coinciding with a receptor atom is flagged as a clash."""
    pose = tmp_path / "pose.pdb"
    receptor = tmp_path / "receptor.pdb"
    pose.write_text(_POSE_AT_10, encoding="utf-8")
    receptor.write_text(_RECEPTOR_AT_10, encoding="utf-8")

    result = validate_docked_pose(pose, receptor, clash_cutoff_angstrom=2.0)

    assert len(result.ligand_protein_clashes) >= 1
    label, dist = result.ligand_protein_clashes[0]
    assert "lig_atom" in label
    assert dist < 0.01


# ---------------------------------------------------------------------------
# _detect_clashes tests
# ---------------------------------------------------------------------------


def test_detect_clashes_no_clash() -> None:
    """Atoms far apart produce no clashes."""
    lig = np.array([[0.0, 0.0, 0.0]])
    rec = np.array([[10.0, 10.0, 10.0]])
    lp, lw, wp = _detect_clashes(lig, rec, [], cutoff=2.0)
    assert lp == []
    assert lw == []
    assert wp == []


def test_detect_clashes_lig_protein() -> None:
    """Ligand atom within cutoff of receptor atom is flagged."""
    lig = np.array([[0.0, 0.0, 0.0]])
    rec = np.array([[0.5, 0.0, 0.0]])
    lp, _, _ = _detect_clashes(lig, rec, [], cutoff=2.0)
    assert len(lp) == 1
    assert "lig_atom0" in lp[0][0]
    assert "rec_atom0" in lp[0][0]
    assert abs(lp[0][1] - 0.5) < 1e-6


def test_detect_clashes_water_protein() -> None:
    """Water oxygen within cutoff of receptor is flagged as water-protein clash."""
    lig = np.array([[50.0, 0.0, 0.0]])
    rec = np.array([[0.5, 0.0, 0.0]])
    water = [("1", np.array([0.0, 0.0, 0.0]))]
    _, _, wp = _detect_clashes(lig, rec, water, cutoff=2.0)
    assert len(wp) == 1
    assert "HOH1" in wp[0][0]


def test_detect_clashes_water_ligand() -> None:
    """Water oxygen within cutoff of ligand is flagged as water-ligand clash."""
    lig = np.array([[0.5, 0.0, 0.0]])
    rec = np.array([[50.0, 0.0, 0.0]])
    water = [("2", np.array([0.0, 0.0, 0.0]))]
    _, lw, _ = _detect_clashes(lig, rec, water, cutoff=2.0)
    assert len(lw) == 1
    assert "HOH2" in lw[0][0]


# ---------------------------------------------------------------------------
# _find_water_bridges tests
# ---------------------------------------------------------------------------


def test_find_water_bridges_detects_bridge() -> None:
    """Water equidistant between ligand and receptor in bridge range is a bridge."""
    lig = np.array([[3.0, 0.0, 0.0]])
    rec = np.array([[-3.0, 0.0, 0.0]])
    water = [("1", np.array([0.0, 0.0, 0.0]))]
    bridges = _find_water_bridges(water, lig, rec, min_angstrom=2.6, max_angstrom=3.2)
    assert len(bridges) == 1
    assert bridges[0][0] == "HOH1"


def test_find_water_bridges_too_close_to_ligand() -> None:
    """Water closer than bridge_min to ligand is not a bridge."""
    lig = np.array([[1.0, 0.0, 0.0]])
    rec = np.array([[-3.0, 0.0, 0.0]])
    water = [("1", np.array([0.0, 0.0, 0.0]))]
    bridges = _find_water_bridges(water, lig, rec, min_angstrom=2.6, max_angstrom=3.2)
    assert bridges == []


def test_find_water_bridges_too_far_from_receptor() -> None:
    """Water farther than bridge_max from receptor is not a bridge."""
    lig = np.array([[3.0, 0.0, 0.0]])
    rec = np.array([[-10.0, 0.0, 0.0]])
    water = [("1", np.array([0.0, 0.0, 0.0]))]
    bridges = _find_water_bridges(water, lig, rec, min_angstrom=2.6, max_angstrom=3.2)
    assert bridges == []


def test_find_water_bridges_empty_water_list() -> None:
    """No waters yields no bridges."""
    lig = np.array([[3.0, 0.0, 0.0]])
    rec = np.array([[-3.0, 0.0, 0.0]])
    assert _find_water_bridges([], lig, rec, min_angstrom=2.6, max_angstrom=3.2) == []
