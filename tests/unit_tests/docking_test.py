"""Unit tests for the lightweight docking adapter layer.

This module keeps only small, local checks that do not require external tools.
The goal is to verify stable contracts for helper behavior such as ligand
preparation output shape and Vina command construction.
Anything that requires real Vina execution, mk_export.py, or filesystem-heavy
workflow chaining belongs in the integration test module instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemmi
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gbsa_pipeline.docking import DockingBox, VinaEngine, prepare_ligand_with_meeko
from gbsa_pipeline.docking._crystal_waters import (
    _in_docking_box,
    _iter_water_oxygens,
    _read_pdb_like,
    _write_chain_as_pdb,
)
from gbsa_pipeline.docking._receptor_prep import _merge_sdfs_into_pdb, _strip_hetatm, merge_pdb_structures

if TYPE_CHECKING:
    from pathlib import Path


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

_MULTI_MODEL_PDBQT = """\
MODEL 1
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ENDMDL
MODEL 1
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
ENDMDL
"""

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


def test_read_pdb_like_multi_model_pdbqt_first_model_accessible(tmp_path: Path) -> None:
    """A Vina PDBQT with duplicate MODEL 1 blocks is read without error.

    Gemmi parses all MODEL blocks; callers use structure[0] to access the first pose.
    """
    pdbqt = tmp_path / "poses.pdbqt"
    pdbqt.write_text(_MULTI_MODEL_PDBQT, encoding="utf-8")
    st = _read_pdb_like(pdbqt)
    assert len(st) >= 1
    atoms = [atom for chain in st[0] for res in chain for atom in res]
    assert len(atoms) == 1  # only the N from the first MODEL 1 block


# ---------------------------------------------------------------------------
# _iter_water_oxygens tests
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


def test_iter_water_oxygens_yields_one_per_residue(tmp_path: Path) -> None:
    """Each residue contributes exactly one (res_id, xyz) pair."""
    pdb = tmp_path / "waters.pdb"
    pdb.write_text(_MULTI_WATER_PDB, encoding="utf-8")

    results = list(_iter_water_oxygens(pdb))

    assert len(results) == 2


def test_iter_water_oxygens_correct_coords(tmp_path: Path) -> None:
    """Yielded coordinates match the first heavy atom of each residue."""
    pdb = tmp_path / "waters.pdb"
    pdb.write_text(_MULTI_WATER_PDB, encoding="utf-8")

    results = list(_iter_water_oxygens(pdb))

    res_ids = [r[0] for r in results]
    assert res_ids == ["1", "2"]
    xyz0 = results[0][1]
    assert abs(xyz0[0] - 10.0) < 0.01
    assert abs(xyz0[1] - 10.0) < 0.01
    assert abs(xyz0[2] - 10.0) < 0.01


def test_iter_water_oxygens_skips_hydrogen_only_residues(tmp_path: Path) -> None:
    """Residues with only hydrogen atoms are skipped."""
    pdb = tmp_path / "honly.pdb"
    pdb.write_text(_HYDROGEN_ONLY_PDB, encoding="utf-8")

    results = list(_iter_water_oxygens(pdb))

    assert results == []


def test_iter_water_oxygens_empty_file(tmp_path: Path) -> None:
    """An empty PDB yields nothing."""
    pdb = tmp_path / "empty.pdb"
    pdb.write_text("END\n", encoding="utf-8")

    results = list(_iter_water_oxygens(pdb))

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
