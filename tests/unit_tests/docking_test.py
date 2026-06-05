"""Unit tests for the lightweight docking adapter layer.

This module keeps only small, local checks that do not require external tools.
The goal is to verify stable contracts for helper behavior such as ligand
preparation output shape and Vina command construction.
Anything that requires real Vina execution, mk_export.py, or filesystem-heavy
workflow chaining belongs in the integration test module instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gbsa_pipeline.docking import DockingBox, VinaEngine, prepare_ligand_with_meeko
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
