"""Unit tests for gromacs_index: moleculetype-based atom selection and index-file writing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gbsa_pipeline.gromacs_index import (
    _moltype_atom_ranges,
    _parse_molecules_section,
    atoms_by_moltype,
    mmbsa_complex_atoms,
    mmbsa_index_groups,
    select_receptor_and_ligand_atoms,
    write_index,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_index(path: Path) -> str:
    return path.read_text()


_ATOMTYPES_BLOCK = """\
[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
1 2 yes 0.5 0.833333

[ atomtypes ]
; name at.num mass charge ptype sigma epsilon
CT 6 12.0107 0.0 A 0.339967 0.457730
"""


def _moleculetype_block(name: str, atoms: list[tuple[str, str]]) -> str:
    """Build a minimal ``[ moleculetype ]`` + ``[ atoms ]`` block ParmEd can parse.

    ``atoms`` is a list of ``(resname, atomname)`` pairs, one per atom.
    """
    lines = [f"[ moleculetype ]\n{name} 3\n\n[ atoms ]"]
    for i, (resname, atomname) in enumerate(atoms, start=1):
        lines.append(f"{i} CT {i} {resname} {atomname} {i} 0.0 12.0107")
    return "\n".join(lines) + "\n"


def _write_top(
    tmp_path: Path, moleculetypes: dict[str, list[tuple[str, str]]], molecules: list[tuple[str, int]]
) -> Path:
    """Write a minimal, ParmEd-parseable GROMACS .top file for a given moleculetype/molecules layout."""
    parts = [_ATOMTYPES_BLOCK]
    for name, atoms in moleculetypes.items():
        parts.append(_moleculetype_block(name, atoms))
    parts.append("[ system ]\nTest system\n")
    molecules_lines = "\n".join(f"{name} {count}" for name, count in molecules)
    parts.append(f"[ molecules ]\n{molecules_lines}\n")

    top_file = tmp_path / "test.top"
    top_file.write_text("\n".join(parts))
    return top_file


# ---------------------------------------------------------------------------
# _parse_molecules_section
# ---------------------------------------------------------------------------


def test_parse_molecules_section_parses_compound_counts(tmp_path: Path) -> None:
    top_file = tmp_path / "raw.top"
    top_file.write_text(
        "[ atomtypes ]\n; a bogus section before\nX 1 1.0 0.0 A 0.1 0.1\n\n"
        "[ molecules ]\n; Compound  #mols\nPROT   1\nSOL 11142\nNA 24\nCL 24\n"
    )

    compounds = _parse_molecules_section(top_file)

    assert compounds == [("PROT", 1), ("SOL", 11142), ("NA", 24), ("CL", 24)]


def test_parse_molecules_section_ignores_other_sections(tmp_path: Path) -> None:
    top_file = tmp_path / "raw.top"
    top_file.write_text("[ atoms ]\n1 CT 1 SOL OW 1 0.0 16.0\n\n[ molecules ]\nSOL 3\n")

    compounds = _parse_molecules_section(top_file)

    assert compounds == [("SOL", 3)]


# ---------------------------------------------------------------------------
# _moltype_atom_ranges / atoms_by_moltype
# ---------------------------------------------------------------------------


def test_moltype_atom_ranges_sequential_1_based() -> None:
    compounds = [("PROT", 1), ("LIG", 1), ("SOL", 3)]
    template_atom_counts = {"PROT": 3, "LIG": 2, "SOL": 1}

    ranges = _moltype_atom_ranges(compounds, template_atom_counts)

    assert ranges == {
        "PROT": [(1, 4)],
        "LIG": [(4, 6)],
        "SOL": [(6, 7), (7, 8), (8, 9)],
    }


def test_moltype_atom_ranges_raises_on_undefined_moleculetype() -> None:
    with pytest.raises(ValueError, match="LIG"):
        _moltype_atom_ranges([("LIG", 1)], template_atom_counts={"PROT": 3})


def test_atoms_by_moltype_selects_wanted_names() -> None:
    ranges = {"PROT": [(1, 4)], "LIG": [(4, 6)], "SOL": [(6, 9)]}

    assert atoms_by_moltype(ranges, ["LIG"]) == [4, 5]
    assert atoms_by_moltype(ranges, ["PROT", "SOL"]) == [1, 2, 3, 6, 7, 8]


def test_atoms_by_moltype_ignores_names_not_present() -> None:
    ranges = {"PROT": [(1, 4)]}

    assert atoms_by_moltype(ranges, ["NA", "CL"]) == []


# ---------------------------------------------------------------------------
# mmbsa_complex_atoms
# ---------------------------------------------------------------------------


def test_mmbsa_complex_atoms_excludes_recognized_solvent() -> None:
    ranges = {"PROT": [(1, 4)], "LIG": [(4, 6)], "SOL": [(6, 9)], "NA": [(9, 10)]}

    assert mmbsa_complex_atoms(ranges, total_atoms=9) == [1, 2, 3, 4, 5]


def test_mmbsa_complex_atoms_keeps_lipids() -> None:
    """Lipids must stay in the complex: cleantop() never strips them."""
    ranges = {"PROT": [(1, 4)], "POP": [(4, 6)], "LIG": [(6, 8)], "SOL": [(8, 11)]}

    assert mmbsa_complex_atoms(ranges, total_atoms=10) == [1, 2, 3, 4, 5, 6, 7]


def test_mmbsa_complex_atoms_raises_on_unrecognized_solvent_name() -> None:
    """Water named "HOH" looks like solvent but isn't a name gmx_MMPBSA's cleantop() strips."""
    ranges = {"PROT": [(1, 4)], "LIG": [(4, 6)], "HOH": [(6, 9)]}

    with pytest.raises(ValueError, match="HOH"):
        mmbsa_complex_atoms(ranges, total_atoms=8)


def test_mmbsa_complex_atoms_does_not_flag_structural_residue() -> None:
    """A non-solvent-looking excluded name (bug elsewhere) is not this function's job to catch."""
    ranges = {"PROT": [(1, 4)], "LIG": [(4, 6)], "COFACTOR": [(6, 7)]}

    # COFACTOR is not in _LOOKS_LIKE_SOLVENT_MOLTYPES, so no stray-solvent error --
    # but it also isn't stripped by cleantop(), so it stays in the complex.
    assert mmbsa_complex_atoms(ranges, total_atoms=6) == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# mmbsa_index_groups
# ---------------------------------------------------------------------------


def test_mmbsa_index_groups_partitions_receptor_and_ligand() -> None:
    ranges = {"PROT": [(1, 4)], "LIG": [(4, 6)], "SOL": [(6, 9)]}

    receptor, ligand = mmbsa_index_groups(ranges, total_atoms=8, ligand_moltype="LIG")

    assert receptor == [1, 2, 3]
    assert ligand == [4, 5]


def test_mmbsa_index_groups_lipids_land_in_receptor() -> None:
    ranges = {"PROT": [(1, 4)], "POP": [(4, 6)], "LIG": [(6, 8)], "SOL": [(8, 11)]}

    receptor, ligand = mmbsa_index_groups(ranges, total_atoms=10, ligand_moltype="LIG")

    assert receptor == [1, 2, 3, 4, 5]
    assert ligand == [6, 7]


def test_mmbsa_index_groups_raises_when_ligand_moltype_absent() -> None:
    ranges = {"PROT": [(1, 4)], "SOL": [(4, 7)]}

    with pytest.raises(RuntimeError, match="LIG"):
        mmbsa_index_groups(ranges, total_atoms=6, ligand_moltype="LIG")


def test_mmbsa_index_groups_raises_when_ligand_moltype_is_cleantop_stripped_name() -> None:
    """A ligand accidentally named "SOL" collides with a name cleantop() strips.

    Must fail loudly, not silently write a Ligand group referencing atoms
    gmx_MMPBSA's own topology cleaning has already removed.
    """
    ranges = {"PROT": [(1, 4)], "SOL": [(4, 7)]}

    with pytest.raises(ValueError, match="SOL"):
        mmbsa_index_groups(ranges, total_atoms=6, ligand_moltype="SOL")


def test_mmbsa_index_groups_raises_when_receptor_empty() -> None:
    """Everything besides the ligand is recognized solvent -- no receptor atoms remain."""
    ranges = {"LIG": [(1, 3)], "SOL": [(3, 6)]}

    with pytest.raises(RuntimeError, match="Protein"):
        mmbsa_index_groups(ranges, total_atoms=5, ligand_moltype="LIG")


# ---------------------------------------------------------------------------
# select_receptor_and_ligand_atoms -- real ParmEd parse of a minimal .top
# ---------------------------------------------------------------------------


def test_select_receptor_and_ligand_atoms_soluble_convention(tmp_path: Path) -> None:
    """[system] (soluble) convention: protein + ligand + water + ions."""
    top_file = _write_top(
        tmp_path,
        moleculetypes={
            "PROT": [("ALA", "CA"), ("ALA", "CB"), ("GLY", "CA")],
            "LIG": [("LIG", "C1"), ("LIG", "C2")],
            "SOL": [("SOL", "OW")],
            "NA": [("NA", "NA")],
        },
        molecules=[("PROT", 1), ("LIG", 1), ("SOL", 3), ("NA", 2)],
    )

    receptor, ligand = select_receptor_and_ligand_atoms(top_file, "LIG")

    assert receptor == [1, 2, 3]
    assert ligand == [4, 5]


def test_select_receptor_and_ligand_atoms_membrane_convention_lipids_in_receptor(tmp_path: Path) -> None:
    """[membrane] convention: lipids must join Receptor, not be dropped or excluded."""
    top_file = _write_top(
        tmp_path,
        moleculetypes={
            "PROT": [("ALA", "CA"), ("ALA", "CB"), ("GLY", "CA")],
            "POP": [("POP", "P8"), ("POP", "C1")],
            "LIG": [("LIG", "C1"), ("LIG", "C2")],
            "SOL": [("SOL", "OW")],
        },
        molecules=[("PROT", 1), ("POP", 2), ("LIG", 1), ("SOL", 4)],
    )

    receptor, ligand = select_receptor_and_ligand_atoms(top_file, "LIG")

    assert receptor == [1, 2, 3, 4, 5, 6, 7]
    assert ligand == [8, 9]


def test_select_receptor_and_ligand_atoms_raises_on_hoh_water(tmp_path: Path) -> None:
    """A prebuilt system naming crystallographic water "HOH" is rejected with a clear message."""
    top_file = _write_top(
        tmp_path,
        moleculetypes={
            "PROT": [("ALA", "CA"), ("ALA", "CB"), ("GLY", "CA")],
            "LIG": [("LIG", "C1"), ("LIG", "C2")],
            "HOH": [("HOH", "O")],
        },
        molecules=[("PROT", 1), ("LIG", 1), ("HOH", 500)],
    )

    with pytest.raises(ValueError, match="HOH"):
        select_receptor_and_ligand_atoms(top_file, "LIG")


# ---------------------------------------------------------------------------
# write_index
# ---------------------------------------------------------------------------


def test_write_index_writes_receptor_and_ligand_groups(tmp_path: Path) -> None:
    out = tmp_path / "test.ndx"
    write_index([1, 2, 3], [4, 5], out)

    content = _read_index(out)
    assert "[ Receptor ]" in content
    assert "[ Ligand ]" in content
    assert "1 2 3" in content
    assert "4 5" in content


def test_write_index_line_wrapping(tmp_path: Path) -> None:
    """16 receptor atoms - first line has 15 atoms, second has 1."""
    out = tmp_path / "test.ndx"
    write_index(list(range(1, 17)), [17], out)

    content = _read_index(out)
    lines = [ln for ln in content.splitlines() if ln and not ln.startswith("[")]
    first_line_nums = lines[0].split()
    assert len(first_line_nums) == 15
    second_line_nums = lines[1].split()
    assert len(second_line_nums) == 1
    assert second_line_nums[0] == "16"


def test_write_index_raises_when_receptor_empty(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Protein"):
        write_index([], [1, 2], tmp_path / "test.ndx")


def test_write_index_raises_when_ligand_empty(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Ligand"):
        write_index([1, 2, 3], [], tmp_path / "test.ndx")
