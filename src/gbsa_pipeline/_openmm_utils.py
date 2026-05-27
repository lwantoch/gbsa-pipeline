"""OpenMM Modeller utility helpers: loading, position extraction, residue deletion, PDB I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection

from openmm import unit as mm_unit
from openmm.app import Modeller, PDBFile

if TYPE_CHECKING:
    from pathlib import Path

    from openmm.app import ForceField


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_pdb_as_modeller(pdb_path: Path) -> Modeller:
    """Load a PDB file and return an OpenMM Modeller ready for further edits."""
    pdb = PDBFile(str(pdb_path))
    return Modeller(pdb.topology, pdb.positions)


def _load_waters_with_hydrogens(pdb_path: Path, forcefield: ForceField) -> Modeller:
    """Load a crystal-water PDB and complete missing hydrogens via OpenMM."""
    modeller = _load_pdb_as_modeller(pdb_path)
    modeller.addHydrogens(forcefield)
    return modeller


# ---------------------------------------------------------------------------
# Topology editing
# ---------------------------------------------------------------------------


def _delete_residues_by_name(modeller: Modeller, names: Collection[str]) -> int:
    """Delete all residues whose name matches any entry in ``names``.

    Comparison is case-insensitive and strips whitespace.  Returns the number
    of residues removed so callers can log the change without re-counting.
    """
    upper_names = {n.strip().upper() for n in names}
    to_delete = [r for r in modeller.topology.residues() if r.name.strip().upper() in upper_names]
    if to_delete:
        modeller.delete(to_delete)
    return len(to_delete)


# ---------------------------------------------------------------------------
# Position extraction
# ---------------------------------------------------------------------------


def _heavy_atom_coords(modeller: Modeller) -> list[tuple[float, float, float]]:
    """Return (x, y, z) nm positions for every non-hydrogen atom in a Modeller."""
    positions_nm = modeller.positions.value_in_unit(mm_unit.nanometer)
    return [
        (pos[0], pos[1], pos[2])
        for atom, pos in zip(modeller.topology.atoms(), positions_nm)
        if atom.element is not None and atom.element.symbol != "H"
    ]


def _oxygen_atom_entries(modeller: Modeller) -> list[tuple[Any, tuple[float, float, float]]]:
    """Return (residue, (x, y, z)) nm pairs for every oxygen atom in a Modeller."""
    positions_nm = modeller.positions.value_in_unit(mm_unit.nanometer)
    return [
        (atom.residue, (pos[0], pos[1], pos[2]))
        for atom, pos in zip(modeller.topology.atoms(), positions_nm)
        if atom.element is not None and atom.element.symbol == "O"
    ]


# ---------------------------------------------------------------------------
# PDB I/O
# ---------------------------------------------------------------------------


def _write_modeller_pdb(modeller: Modeller, output_pdb: Path) -> None:
    """Write the current Modeller state to a PDB file for inspection."""
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
