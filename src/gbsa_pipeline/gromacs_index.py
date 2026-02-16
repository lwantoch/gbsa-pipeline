"""Module to generate gromacs index files for a system."""

from collections.abc import Sequence
from io import TextIOWrapper
from pathlib import Path

import BioSimSpace as BSS
import sire.system


def write_index_from_system(
    system: BSS._SireWrappers.System,
    protein: sire.mol.Molecule,
    ligand: sire.mol.Molecule,
    index_file: Path,
) -> None:
    """Generate GROMACS index file using BioSimSpace molecule indices."""
    protein_idx = system.getIndex(protein)
    ligand_idx = system.getIndex(ligand)

    receptor_atoms: list[int] = []
    ligand_atoms: list[int] = []

    atom_counter = 1  # GROMACS uses 1-based indexing

    for i, mol in enumerate(system):
        natoms = mol.nAtoms()
        start = atom_counter
        end = atom_counter + natoms
        if i == protein_idx:
            receptor_atoms.extend(range(start, end))
        elif i == ligand_idx:
            ligand_atoms.extend(range(start, end))
        atom_counter = end

    if not receptor_atoms:
        raise RuntimeError("Protein atoms not found in system.")

    if not ligand_atoms:
        raise RuntimeError("Ligand atoms not found in system.")

    with open(index_file, "w") as f:
        f.write("[ Receptor ]\n")
        _write_group(f, receptor_atoms)

        f.write("\n[ Ligand ]\n")
        _write_group(f, ligand_atoms)


def _write_group(f: TextIOWrapper, atoms: Sequence[int], per_line: int = 15) -> None:
    for i in range(0, len(atoms), per_line):
        f.write(" ".join(map(str, atoms[i : i + per_line])) + "\n")
