"""Module to generate GROMACS index files for a Sire molecular system.

This module provides a single public function, ``write_index_from_system``,
that iterates over all molecules in a Sire system to collect 1-based atom
indices for the receptor (protein) and ligand groups, then writes them in
the standard GROMACS ``.ndx`` format.  The output file is intended to be
passed to ``gmx_MMPBSA`` via the ``-ci`` flag or to standard GROMACS tools
such as ``gmx trjconv`` and ``gmx make_ndx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sire

if TYPE_CHECKING:
    from collections.abc import Sequence
    from io import TextIOWrapper
    from pathlib import Path

    import sire.mol
    import sire.system


def write_index_from_system(
    system: sire.system.System,
    protein: sire.mol.Molecule,
    ligand: sire.mol.Molecule,
    index_file: Path,
    *,
    cofactors: Sequence[sire.mol.Molecule] = (),
) -> None:
    """Write a GROMACS index file with Receptor and Ligand atom groups.

    The function iterates over every molecule in ``system`` in order,
    accumulating 1-based global atom indices for the molecule matched by
    ``protein`` and the molecule matched by ``ligand``; all other molecules
    are counted but silently skipped.  Molecules are matched by their
    ``number()`` identifier rather than by Python object identity, so the
    function works correctly with sire systems where iteration may yield new
    wrapper objects around the same underlying C++ molecule.  The groups are
    written as ``[ Receptor ]`` and ``[ Ligand ]`` sections, which are the
    names expected by gmx_MMPBSA when the ``-cg`` flag is used with group
    numbers 0 and 1.  Raises ``RuntimeError`` if either molecule is not found
    in the system, to fail early rather than silently produce an incomplete
    index file.

    Co-factors (e.g. FAD, heme, metal ions) are passed via ``cofactors``; their
    atoms are appended to the ``[ Receptor ]`` group so gmx_MMPBSA treats them
    as part of the receptor rather than the scored ligand.

    See the GROMACS index file format documentation at
    https://manual.gromacs.org/documentation/current/reference-manual/file-formats.html#ndx
    for the format specification.
    """
    protein_num = protein.number()
    ligand_num = ligand.number()
    cofactor_nums = {mol.number() for mol in cofactors}

    receptor_atoms: list[int] = []
    ligand_atoms: list[int] = []

    atom_counter = 1  # GROMACS uses 1-based indexing

    for mol in system:
        natoms = len(mol.atoms())
        start = atom_counter
        end = atom_counter + natoms
        num = mol.number()
        if num == protein_num or num in cofactor_nums:
            receptor_atoms.extend(range(start, end))
        elif num == ligand_num:
            ligand_atoms.extend(range(start, end))
        atom_counter = end

    if not receptor_atoms:
        raise RuntimeError("Protein atoms not found in system.")

    if not ligand_atoms:
        raise RuntimeError("Ligand atoms not found in system.")

    with index_file.open("w") as f:
        f.write("[ Receptor ]\n")
        _write_group(f, receptor_atoms)

        f.write("\n[ Ligand ]\n")
        _write_group(f, ligand_atoms)


def _write_group(f: TextIOWrapper, atoms: Sequence[int], per_line: int = 15) -> None:
    """Write a single index group body, 15 atom indices per line."""
    for i in range(0, len(atoms), per_line):
        f.write(" ".join(map(str, atoms[i : i + per_line])) + "\n")
