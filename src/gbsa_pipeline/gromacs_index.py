"""Module to generate GROMACS index files from a GROMACS topology's moleculetypes.

This module separates *selecting* which atoms belong to the Receptor/Ligand
groups from *writing* them out in the standard GROMACS ``.ndx`` format. The
output file is intended to be passed to ``gmx_MMPBSA`` via the ``-ci`` flag or
to standard GROMACS tools such as ``gmx trjconv`` and ``gmx make_ndx``.

Selection is driven entirely by GROMACS moleculetype identity -- the compound
name declared in the topology's ``[ molecules ]`` section -- read via ParmEd
(:class:`parmed.gromacs.GromacsTopologyFile`, which parses a plain-text
``.top`` directly; no ``.tpr`` is needed). This is the same mechanism
gmx_MMPBSA's own topology cleaning (``GMXMMPBSA.make_top.cleantop``) uses to
decide what to strip, so a single, generic partition (complex = everything
cleantop() will keep; Receptor = complex minus ligand) works for both a
[system] (soluble) and a [membrane] run, with no molecule-position or
-number bookkeeping in this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import parmed

from gbsa_pipeline._constants import ION_RESIDUE_NAMES, WATER_RESIDUE_NAMES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from io import TextIOWrapper
    from pathlib import Path

# Residue names GMXMMPBSA.make_top.cleantop() actually strips from the "-cp"
# topology before matching it against our Receptor/Ligand index (hardcoded in
# the installed GMXMMPBSA.make_top source -- there is no public API or CLI
# flag for this list, so it was read directly out of the installed package).
# Notably missing "HOH": a real prebuilt system (e.g. CHARMM-GUI output, or
# any system carrying raw PDB-style crystallographic water) commonly names
# bulk water "HOH", which cleantop() will NOT strip.
_CLEANTOP_STRIPPED_RESNAMES: frozenset[str] = frozenset(
    {
        "NA",
        "CL",
        "SOL",
        "SOD",
        "Na+",
        "CLA",
        "Cl-",
        "POT",
        "K+",
        "TIP3P",
        "TIP3",
        "TP3",
        "TIPS3P",
        "TIP3o",
        "TIP4P",
        "TIP4PEW",
        "T4E",
        "TIP4PD",
        "TIP5P",
        "SPC",
        "SPC/E",
        "SPCE",
        "WAT",
        "OPC",
    }
)

# Broader, human-recognizable solvent/ion names -- a superset of what
# cleantop() actually strips (e.g. it also recognizes "HOH", which cleantop()
# does not). Built from the package's shared water/ion name constants (see
# gbsa_pipeline._constants) rather than a fresh list, so this stays in sync
# with the names other stages already recognize as solvent.
_LOOKS_LIKE_SOLVENT_MOLTYPES: frozenset[str] = WATER_RESIDUE_NAMES | ION_RESIDUE_NAMES


def _parse_molecules_section(top_file: Path) -> list[tuple[str, int]]:
    """Parse the ``[ molecules ]`` section of a GROMACS ``.top`` file.

    Returns ``(moleculetype_name, count)`` pairs in file order -- the same
    order GROMACS (and ParmEd, and gmx_MMPBSA's own cleantop()) expand into
    the final, flat atom sequence. ParmEd parses and validates this same
    section internally but does not retain the ordered per-occurrence count
    list on the returned :class:`~parmed.gromacs.GromacsTopologyFile` (only
    the *distinct* moleculetypes, via its ``.molecules`` mapping), so it is
    re-read here directly; this mirrors the parsing convention already used
    for the same section in :func:`gbsa_pipeline._gro_io._update_topology_water_counts`.
    """
    lines = top_file.read_text(encoding="utf-8", errors="replace").splitlines()
    in_molecules = False
    compounds: list[tuple[str, int]] = []

    for line in lines:
        stripped = line.split(";", 1)[0].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_molecules = stripped.strip("[]").strip().lower() == "molecules"
            continue
        if not in_molecules or not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 2:  # noqa: PLR2004
            continue
        compounds.append((fields[0], int(fields[1])))

    return compounds


def _moltype_atom_ranges(
    compounds: Sequence[tuple[str, int]],
    template_atom_counts: Mapping[str, int],
) -> dict[str, list[tuple[int, int]]]:
    """1-based, half-open atom-index ranges for each moleculetype occurrence, in file order.

    ``compounds`` is the ``[ molecules ]`` section (see
    :func:`_parse_molecules_section`); ``template_atom_counts`` maps each
    moleculetype name to the atom count of its single ``[ moleculetype ]``
    definition (see :class:`parmed.gromacs.GromacsTopologyFile`'s
    ``.molecules`` mapping). Replays the exact same sequential expansion
    ParmEd's own parser performs (``self += molecules[molname][0] * num``,
    in ``gromacstop.py``) to recover which atom-index range each occurrence
    covers -- information ParmEd itself does not retain after parsing.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    atom_counter = 1  # GROMACS uses 1-based indexing

    for name, count in compounds:
        if name not in template_atom_counts:
            raise ValueError(
                f"Moleculetype {name!r} appears in the '[ molecules ]' section but has no "
                "matching '[ moleculetype ]' definition in the topology."
            )
        natoms = template_atom_counts[name]
        for _ in range(count):
            start = atom_counter
            end = atom_counter + natoms
            ranges.setdefault(name, []).append((start, end))
            atom_counter = end

    return ranges


def atoms_by_moltype(ranges: Mapping[str, list[tuple[int, int]]], moltypes: Iterable[str]) -> list[int]:
    """Flat, sorted 1-based atom indices belonging to any moleculetype in ``moltypes``.

    A range lookup rather than a GROMACS selection string: moleculetype
    names like ``"Na+"`` or ``"SPC/E"`` don't survive typical selection
    grammars.
    """
    wanted = set(moltypes)
    indices: list[int] = []
    for name in wanted & ranges.keys():
        for start, end in ranges[name]:
            indices.extend(range(start, end))
    return sorted(indices)


def mmbsa_complex_atoms(ranges: Mapping[str, list[tuple[int, int]]], total_atoms: int) -> list[int]:
    """The complex exactly as gmx_MMPBSA will see it after its topology cleaning.

    Raises if a moleculetype that looks like solvent/ions
    (``_LOOKS_LIKE_SOLVENT_MOLTYPES``) survives cleantop()'s actual, narrower
    strip list (``_CLEANTOP_STRIPPED_RESNAMES``) -- e.g. "HOH" -- since it
    would otherwise be silently treated as part of the receptor.
    """
    stripped = set(atoms_by_moltype(ranges, _CLEANTOP_STRIPPED_RESNAMES))
    complex_atoms = [i for i in range(1, total_atoms + 1) if i not in stripped]

    stray = (set(ranges) - _CLEANTOP_STRIPPED_RESNAMES) & _LOOKS_LIKE_SOLVENT_MOLTYPES
    if stray:
        raise ValueError(
            f"Moleculetype(s) {sorted(stray)} look like solvent/ions but are not names "
            "gmx_MMPBSA's own topology cleaning (GMXMMPBSA.make_top.cleantop) recognizes -- "
            "they survive into the cleaned topology and the atom counts will no longer match "
            f"the Receptor/Ligand index. Rename them to a recognized name (one of "
            f"{sorted(_CLEANTOP_STRIPPED_RESNAMES)}) before the MMPBSA stage."
        )

    return complex_atoms


def mmbsa_index_groups(
    ranges: Mapping[str, list[tuple[int, int]]],
    total_atoms: int,
    ligand_moltype: str,
) -> tuple[list[int], list[int]]:
    """Partition the cleaned complex into the Receptor/Ligand atom-index groups.

    Works for both a [system] (soluble) run and a [membrane] run: the
    Receptor is "everything gmx_MMPBSA's own topology cleaning will keep,
    minus the ligand" -- protein for a soluble run, protein and lipids for a
    membrane run, since cleantop() never strips lipids either.
    """
    if ligand_moltype in _CLEANTOP_STRIPPED_RESNAMES:
        # mmbsa_complex_atoms() below strips every atom of this moleculetype
        # before the ligand is ever considered, since it exactly matches a
        # name cleantop() itself strips. atoms_by_moltype() would still find
        # them by name regardless, so without this check the returned
        # "Ligand" index group would reference atoms that no longer exist in
        # gmx_MMPBSA's own cleaned topology -- the same silent atom-count
        # mismatch this module exists to prevent, just moved to the ligand
        # side instead of the receptor side.
        raise ValueError(
            f"Ligand moleculetype {ligand_moltype!r} is a name gmx_MMPBSA's own topology "
            "cleaning (GMXMMPBSA.make_top.cleantop) strips as solvent/ions -- it would be "
            "removed from the cleaned topology before the Receptor/Ligand index is ever "
            "applied. Rename the ligand's moleculetype to something else."
        )

    complex_atoms = mmbsa_complex_atoms(ranges, total_atoms)
    ligand_atoms = set(atoms_by_moltype(ranges, [ligand_moltype]))
    if not ligand_atoms:
        raise RuntimeError(f"No atoms with moleculetype {ligand_moltype!r} found in the topology.")

    receptor_atoms = [i for i in complex_atoms if i not in ligand_atoms]
    if not receptor_atoms:
        raise RuntimeError("Protein/lipid atoms not found in system.")

    return receptor_atoms, sorted(ligand_atoms)


def select_receptor_and_ligand_atoms(top_file: Path, ligand_moltype: str) -> tuple[list[int], list[int]]:
    """Select Receptor/Ligand atom indices for gmx_MMPBSA, from a GROMACS topology alone.

    Replaces molecule-position/-number bookkeeping with moleculetype
    identity read directly from ``top_file`` -- the same file, and the same
    ``[ molecules ]`` compound names, gmx_MMPBSA's own topology cleaning
    (``GMXMMPBSA.make_top.cleantop``) uses. ``ligand_moltype`` is the
    ligand's moleculetype/residue name (identical for the single-residue
    small molecules this pipeline's ligands always are, e.g. "UNK") --
    confirmed against this project's own production topologies. No ``.gro``
    or coordinate file is needed since only topology bookkeeping (moleculetype
    identity and atom counts), not geometry, is used here.
    """
    top = parmed.gromacs.GromacsTopologyFile(str(top_file), parametrize=False)
    template_atom_counts = {name: len(struct.atoms) for name, (struct, _nrexcl) in top.molecules.items()}
    compounds = _parse_molecules_section(top_file)
    ranges = _moltype_atom_ranges(compounds, template_atom_counts)

    return mmbsa_index_groups(ranges, len(top.atoms), ligand_moltype)


def write_index(
    receptor_atoms: Sequence[int],
    ligand_atoms: Sequence[int],
    index_file: Path,
) -> None:
    """Write a GROMACS index file with Receptor and Ligand atom groups.

    The groups are written as ``[ Receptor ]`` and ``[ Ligand ]`` sections,
    which are the names expected by gmx_MMPBSA when the ``-cg`` flag is used
    with group numbers 0 and 1. Raises ``RuntimeError`` if either group is
    empty, to fail early rather than silently produce an incomplete index
    file. See the GROMACS index file format documentation at
    https://manual.gromacs.org/documentation/current/reference-manual/file-formats.html#ndx
    for the format specification.
    """
    if not receptor_atoms:
        raise RuntimeError("Protein/lipid atoms not found in system.")

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
