"""GROMACS GRO/TOP I/O helpers: parsing, clash removal, topology patching."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)

_GRO_MIN_LINE_LEN = 44  # coordinates end at column 44


class _GROAtom(NamedTuple):
    atom_idx: int  # 1-based atom index
    res_num: int
    res_name: str
    atom_name: str
    x: float  # nm
    y: float  # nm
    z: float  # nm


# ---------------------------------------------------------------------------
# Line-level parsers
# ---------------------------------------------------------------------------


def _parse_gro_atom_line(line: str) -> _GROAtom:
    """Parse one GRO atom line into a :class:`_GROAtom`.

    Raises ``ValueError`` on malformed input so callers that process
    intermediate files fail explicitly rather than silently producing
    wrong geometry.
    """
    try:
        return _GROAtom(
            atom_idx=int(line[15:20]),
            res_num=int(line[0:5]),
            res_name=line[5:10].strip(),
            atom_name=line[10:15].strip(),
            x=float(line[20:28]),
            y=float(line[28:36]),
            z=float(line[36:44]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse GRO atom line: {line!r}") from exc


def _renumber_gro_atom_line(line: str, atom_number: int) -> str:
    """Return a GRO atom line with an updated five-column atom number."""
    if atom_number <= 0:
        raise ValueError("atom_number must be positive.")
    if atom_number > 99999:  # noqa: PLR2004
        atom_number = atom_number % 100000
        if atom_number == 0:
            atom_number = 99999
    return f"{line[:15]}{atom_number:5d}{line[20:]}"


# ---------------------------------------------------------------------------
# File-level parsers
# ---------------------------------------------------------------------------


def _parse_gro(gro_path: Path) -> list[_GROAtom]:
    """Read a GROMACS GRO file and return one :class:`_GROAtom` per atom.

    Short lines (< 44 characters) are skipped silently — they indicate
    truncated or velocity-only trailing records that do not carry coordinates.
    """
    with gro_path.open(encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    n_atoms = int(lines[1])
    return [_parse_gro_atom_line(line) for line in lines[2 : 2 + n_atoms] if len(line) >= _GRO_MIN_LINE_LEN]


# ---------------------------------------------------------------------------
# Clash detection and solvent cleanup
# ---------------------------------------------------------------------------


def _find_clashing_water_residues(
    atom_lines: list[str],
    cutoff_nm: float,
    water_resnames: set[str],
) -> set[tuple[int, str]]:
    """Identify whole water residues with impossible solute contacts.

    A water residue is flagged when any of its atoms is closer than
    ``cutoff_nm`` to any non-water atom. Uses a spatial grid to avoid an
    O(Nwater*Nsolute) full pair loop. Returns a set of ``(res_num, res_name)``
    keys so all atoms of each clashing water are removed together.
    """
    if cutoff_nm <= 0:
        raise ValueError("cutoff_nm must be positive.")

    cutoff2 = cutoff_nm * cutoff_nm
    cell_size = cutoff_nm
    grid: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    water_atoms: list[tuple[tuple[int, str], tuple[float, float, float]]] = []

    def cell_for(coords: tuple[float, float, float]) -> tuple[int, int, int]:
        return (int(coords[0] // cell_size), int(coords[1] // cell_size), int(coords[2] // cell_size))

    for line in atom_lines:
        atom = _parse_gro_atom_line(line)
        coords = (atom.x, atom.y, atom.z)
        if atom.res_name in water_resnames:
            water_atoms.append(((atom.res_num, atom.res_name), coords))
        else:
            grid.setdefault(cell_for(coords), []).append(coords)

    clashing: set[tuple[int, str]] = set()
    neighbour_offsets = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))

    for water_key, (wx, wy, wz) in water_atoms:
        if water_key in clashing:
            continue
        cx, cy, cz = cell_for((wx, wy, wz))
        for dx, dy, dz in neighbour_offsets:
            for sx, sy, sz in grid.get((cx + dx, cy + dy, cz + dz), ()):
                if (wx - sx) ** 2 + (wy - sy) ** 2 + (wz - sz) ** 2 < cutoff2:
                    clashing.add(water_key)
                    break
            if water_key in clashing:
                break

    return clashing


def _write_cleaned_gro(
    input_gro: Path,
    output_gro: Path,
    cutoff_nm: float,
    water_resnames: set[str],
) -> dict[str, int]:
    """Write a GRO file with clashing solvent waters removed.

    Removes whole water residues whose atoms are within ``cutoff_nm`` of any
    non-water atom, updates the atom count, and renumbers atom serials.
    Returns a ``{resname: count}`` dict of removed molecules for topology patching.
    """
    lines = input_gro.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:  # noqa: PLR2004
        raise ValueError(f"GRO file is too short: {input_gro}")

    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"Could not read GRO atom count from {input_gro}") from exc

    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"GRO file ended before all atom records were read: {input_gro}")

    clashing = _find_clashing_water_residues(atom_lines, cutoff_nm, water_resnames)

    cleaned = [
        line
        for line in atom_lines
        if not (
            (atom := _parse_gro_atom_line(line)).res_name in water_resnames
            and (atom.res_num, atom.res_name) in clashing
        )
    ]

    removed_counts: dict[str, int] = {}
    for _resnr, resname in clashing:
        removed_counts[resname] = removed_counts.get(resname, 0) + 1

    output_lines = [lines[0], f"{len(cleaned):5d}"]
    output_lines.extend(_renumber_gro_atom_line(line, i) for i, line in enumerate(cleaned, start=1))
    output_lines.append(lines[2 + atom_count])
    output_gro.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    return removed_counts


# ---------------------------------------------------------------------------
# Topology patching
# ---------------------------------------------------------------------------


def _update_topology_water_counts(
    input_top: Path,
    output_top: Path,
    removed_counts: dict[str, int],
) -> None:
    """Write a topology with [ molecules ] water counts reduced to match a cleaned GRO.

    Raises ``ValueError`` if waters were removed but no matching entry can be
    found in the ``[ molecules ]`` section.
    """
    if not removed_counts:
        output_top.write_text(input_top.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        return

    lines = input_top.read_text(encoding="utf-8", errors="replace").splitlines()
    in_molecules = False
    remaining = dict(removed_counts)
    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_molecules = stripped.strip("[]").strip().lower() == "molecules"
            output_lines.append(line)
            continue

        if not in_molecules or not stripped or stripped.startswith((";", "#")):
            output_lines.append(line)
            continue

        body, *comment_parts = line.split(";", 1)
        comment = ";" + comment_parts[0] if comment_parts else ""
        fields = body.split()
        if len(fields) < 2 or fields[0] not in remaining:  # noqa: PLR2004
            output_lines.append(line)
            continue

        molecule_name = fields[0]
        try:
            old_count = int(fields[1])
        except ValueError as exc:
            raise ValueError(f"Could not parse molecule count in topology line: {line!r}") from exc

        new_count = old_count - remaining[molecule_name]
        if new_count < 0:
            raise ValueError(
                f"Cannot remove {remaining[molecule_name]} {molecule_name} molecules: topology count is {old_count}."
            )

        prefix = line[: line.find(molecule_name)] if molecule_name in line else ""
        output_lines.append(f"{prefix}{molecule_name:<16} {new_count}{(' ' + comment) if comment else ''}".rstrip())
        del remaining[molecule_name]

    if remaining:
        raise ValueError(
            "Removed solvent waters but could not update matching topology molecule counts: "
            + ", ".join(f"{name}={count}" for name, count in sorted(remaining.items()))
        )

    output_top.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
