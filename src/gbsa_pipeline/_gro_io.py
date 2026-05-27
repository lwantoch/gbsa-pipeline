"""Minimal GROMACS GRO file parser shared by md.py and md_diagnostics.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

_GRO_MIN_LINE_LEN = 44  # coordinates end at column 44


class _GROAtom(NamedTuple):
    atom_idx: int  # 1-based atom index
    res_num: int
    res_name: str
    atom_name: str
    x: float  # nm
    y: float  # nm
    z: float  # nm


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


def _parse_gro(gro_path: Path) -> list[_GROAtom]:
    """Read a GROMACS GRO file and return one :class:`_GROAtom` per atom.

    Short lines (< 44 characters) are skipped silently — they indicate
    truncated or velocity-only trailing records that do not carry coordinates.
    """
    with gro_path.open(encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    n_atoms = int(lines[1])
    return [_parse_gro_atom_line(line) for line in lines[2 : 2 + n_atoms] if len(line) >= _GRO_MIN_LINE_LEN]
