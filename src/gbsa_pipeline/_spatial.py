"""Format-agnostic spatial helpers: cell-grid construction and clash detection."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

_Coords = tuple[float, float, float]
_ResKey = tuple[int, str]  # canonical key for GRO residues; kept for _gro_io.py annotations
_CellGrid = dict[tuple[int, int, int], list[_Coords]]
_NEIGHBOUR_OFFSETS = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))

_K = TypeVar("_K", bound=Hashable)


def _build_cell_grid(coords: list[_Coords], cell_size: float) -> _CellGrid:
    """Index a list of coordinates into a spatial hash grid with given cell size."""
    grid: _CellGrid = {}
    for c in coords:
        key = (int(c[0] // cell_size), int(c[1] // cell_size), int(c[2] // cell_size))
        grid.setdefault(key, []).append(c)
    return grid


def _find_clashing_residues(
    candidate_entries: list[tuple[_K, _Coords]],
    reference_coords: list[_Coords],
    cutoff: float,
) -> set[_K]:
    """Return residue keys whose atoms come within ``cutoff`` of any reference coordinate.

    ``candidate_entries`` is a list of ``(residue_key, coords)`` pairs (e.g. water
    molecules).  ``reference_coords`` is the set of coordinates to check against
    (e.g. solute heavy atoms).  Both are in the same unit system; ``cutoff`` uses
    the same units.

    Uses a spatial hash grid so the check is O(N) rather than O(N*M).
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")

    grid = _build_cell_grid(reference_coords, cutoff)
    cutoff2 = cutoff * cutoff
    clashing: set[_K] = set()

    for res_key, (wx, wy, wz) in candidate_entries:
        if res_key in clashing:
            continue
        cx, cy, cz = int(wx // cutoff), int(wy // cutoff), int(wz // cutoff)
        for dx, dy, dz in _NEIGHBOUR_OFFSETS:
            for sx, sy, sz in grid.get((cx + dx, cy + dy, cz + dz), ()):
                if (wx - sx) ** 2 + (wy - sy) ** 2 + (wz - sz) ** 2 < cutoff2:
                    clashing.add(res_key)
                    break
            if res_key in clashing:
                break

    return clashing
