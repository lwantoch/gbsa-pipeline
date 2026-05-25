"""Compute a Vina docking box from a co-crystallised ligand in a PDB file.

Usage
-----
    python scripts/docking_box.py receptor.pdb CMU
    python scripts/docking_box.py receptor.pdb CMU --padding 10.0

The script reads all ATOM/HETATM records for the given residue name, computes
the centre of mass (unweighted geometric centre) and the bounding box extent,
then prints the centre, per-axis spans, and a recommended cubic box size with
the requested padding added to the longest span.  The output includes a
ready-to-paste ``DockingBox(...)`` line for ``modules_integration_test.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Vina docking box from a co-crystallised ligand.")
    p.add_argument("pdb", type=Path, help="Receptor PDB file containing the ligand.")
    p.add_argument("resname", help="Three-letter residue name of the ligand (e.g. CMU).")
    p.add_argument(
        "--padding",
        type=float,
        default=10.0,
        help="Padding in Å added to the longest span to get the box side (default: 10.0).",
    )
    return p.parse_args()


def _read_ligand_coords(pdb: Path, resname: str) -> list[tuple[float, float, float]]:
    """Return (x, y, z) for every ATOM/HETATM record matching resname."""
    coords: list[tuple[float, float, float]] = []
    target = resname.strip().upper()
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        rec_res = line[17:20].strip().upper()
        if rec_res != target:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append((x, y, z))
    return coords


def main() -> None:
    args = _parse_args()

    if not args.pdb.exists():
        sys.exit(f"Error: PDB file not found: {args.pdb}")

    coords = _read_ligand_coords(args.pdb, args.resname)

    if not coords:
        sys.exit(
            f"Error: no ATOM/HETATM records found for residue '{args.resname}' in {args.pdb}.\n"
            "Check the residue name (case-insensitive) and that the file contains the ligand."
        )

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)
    longest = max(span_x, span_y, span_z)
    box_side = round(longest + args.padding, 1)

    print(f"Ligand residue : {args.resname.upper()}")
    print(f"Atoms found    : {len(coords)}")
    print()
    print(f"Centre of mass : ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    print()
    print(f"Span X         : {span_x:.2f} Å")
    print(f"Span Y         : {span_y:.2f} Å")
    print(f"Span Z         : {span_z:.2f} Å")
    print(f"Longest span   : {longest:.2f} Å")
    print(f"Padding        : {args.padding:.1f} Å")
    print(f"Box side       : {box_side:.1f} Å  (longest + padding)")
    print()
    print("Ready to paste:")
    print(
        f"DOCKPROTEIN_BOX = DockingBox(\n"
        f"    center=({cx:.3f}, {cy:.3f}, {cz:.3f}),\n"
        f"    size=({box_side}, {box_side}, {box_side}),\n"
        f")"
    )


if __name__ == "__main__":
    main()
