#!/usr/bin/env python3
"""param_to_gromacs.py.

Read a protein PDB + ligand SDF, parameterise (protein: AMBER ff14SB by default,
ligand: GAFF2 with AM1-BCC by default), build a complex System, and export
GROMACS-ready input files: <prefix>.gro and <prefix>.top

Assumptions:
- Ligand coordinates are already in the same reference frame as the protein (i.e. from a complex).
- BioSimSpace is installed and can find AmberTools/CGenFF tooling as needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gbsa_pipeline.parametrization import (
    export_gromacs_top_gro,
    load_and_parameterise,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Parameterise protein (AMBER) + ligand (GAFF2) with BioSimSpace and export GROMACS .top/.gro."
    )
    p.add_argument("protein_pdb", type=Path, help="Protein PDB path")
    p.add_argument("ligand_sdf", type=Path, help="Ligand SDF path (3D coordinates recommended)")
    p.add_argument(
        "-o",
        "--out-prefix",
        default="complex",
        help="Output prefix (writes <prefix>.gro and <prefix>.top)",
    )
    p.add_argument(
        "--protein-ff",
        default="ff14SB",
        choices=["ff14SB", "ff19SB", "ff99SB"],
        help="AMBER protein force field",
    )
    p.add_argument(
        "--ligand-charge-method",
        default="BCC",
        help="Ligand charge method for GAFF2 (e.g. BCC).",
    )
    p.add_argument(
        "--ligand-net-charge",
        type=int,
        default=None,
        help="Integer net charge for the ligand (recommended if known).",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for parameterisation intermediates (optional).",
    )

    args = p.parse_args(argv)

    # Basic sanity checks
    if not args.protein_pdb.exists():
        raise FileNotFoundError(args.protein_pdb)
    if not args.ligand_sdf.exists():
        raise FileNotFoundError(args.ligand_sdf)

    complex_ = load_and_parameterise(
        protein_pdb=args.protein_pdb,
        ligand_sdf=args.ligand_sdf,
        protein_ff=args.protein_ff,
        ligand_net_charge=args.ligand_net_charge,
        ligand_charge_method=args.ligand_charge_method,
        work_dir=args.work_dir,
    )

    written = export_gromacs_top_gro(complex_.system, args.out_prefix)

    print("Wrote:")
    for f in written:
        print(f"  - {f}")

    print("\nNext steps (example):")
    print(f"  gmx grompp -f md.mdp -c {args.out_prefix}.gro -p {args.out_prefix}.top -o md.tpr")
    print("  gmx mdrun -deffnm md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
