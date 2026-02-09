#!/usr/bin/env python3
# param_to_gromacs.py.

from __future__ import annotations

import logging
from pathlib import Path

import BioSimSpace as BSS
import argparse, time, logging

from gbsa_pipeline.equilibration import run_heating
from gbsa_pipeline.ligand_preparation import ligand_converter
from gbsa_pipeline.parametrization import load_and_parameterise
from gbsa_pipeline.solvation_box import run_solvation
from gbsa_pipeline.minimization import run_minimization


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Performs parametrization and minimization of protein (AMBER) + ligand (GAFF2) with BioSimSpace"
    )
    p.add_argument("protein_pdb", type=Path, help="Protein PDB path")
    p.add_argument(
        "ligand_sdf", type=Path, help="Ligand SDF path (3D coordinates recommended)"
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
    p.add_argument(
        "--water-mod",
        type=str,
        default="tip3p",
        help="Water model used for system solvation.",
    )
    p.add_argument(
        "--box-size",
        type=float,
        default="8",
        help="Size of solvation water box",
    )
    p.add_argument(
        "--min_steps",
        type=int,
        default="10000",
        help="Maximum number of steps for minimization procedure",
    )
    args = p.parse_args(argv)

    # Basic sanity checks
    if not args.protein_pdb.exists():
        raise FileNotFoundError(args.protein_pdb)
    if not args.ligand_sdf.exists():
        raise FileNotFoundError(args.ligand_sdf)

    # Preparing ligand - Reading, Standardizing, protonating and converting to sire
    bss_ligand = ligand_converter(sdf_path=args.ligand_sdf)

    # Calculating ligand parameter and adding protein forcefield
    parametrized_complex = load_and_parameterise(
        protein_pdb=args.protein_pdb,
        ligand_sire=bss_ligand,
        protein_ff=args.protein_ff,
        ligand_net_charge=args.ligand_net_charge,
        ligand_charge_method=args.ligand_charge_method,
        work_dir=args.work_dir,
    )

    # pickle.dump(parametrized_complex, "parametrized_complex.pickle")
    # pickle.load("parametrized_complex.pickle")

    logging.info("Parametrized the complex!")

    solvated_box = run_solvation(
        system=parametrized_complex.system,
        water_model=args.water_mod,
        box_size=args.box_size,
    )

    logging.info("Solvation Done!")

    BSS.IO.saveMolecules("box.pdb", solvated_box, fileformat="PDB")

    minimized = run_minimization(nsteps=args.min_steps, system=solvated_box)

    logging.info("We are done with minimization!")

    BSS.IO.saveMolecules("minimized.pdb", minimized, fileformat="PDB"

    t0 = time.time()
    equilibrated_system = run_heating(500 * BSS.Units.Time.picosecond, minimized)

    logging.info("After heating (%.1fs)", time.time() - t0)
    t1 = time.time()
    BSS.IO.saveMolecules("equilibrated.pdb", equilibrated_system, fileformat="PDB")

    logging.info("Heating done(%.1fs)", time.time() - t1)")


if __name__ == "__main__":
    raise SystemExit(main())
