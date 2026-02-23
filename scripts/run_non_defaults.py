"""Performing first MD Run.

We start from SDF file, read ligand, standardize and hydrogen it;
We load protein;
We parametrize ligand and load the protein force field parameters;
We add solvent of a chosen water model and counter ions;
We minimize the system;
We perform equilibration with restraints by stepwise heating the system from 0 to 300 K.
"""

from __future__ import annotations

import logging
from pathlib import Path

import BioSimSpace as BSS

from gbsa_pipeline.change_defaults import GromacsCustom, GromacsParams

logger = logging.getLogger(__name__)


def main() -> None:
    """Run a GROMACS process using a protocol with in-code parameter overrides."""
    logging.basicConfig(level=logging.INFO)

    repo = Path(__file__).resolve().parents[1]  # scripts/ -> repo root

    # Input system
    test_system = BSS.IO.readMolecules(
        files=[
            str(repo / "tests" / "testdata" / "test.gro"),
            str(repo / "tests" / "testdata" / "test.top"),
        ],
        make_whole=True,
    )

    logger.info("Read System")

    # Build protocol from params only
    params = GromacsParams(
        integrator="md",
        dt=0.001,
        nsteps=100,
        nstlog=500,
        nstenergy=500,
        nstxout_compressed=500,
    )

    prot = GromacsCustom(params=params)

    # Run process
    logger.info("Starting GROMACS process")
    proc = BSS.Process.Gromacs(test_system, protocol=prot)
    proc.start()
    proc.wait()
    logger.info("GROMACS finished")

    customized = proc.getSystem(block=True)

    # Save result
    out_path = repo / "customized.pdb"
    BSS.IO.saveMolecules(str(out_path), customized, fileformat="PDB")
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
