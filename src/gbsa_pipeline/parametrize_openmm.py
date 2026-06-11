"""OpenMM-based parametrization path."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import parmed as pmd
from openff.toolkit.topology import Molecule
from openmm.app import ForceField, Modeller, NoCutoff
from openmmforcefields.generators import GAFFTemplateGenerator

from gbsa_pipeline._constants import WATER_RESIDUE_NAMES
from gbsa_pipeline._openmm_utils import _delete_residues_by_name, _load_pdb_as_modeller
from gbsa_pipeline.parametrization_enum import LigandFF, ProteinFF
from gbsa_pipeline.parametrization_models import (
    ParametrisedComplex,
    ParametrizationInput,
    _write_crystal_waters_pdb,
)

logger = logging.getLogger(__name__)

# OpenMM XML files (shipped with OpenMM / AmberTools) for each protein FF.
# ff14SB is bundled with OpenMM; ff19SB and ff99SB-ILDN require the XML
# files distributed with AmberTools 24+.
_PROTEIN_FF_XML: dict[ProteinFF, list[str]] = {
    ProteinFF.FF14SB: ["amber14-all.xml"],
    ProteinFF.FF19SB: ["amber/protein.ff19SB.xml"],
    ProteinFF.FF99SB: ["amber/protein.ff99SBildn.xml"],
}

# GAFF version strings accepted by GAFFTemplateGenerator.
# Verify against the version of openmmforcefields installed in your environment.
_GAFF_FF_VERSION: dict[LigandFF, str] = {
    LigandFF.GAFF: "gaff-1.81",
    LigandFF.GAFF2: "gaff-2.11",
}


def _assign_nagl_charges_direct(mol: Molecule) -> None:
    """Assign AM1-BCC charges via the openff-nagl GNNModel API, bypassing the toolkit wrapper.

    The OpenFF toolkit's NAGL toolkit wrapper may not register in all environments
    (e.g. SLURM compute nodes). This function loads the model directly and writes
    charges back to the molecule, normalised to the molecular formal charge sum.
    """
    import numpy as np  # noqa: PLC0415
    import openff.nagl as _nagl  # noqa: PLC0415
    import openff.nagl_models as _nm  # noqa: PLC0415
    from openff.units import unit as _unit  # noqa: PLC0415

    model_path = str(_nm.get_model("openff-gnn-am1bcc-1.0.0.pt"))
    model = _nagl.GNNModel.load(model_path, eval_mode=True)
    charges_dict = model.compute_properties(mol)
    charges = charges_dict["am1bcc_charges"].astype(float)
    formal_sum = float(sum(a.formal_charge.m for a in mol.atoms))
    charges -= (float(np.sum(charges)) - formal_sum) / len(charges)
    mol.partial_charges = _unit.Quantity(charges, _unit.elementary_charge)


def _parametrize_openmm(inp: ParametrizationInput) -> ParametrisedComplex:
    work_dir = inp.work_dir or Path(tempfile.mkdtemp(prefix="gbsa_param_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Protein -------------------------------------------------------
    logger.debug("Loading protein PDB: %s …", inp.protein_pdb)
    crystal_waters_pdb = _write_crystal_waters_pdb(
        inp.protein_pdb,
        work_dir / "crystal_waters.pdb",
    )
    if crystal_waters_pdb is None:
        logger.debug("No crystallographic waters found in protein PDB.")
    else:
        logger.debug("Crystal waters written → %s.", crystal_waters_pdb)

    protein_modeller = _load_pdb_as_modeller(inp.protein_pdb)
    n_removed = _delete_residues_by_name(protein_modeller, WATER_RESIDUE_NAMES)
    if n_removed:
        logger.debug(
            "Removed %d crystallographic water residue(s) from protein topology.",
            n_removed,
        )
    logger.debug(
        "Protein PDB loaded without crystal waters (%d atoms).",
        protein_modeller.topology.getNumAtoms(),
    )

    # --- Ligand --------------------------------------------------------
    logger.debug("Loading ligand SDF: %s …", inp.ligand_sdf)
    ligand = Molecule.from_file(str(inp.ligand_sdf), allow_undefined_stereo=True)
    if not ligand.conformers:
        raise ValueError(
            f"Ligand SDF '{inp.ligand_sdf}' contains no 3-D conformers. "
            "Provide an SDF file with embedded 3-D coordinates."
        )
    logger.debug(
        "Ligand loaded (%d atoms, %d conformers).",
        ligand.n_atoms,
        len(ligand.conformers),
    )

    logger.debug("Assigning partial charges (method=%s) …", inp.config.charge_method.value)
    kwargs: dict[str, Any] = {
        "partial_charge_method": inp.config.charge_method.value,
        "normalize_partial_charges": True,
        "use_conformers": ligand.conformers,
    }
    if inp.net_charge is not None:
        kwargs["partial_charges"] = None  # reset; net_charge is passed separately
    ligand.assign_partial_charges(**kwargs)
    logger.debug("Partial charges assigned.")

    # --- Cofactors -----------------------------------------------------
    cofactors: list[Molecule] = []
    for cof_path in inp.cofactor_sdfs:
        logger.debug("Loading cofactor SDF: %s …", cof_path)
        cof = Molecule.from_file(str(cof_path), allow_undefined_stereo=True)
        if not cof.conformers:
            raise ValueError(
                f"Cofactor SDF '{cof_path}' contains no 3-D conformers. "
                "Provide an SDF file with embedded 3-D coordinates."
            )
        logger.debug(
            "Cofactor loaded (%d atoms, %d conformers).",
            cof.n_atoms,
            len(cof.conformers),
        )
        # Cofactors are parametrized like the ligand (GAFF2) but their charges
        # are assigned with NAGL (graph-neural-net), NEVER AM1-BCC. Cofactors
        # (ADP, NAD, FAD, ...) are large and/or highly charged, and AM1-BCC's
        # sqm geometry optimization stalls or crashes on exactly such molecules
        # (the failure first seen on a strained 1A5H ligand). NAGL needs no QM
        # step, so it is both fast and robust here.
        logger.info(
            "Assigning cofactor partial charges via NAGL (GNN; AM1-BCC sqm is "
            "avoided for large/charged cofactors) …"
        )
        _assign_nagl_charges_direct(cof)
        cofactors.append(cof)
    if cofactors:
        logger.debug("Loaded and parametrized %d cofactor(s).", len(cofactors))

    # --- Force field ---------------------------------------------------
    protein_xmls = _PROTEIN_FF_XML[inp.config.protein_ff]
    extra_xmls = [str(p) for p in inp.config.extra_ff_files]
    logger.debug(
        "Building force field (protein=%s, extra=%d files) …",
        inp.config.protein_ff.value,
        len(extra_xmls),
    )
    forcefield = ForceField(*protein_xmls, *extra_xmls)
    logger.debug("Force field built.")

    logger.debug(
        "Registering GAFF template generator (%s) …",
        _GAFF_FF_VERSION[inp.config.ligand_ff],
    )
    gaff = GAFFTemplateGenerator(
        molecules=[ligand, *cofactors],
        forcefield=_GAFF_FF_VERSION[inp.config.ligand_ff],
        cache=None,
    )
    forcefield.registerTemplateGenerator(gaff.generator)
    logger.debug("GAFF template generator registered.")

    logger.debug("Combining protein+ligand topology …")
    modeller = Modeller(protein_modeller.topology, protein_modeller.positions)
    modeller.add(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())
    for cof in cofactors:
        modeller.add(cof.to_topology().to_openmm(), cof.conformers[0].to_openmm())
    logger.debug("Combined dry topology: %d atoms.", modeller.topology.getNumAtoms())

    logger.debug("Creating OpenMM system (may take a moment) …")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        # VERY IMPORTANT THIS INFORMATION DOES NOT GET LOST: Constraints None is safe, HBOND fails as the generated top file does not fit with the parameters
    )
    logger.debug("OpenMM system created (%d particles).", system.getNumParticles())

    logger.debug("Converting OpenMM system to ParmEd structure …")
    structure = pmd.openmm.load_topology(modeller.topology, system, modeller.positions)
    logger.debug("ParmEd structure ready (%d atoms).", len(structure.atoms))

    gro_file = work_dir / "complex.gro"
    top_file = work_dir / "complex.top"
    gro_file.unlink(missing_ok=True)
    top_file.unlink(missing_ok=True)
    logger.debug("Writing GROMACS topology → %s …", top_file)
    structure.save(str(top_file), format="gromacs")
    logger.debug("Writing GROMACS coordinates → %s …", gro_file)
    structure.save(str(gro_file))
    logger.debug("GROMACS files written.")

    complex = ParametrisedComplex(
        gro_file=gro_file,
        top_file=top_file,
        config=inp.config,
        forcefield=forcefield,
        parmed_structure=structure,
        crystal_waters_pdb=crystal_waters_pdb,
    )

    return complex
