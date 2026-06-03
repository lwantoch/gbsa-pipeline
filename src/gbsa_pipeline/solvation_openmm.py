"""OpenMM/ParmEd solvation and water-relaxation helpers.

``solvate_openmm`` builds a solvated GROMACS system from an already
parametrized complex using OpenMM ``Modeller.addSolvent`` and ParmEd.
``relax_solvated_water`` is a post-solvation fix for the LJ clash problem
inherent in OpenMM placement: ``addSolvent`` enforces a 0.23 nm minimum
distance between new water oxygens and existing heavy atoms, but the
Lennard-Jones repulsive core for typical C-O pairs has sigma ~0.32 nm,
so newly placed water molecules can sit inside the repulsive wall.  The
relaxation function freezes all non-water atoms and runs a short
steepest-descent minimisation so water can escape those close contacts
before the first MD step.

Note on force field compatibility
----------------------------------
ParmEd writes explicit harmonic O-H bond springs (k ~463 000 kJ/mol/nm2)
when rigidWater=False is used in the OpenMM system.  During a frozen-solute
steepest-descent run these springs dominate the gradient (k/epsilon ratio
~727 000) and can prevent water from moving away from LJ clashes.  If the
LJ energy does not decrease after relaxation the caller should switch to a
solvation path that uses pre-equilibrated water boxes (e.g. BSS.Solvent,
which calls gmx solvate and places water at a safe minimum distance with
SETTLE constraints intact).

Crystal-water handling
-----------------------
``_restore_crystal_waters_before_solvation`` re-adds crystallographic waters
extracted during parametrization before bulk solvent is placed.  A clash
filter (``_CLASH_CUTOFF_NM = 0.25 nm``) drops any crystal water whose oxygen
overlaps with a ligand or protein heavy atom so that docked poses do not
create irrecoverable geometry at the binding site.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import IO, TYPE_CHECKING, Any

import parmed as pmd
from openmm import Vec3
from openmm import unit as mm_unit
from openmm.app import ForceField, Modeller, NoCutoff

from gbsa_pipeline._gro_io import _parse_gro
from gbsa_pipeline._openmm_utils import (
    _heavy_atom_coords,
    _load_waters_with_hydrogens,
    _oxygen_atom_entries,
    _write_modeller_pdb,
)
from gbsa_pipeline._spatial import _find_clashing_residues
from gbsa_pipeline.solvation_box import BoxShape, SolvatedComplex, SolvationParams, WaterModel

if TYPE_CHECKING:
    from pathlib import Path

    from gbsa_pipeline.parametrization import ParametrisedComplex

logger = logging.getLogger(__name__)



def solvate_openmm(
    parametrized: ParametrisedComplex,
    params: SolvationParams,
    output_gro: Path,
    output_top: Path,
) -> SolvatedComplex:
    """Solvate a parametrised complex with OpenMM + ParmEd.

    Reuses the ``ForceField`` and ParmEd ``Structure`` carried by
    *parametrized*, so ligand charges and protein-ligand parameters are not
    regenerated. If parametrization extracted crystallographic waters, those
    waters are restored into the OpenMM modeller before bulk solvent is added.
    This makes retained waters part of the pre-solvation system, so newly
    generated solvent is placed around protein, ligand, and crystal waters
    together. The function writes GROMACS ``.gro`` and ``.top`` files and
    returns a small object carrying the paths and in-memory ParmEd structure.
    """
    if parametrized.forcefield is None or parametrized.parmed_structure is None:
        raise ValueError(
            "solvate_openmm requires parametrized.forcefield and "
            "parametrized.parmed_structure to be set. "
            "Use parametrize() (not load_amber_complex) before calling this function."
        )

    output_gro.parent.mkdir(parents=True, exist_ok=True)

    water_model = params.water_model
    box_shape = params.shape

    # ------------------------------------------------------------------
    # 1. Reuse the in-memory ParmEd structure from parametrization.
    #    This is the dry protein-ligand complex, with all non-water
    #    protein and ligand parameters already assigned.
    # ------------------------------------------------------------------
    existing: Any = parametrized.parmed_structure
    n_orig = len(existing.atoms)
    logger.debug("Using in-memory ParmEd structure (%d atoms).", n_orig)

    # ------------------------------------------------------------------
    # 2. Extend the existing ForceField with water templates.
    #    GAFF is already registered with pre-assigned ligand charges,
    #    so AM1-BCC will not re-run.
    # ------------------------------------------------------------------
    ff: ForceField = parametrized.forcefield
    logger.debug("Loading water FF (%s) into existing ForceField …", water_model.openmm_xml)
    ff.loadFile(water_model.openmm_xml)

    # ------------------------------------------------------------------
    # 3. Build a complete pre-solvation modeller.
    #    Crystal waters are restored before addSolvent so the bulk solvent
    #    placement sees them as existing atoms and avoids overlaps.
    # ------------------------------------------------------------------
    logger.debug("Creating Modeller from existing topology …")
    modeller = Modeller(existing.topology, existing.positions)

    restored_waters_pdb = _restore_crystal_waters_before_solvation(
        modeller=modeller,
        forcefield=ff,
        crystal_waters_pdb=parametrized.crystal_waters_pdb,
        output_pdb=output_gro.parent / "restored_crystal_waters.pdb",
    )
    if restored_waters_pdb is not None:
        logger.debug("Restored crystallographic waters from %s.", restored_waters_pdb)

    # ------------------------------------------------------------------
    # 4. Add bulk solvent and ions via OpenMM Modeller.
    # ------------------------------------------------------------------
    kwargs: dict[str, Any] = {
        "model": water_model.gmx_water_name,
        "neutralize": params.neutralize,
    }
    if params.ion_concentration is not None:
        kwargs["ionicStrength"] = params.ion_concentration * mm_unit.molar
    if params.padding is not None:
        kwargs["padding"] = params.padding * mm_unit.nanometer
    else:
        kwargs["boxSize"] = Vec3(params.box_size, params.box_size, params.box_size) * mm_unit.nanometer
    if box_shape is BoxShape.TRUNCATED_OCTAHEDRON:
        kwargs["boxShape"] = "octahedron"

    logger.debug(
        "Adding solvent (model=%s, %s, box_shape=%s) …",
        kwargs["model"],
        f"padding={params.padding} nm" if params.padding is not None else f"box_size={params.box_size} nm",
        box_shape,
    )
    modeller.addSolvent(ff, **kwargs)
    logger.debug("Solvated topology: %d atoms.", modeller.topology.getNumAtoms())

    # ------------------------------------------------------------------
    # 5. Build full OpenMM system → ParmEd structure.
    #    rigidWater=False keeps O-H bonds in HarmonicBondForce so ParmEd
    #    can resolve water geometry when writing the GROMACS topology.
    # ------------------------------------------------------------------
    logger.debug("Creating solvated OpenMM system …")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    logger.debug("Converting to ParmEd structure …")
    structure = pmd.openmm.load_topology(modeller.topology, system, modeller.positions)

    # ------------------------------------------------------------------
    # 7. Write GROMACS gro + top.
    # ------------------------------------------------------------------
    logger.debug("Writing GROMACS topology → %s …", output_top)
    structure.save(str(output_top), format="gromacs", overwrite=True)
    logger.debug("Writing GROMACS coordinates → %s …", output_gro)
    structure.save(str(output_gro), overwrite=True)
    logger.debug("Solvated files written.")

    return SolvatedComplex(
        gro_file=output_gro,
        top_file=output_top,
        parmed_structure=structure,
    )


_WATER_RESIDUES: frozenset[str] = frozenset({"HOH", "WAT", "SOL"})

_WATER_RELAX_MDP = """\
integrator          = steep
nsteps              = {nsteps}
emtol               = 10.0
emstep              = 0.001
cutoff-scheme       = Verlet
nstlist             = 20
rlist               = 1.2
coulombtype         = PME
rcoulomb            = 1.2
fourierspacing      = 0.16
pme-order           = 4
vdwtype             = Cut-off
vdw-modifier        = Force-switch
rvdw-switch         = 1.0
rvdw                = 1.2
pbc                 = xyz
freezegrps          = non-Water
freezedim           = Y Y Y
; SETTLE keeps water geometry rigid so SD follows the LJ gradient (not the
; stiff ParmEd OH spring gradient that would otherwise dominate and push
; oxygens in the wrong direction).
constraints         = h-bonds
constraint-algorithm = LINCS
lincs-order         = 4
lincs-warnangle     = 30
nstlog              = 200
nstenergy           = 200
"""


def relax_solvated_water(gro: Path, top: Path, work_dir: Path, *, nsteps: int = 2000) -> None:
    """Run a frozen-solute GROMACS SD to push water out of the solute LJ core.

    OpenMM ``addSolvent`` places water with a 0.23 nm minimum distance from
    solute heavy atoms, which can be inside the LJ repulsive core (sigma ~
    0.32 nm for C-O pairs). This function freezes all non-water atoms and runs
    steepest-descent minimisation so water can move to lower-energy positions.
    The relaxed coordinates overwrite *gro* in place; the topology is unchanged.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    atoms = _parse_gro(gro)
    all_indices = [a.atom_idx for a in atoms]
    nonwater_atoms = [a.atom_idx for a in atoms if a.res_name not in _WATER_RESIDUES]

    ndx = work_dir / "freeze.ndx"
    with ndx.open("w") as f:
        f.write("[ System ]\n")
        _write_ndx_indices(f, all_indices)
        f.write("\n[ non-Water ]\n")
        _write_ndx_indices(f, nonwater_atoms)

    mdp = work_dir / "water_relax.mdp"
    mdp.write_text(_WATER_RELAX_MDP.format(nsteps=nsteps))

    tpr = work_dir / "water_relax.tpr"
    grompp = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "gmx",
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(gro),
            "-p",
            str(top),
            "-n",
            str(ndx),
            "-o",
            str(tpr),
            "-maxwarn",
            "10",
        ],
        capture_output=True,
        cwd=work_dir,
        check=False,
    )
    if grompp.returncode != 0:
        raise RuntimeError(f"gmx grompp failed during water relaxation:\n{grompp.stderr.decode()}")

    mdrun = subprocess.run(
        ["gmx", "mdrun", "-deffnm", "water_relax"],  # noqa: S607
        capture_output=True,
        cwd=work_dir,
        check=False,
    )
    if mdrun.returncode != 0:
        raise RuntimeError(f"gmx mdrun failed during water relaxation:\n{mdrun.stderr.decode()}")

    relaxed_gro = work_dir / "water_relax.gro"
    if not relaxed_gro.exists():
        raise FileNotFoundError(f"Water relaxation output not found: {relaxed_gro}")

    shutil.copy(relaxed_gro, gro)
    logger.debug("Water relaxation complete; updated %s.", gro)


_NDX_INDICES_PER_LINE = 15


def _write_ndx_indices(f: IO[str], indices: Any) -> None:
    """Write atom indices into an open GROMACS NDX file, 15 per line."""
    batch: list[str] = []
    for idx in indices:
        batch.append(str(idx))
        if len(batch) == _NDX_INDICES_PER_LINE:
            f.write(" ".join(batch) + "\n")
            batch = []
    if batch:
        f.write(" ".join(batch) + "\n")


_CLASH_CUTOFF_NM: float = 0.25


def _restore_crystal_waters_before_solvation(
    *,
    modeller: Modeller,
    forcefield: ForceField,
    crystal_waters_pdb: Path | None,
    output_pdb: Path,
) -> Path | None:
    """Restore extracted crystallographic waters before bulk solvation.

    The parametrization stage stores crystal waters separately because the
    protein-ligand system is parameterized dry. This helper uses OpenMM's own
    ``Modeller`` machinery to add missing hydrogens to those waters, writes a
    small inspection PDB, and adds the completed waters to the pre-solvation
    modeller before ``addSolvent`` is called. Waters whose oxygen is within
    ``_CLASH_CUTOFF_NM`` of any existing heavy atom are dropped so that
    binding-site crystal waters do not clash with the docked ligand.
    ``None`` is returned when no retained-water file is available.
    """
    if crystal_waters_pdb is None:
        return None

    if not crystal_waters_pdb.exists():
        _remove_stale_file(output_pdb)
        logger.debug("Crystal water file not found: %s.", crystal_waters_pdb)
        return None

    water_modeller = _load_waters_with_hydrogens(crystal_waters_pdb, forcefield)

    clashing = _find_clashing_residues(
        _oxygen_atom_entries(water_modeller),
        _heavy_atom_coords(modeller),
        _CLASH_CUTOFF_NM,
    )
    if clashing:
        logger.debug(
            "Dropping %d crystal water(s) clashing with existing atoms (cutoff %.2f nm).",
            len(clashing),
            _CLASH_CUTOFF_NM,
        )
        water_modeller.delete(list(clashing))

    n_restored = water_modeller.topology.getNumResidues()
    _write_modeller_pdb(water_modeller, output_pdb)
    modeller.add(water_modeller.topology, water_modeller.positions)

    logger.debug("Restored %d crystal water(s).", n_restored)
    return output_pdb


def _remove_stale_file(path: Path) -> None:
    """Remove a generated file if it exists.

    Persistent integration-test directories can otherwise keep stale artefacts
    from an earlier run, which makes visual inspection misleading. The helper is
    intentionally small and silent because the absence of a generated water file
    is valid when no crystal waters were retained. Only ``FileNotFoundError`` is
    suppressed; other filesystem errors should still surface. This keeps stale
    cleanup local to the generated restore artefact.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        return
