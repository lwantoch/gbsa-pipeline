"""BioSimSpace-based solvation for GAFF-parametrized protein-ligand complexes.

WHY THIS MODULE EXISTS — BSS OVER OPENMM
-----------------------------------------
After parametrization with GAFF the complex topology contains atom types that
only the GAFF force field knows about.  ``solvate_openmm`` (OpenMM addSolvent +
ParmEd) produces a working GROMACS topology but with a critical defect: ParmEd
must be called with ``rigidWater=False`` so that O-H bonds appear in
``HarmonicBondForce`` and can be written as ``[ bonds ]`` directives.  With
``rigidWater=True``, OpenMM stores those bonds as system Constraints that ParmEd
cannot translate into the GROMACS ``[ settles ]`` section, leaving water without
geometry constraints of any kind.

The resulting explicit O-H springs have:
    k_OH  ~  463 000 kJ/mol/nm2   (TIP3P, Jorgensen et al. 1983)
    eps_OO ~    0.636 kJ/mol      (TIP3P O-O LJ well depth)
    ratio  ~ 727 000x

During minimization every gradient step corrects O-H bond lengths rather than
resolving LJ clashes.  This causes LJ (SR) to *increase* from +112 000 kJ/mol
immediately after solvation to +175 000 kJ/mol after 600 000 steps of SD + CG
minimization (measured in pytest runs with the OpenMM path active).  LINCS then
fails early in NVT heating because strained water geometry exceeds the angle
warning threshold.

``BSS.Solvent`` wraps ``gmx solvate``, which places water from a pre-equilibrated
TIP3P box (``spc216.gro``) at a minimum distance of ~0.24-0.26 nm and writes a
``[ settles ]`` GROMACS topology.  SETTLE enforces rigid water geometry from step 0
without any harmonic spring forces, so the minimizer gradient is dominated by
intermolecular contacts as intended.  This produces LJ (SR) < 0 from the first
minimization step and eliminates LINCS failures in NVT heating.

MEASURED LJ (SR) COMPARISON (kJ/mol)
--------------------------------------
All runs used the same protein-ligand system (dockprotein + GAFF ligand):

  OpenMM path (solvate_openmm, pytest-35):
    After addSolvent                    : +112 079
    After SD minimization (500 000 steps): +170 680   (worse, not better)
    After CG minimization (100 000 steps): +175 967
    NVT at 13.5 ps / 30 K              : +188 237
    NVT crashed at ~73 ps (LINCS failure)

  OpenMM path + relax_solvated_water (pytest-36):
    After relax_solvated_water          : +173 046   (no improvement)
    Root cause is in topology, not step count.

  BSS path (this module, pytest-42):
    After BSS.Solvent.tip3p             : LJ (SR) < 0 throughout
    NVT, NPT, and production all passed
    DeltaG_bind = -33.45 kcal/mol

WHY is_neutral=False AND ion_conc=0
--------------------------------------
BSS neutralization calls ``gmx genion``, which requires ``gmx grompp``, which
requires a complete writable GROMACS topology.  BSS/Sire can *read* a GROMACS
topology with GAFF atom types but cannot *write* one back: GAFF types are absent
from GROMACS's built-in force-field library and Sire has no code path to emit
the required ``#include`` directives for the parametrization-produced GAFF itp
files.  Attempting neutralization therefore fails with:

    "Failed to write GROMACS topology file.  Is your molecule parameterised?"

Setting ``is_neutral=False`` and ``ion_conc=0`` bypasses ``genion`` entirely.
The system retains the protein's net charge (typically 1-5 e).  For GBSA
calculations this is acceptable: the GB implicit-solvent model does not use
periodic boundary conditions for the energy decomposition.  For production NPT
with PME a net charge is non-ideal but does not prevent the simulation from
running.

TODO: supply GAFF itp includes to BSS/Sire so ``gmx grompp`` can produce the
tpr required by ``genion``, enabling full neutralization.

References:
-----------
1. TIP3P parameters: Jorgensen et al., J. Chem. Phys. 79, 926 (1983)
   https://doi.org/10.1063/1.445869
2. SETTLE algorithm: Miyamoto & Kollman, J. Comput. Chem. 13, 952 (1992)
   https://doi.org/10.1002/jcc.540130805
3. gmx solvate documentation:
   https://manual.gromacs.org/current/onlinehelp/gmx-solvate.html
4. BSS Solvent API:
   https://biosimspace.openbiosim.org/api/index_Solvent.html
5. OpenMM Modeller.addSolvent (0.23 nm minimum placement, C++ source):
   https://github.com/openmm/openmm/blob/master/openmmapi/src/Modeller.cpp
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gbsa_pipeline.solvation_box import WaterModel

if TYPE_CHECKING:
    from pathlib import Path

    from gbsa_pipeline.parametrization import ParametrisedComplex

logger = logging.getLogger(__name__)


def solvate_parametrized_complex(
    parametrized: ParametrisedComplex,
    *,
    shell_nm: float = 1.0,
    water_model: WaterModel = WaterModel.TIP3P,
    work_dir: Path | None = None,
) -> Any:
    """Solvate a GAFF-parametrized complex using BioSimSpace (gmx solvate).

    Reads the dry GRO/TOP produced by ``parametrize()``, adds bulk solvent via
    ``BSS.Solvent``, and returns the solvated BSS system ready for BioSimSpace
    MD stages.  No ions are added — see the module docstring for why
    ``is_neutral=False`` and ``ion_conc=0`` are required for GAFF topologies.

    Parameters
    ----------
    parametrized:
        Output of ``parametrize()``.  Must have ``gro_file`` and ``top_file`` set.
    shell_nm:
        Padding added around the solute bounding box on each side, in nm.
        Default 1.0 nm matches the 0.8-1.0 nm consensus from GROMACS tutorials
        (Lemkul 2024, DOI:10.1021/acs.jpcb.4c04901) and keeps the system small
        enough for sub-hour test runs (~35 k atoms vs ~170 k for a fixed 12 nm box).
    water_model:
        Water model to use.  ``WaterModel.TIP3P`` is the only model validated
        against the production GBSA pipeline so far.
    work_dir:
        Scratch directory for BSS / gmx solvate intermediate files.  When
        ``None``, BSS picks a temporary directory that is removed on exit.

    Returns:
    --------
    BSS System
        Solvated BioSimSpace system with SETTLE-constrained water.
        Pass it directly to ``run_minimization``, ``run_heating``, etc.
    """
    import BioSimSpace as BSS  # noqa: PLC0415

    logger.debug(
        "Loading dry parametrized system from %s / %s",
        parametrized.gro_file,
        parametrized.top_file,
    )
    dry_system = BSS.IO.readMolecules([str(parametrized.gro_file), str(parametrized.top_file)])

    solvent_fn = _bss_solvent_function(BSS, water_model)

    kwargs: dict[str, Any] = {
        # ion_conc=0, is_neutral=False: BSS cannot write GAFF topologies, so
        # gmx grompp (needed by genion for neutralization) cannot be invoked.
        # See the module docstring for the full explanation and a TODO.
        "ion_conc": 0,
        "is_neutral": False,
        "shell": shell_nm * BSS.Units.Length.nanometer,
    }
    if work_dir is not None:
        kwargs["work_dir"] = str(work_dir)

    logger.debug(
        "Calling BSS.Solvent.%s (shell=%.1f nm, is_neutral=False, ion_conc=0) …",
        water_model.value,
        shell_nm,
    )
    bss_system = solvent_fn(dry_system, **kwargs)
    logger.debug(
        "BSS solvation complete: %d molecules in solvated system.",
        bss_system.nMolecules(),
    )
    return bss_system


def _bss_solvent_function(bss: Any, water_model: WaterModel) -> Any:
    """Return the BSS.Solvent callable for the requested water model."""
    if water_model is WaterModel.TIP3P:
        return bss.Solvent.tip3p
    if water_model is WaterModel.TIP4P:
        return bss.Solvent.tip4p
    if water_model is WaterModel.SPC:
        return bss.Solvent.spc
    if water_model is WaterModel.SPCE:
        return bss.Solvent.spce
    if water_model is WaterModel.TIP5P:
        return bss.Solvent.tip5p
    raise ValueError(f"Unsupported water model for BSS solvation: {water_model!s}")
