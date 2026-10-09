"""Explicit GROMACS stage builders for the GBSA pipeline.

Each builder returns a validated :class:`~gbsa_pipeline.mdp.GromacsParams`
instance that is ready to pass to :func:`~gbsa_pipeline.gmx.run_stage`.  The
seven stages replace the implicit MDP generation that
:class:`BSS.Protocol.Equilibration`/:class:`BSS.Protocol.Production` used to
perform behind :class:`BSS.Process.Gromacs`.  Explicit builders make every MDP
field visible to the pipeline reviewer and remove the fragile
``_apply_gromacs_params_to_config`` layer in :mod:`gbsa_pipeline.md`.

AMBER nonbonded defaults
------------------------
All seven stages share the AMBER-compatible nonbonded block verified against
``grompp -maxwarn 0`` on the project test system:

* ``rcoulomb = rvdw = 1.0`` nm
* ``coulomb_modifier = vdw_modifier = Potential-shift``
* ``coulombtype = PME``, ``vdwtype = Cut-off``
* ``dispcorr = EnerPres`` (long-range dispersion with pressure contribution)
* ``fourierspacing = 0.12`` nm (replaces BSS's hard-coded ``fourier-nx/ny/nz = 52``)
* ``mts = no`` (incompatible with ``sd``/``steep`` integrators and with h-bonds
  constraints)
* ``constraints = h-bonds`` (prerequisite for ``dt = 2 fs`` in NPT/production)

Position restraints
-------------------
The restrained stages toggle restraint groups via ``define = -D<POSRES_FLAG>``.
Three flags are supported, matching the posre split written by the ParmEd
topology layer:

* ``POSRES_HEAVY`` — all non-hydrogen atoms (used by :func:`relax`).
* ``POSRES_BB`` — protein backbone (used by :func:`heating`,
  :func:`npt_restrained` with ``restraint="backbone"``).
* ``POSRES_LIPID`` — lipid phosphate head groups (unioned with ``POSRES_BB``
  for membrane runs; see :func:`heating`/:func:`npt_restrained`).

Membrane variant
----------------
The pressure-coupled stages (:func:`npt_restrained`, :func:`npt`, :func:`prod`)
accept ``membrane=True`` to switch pressure coupling from isotropic to
semiisotropic C-rescale.  Temperature coupling stays on a single ``System``
group; multi-group thermostats (``SOLU/MEMB/SOLV``) would require promoting
:attr:`GromacsParams.tau_t`/``ref_t`` to accept lists and are not needed to
reproduce the current pipeline's behaviour.
"""

from __future__ import annotations

from typing import Any

from gbsa_pipeline.mdp import GromacsParams, SemiisotropicValue
from gbsa_pipeline.mdp_enum import (
    Barostat,
    Constraints,
    CoulombModifier,
    CoulombType,
    DispCorr,
    Integrator,
    NghCutoffScheme,
    PCoupleType,
    Thermostat,
    VDWModifier,
    VDWType,
    VelocityGeneration,
)

# ---------------------------------------------------------------------------
# Shared AMBER nonbonded block
# ---------------------------------------------------------------------------
# Reused as ``**_AMBER_NONBONDED`` by every builder below.  Keep the keys here
# aligned with the module docstring; changes here propagate to every stage.

_AMBER_NONBONDED: dict[str, Any] = {
    "cutoff_scheme": NghCutoffScheme.VERLET,
    "coulombtype": CoulombType.PME,
    "coulomb_modifier": CoulombModifier.POTENTIAL_SHIFT,
    "rcoulomb": 1.0,
    "vdwtype": VDWType.CUT_OFF,
    "vdw_modifier": VDWModifier.POTENTIAL_SHIFT,
    "rvdw": 1.0,
    "dispcorr": DispCorr.ENERGY_PRESSURE,
    "fourierspacing": 0.12,
    "mts": False,
    "constraints": Constraints.HYDROGENS_BONDS,
}

# Multi-group PME meshes (``fourier_nx/ny/nz``) are not needed when
# ``fourierspacing`` is set: grompp picks the mesh automatically from the box
# size.  Leaving the hard-coded 52 defaults would silently override the mesh
# for every box size, so they are explicitly cleared on each stage via
# ``_clear_mesh`` below.

_CLEAR_MESH: dict[str, int] = {
    "fourier_nx": 0,
    "fourier_ny": 0,
    "fourier_nz": 0,
}


def _common(**overrides: Any) -> dict[str, Any]:
    """Merge the AMBER nonbonded block and PME-mesh clear with per-stage overrides."""
    return {**_AMBER_NONBONDED, **_CLEAR_MESH, **overrides}


# ---------------------------------------------------------------------------
# Minimisation stages
# ---------------------------------------------------------------------------


def steep(nsteps: int = 50_000, emtol: float = 1000.0, emstep: float = 0.01) -> GromacsParams:
    """Steepest-descent energy minimisation.

    Default ``emtol = 1000 kJ/mol/nm`` matches the GROMACS tutorial for a
    solvated protein+lipid system.  ``nsteps`` is an upper bound; grompp exits
    as soon as the force norm drops below ``emtol``.
    """
    return GromacsParams(
        **_common(
            integrator=Integrator.STEEP,
            nsteps=nsteps,
            emtol=emtol,
            emstep=emstep,
            nstlog=100,
            nstenergy=100,
            nstxout_compressed=0,
        )
    )


def cg(nsteps: int = 10_000, emtol: float = 100.0, nstcgsteep: int = 1000) -> GromacsParams:
    """Conjugate-gradient minimisation.

    Tighter default ``emtol`` than :func:`steep` because CG is used as a
    polish step on the output of steep.

    Constraints are explicitly disabled (``constraints=none``) — the shared
    AMBER nonbonded block defaults to ``h-bonds`` for MD stability, but
    conjugate-gradient line search is numerically unstable when LINCS
    actively constrains bonds (the gradient is discontinuous at the
    constraint surface and CG trips SIGSEGV in mdrun with
    ``"Update groups can not be used for this system because atoms that
    are (in)directly constrained together are interdispersed with other
    atoms"`` on a stressed starting geometry).  Minimisation does not need
    constraints because there is no integrator step size to worry about;
    the next stage (relax) re-enables them for MD.
    """
    return GromacsParams(
        **_common(
            integrator=Integrator.CONJUGATE_GRADIENT,
            nsteps=nsteps,
            emtol=emtol,
            nstcgsteep=nstcgsteep,
            constraints=Constraints.NONE,
            nstlog=100,
            nstenergy=100,
            nstxout_compressed=0,
        )
    )


# ---------------------------------------------------------------------------
# Solvent/ion relaxation (SD integrator with heavy-atom restraints)
# ---------------------------------------------------------------------------


def relax(nsteps: int = 10_000, dt: float = 0.001, temperature: float = 300.0) -> GromacsParams:
    """Short SD run with heavy-atom restraints to settle solvent and ions.

    Uses the ``sd`` (stochastic dynamics) integrator — temperature coupling is
    folded into the propagator, so no separate thermostat is needed.  Fresh
    Maxwell-Boltzmann velocities are drawn at ``temperature`` (default 300 K)
    because this is the first step after minimisation and no checkpoint
    carries over velocities.  Heavy-atom restraints are enabled via
    ``define = -DPOSRES_HEAVY``; the ParmEd topology layer must emit a matching
    ``#ifdef POSRES_HEAVY`` block.
    """
    return GromacsParams(
        **_common(
            integrator=Integrator.LEAP_FROG_STOCHASTIC,
            nsteps=nsteps,
            dt=dt,
            tcoupl=Thermostat.NO,  # sd handles temperature internally
            ref_t=temperature,
            tau_t=1.0,
            gen_vel=VelocityGeneration.YES,
            gen_temp=temperature,
            define="-DPOSRES_HEAVY",
            nstlog=500,
            nstenergy=500,
            nstxout_compressed=0,
        )
    )


# ---------------------------------------------------------------------------
# NVT heating with V-rescale annealing
# ---------------------------------------------------------------------------


def heating(
    nsteps: int = 50_000,
    dt: float = 0.002,
    temperature_start: float = 50.0,
    temperature_end: float = 300.0,
    *,
    restrain_lipid: bool = False,
) -> GromacsParams:
    """NVT heating from ``temperature_start`` → ``temperature_end`` with backbone restraints.

    Uses :class:`Thermostat.VRESCALE` with a linear two-point annealing schedule
    over the full stage.  Backbone position restraints keep the protein in its
    equilibrated frame while waters/ions follow the heating; for a membrane run
    pass ``restrain_lipid=True`` to additionally restrain the lipid phosphate
    heads (prevents flip-flop before the barostat has equilibrated).  Fresh
    velocities are drawn at ``temperature_start`` because this is the first MD
    stage; subsequent stages carry velocities via ``gmx grompp -t``.

    The ``annealing_time`` end-point matches ``nsteps * dt`` so the final frame
    is at ``temperature_end`` exactly; any additional steps would hold at
    ``temperature_end`` (GROMACS extrapolates the last point).
    """
    define_flags = "-DPOSRES_BB"
    if restrain_lipid:
        define_flags += " -DPOSRES_LIPID"
    sim_time_ps = nsteps * dt
    return GromacsParams(
        **_common(
            integrator=Integrator.LEAP_FROG,
            nsteps=nsteps,
            dt=dt,
            tcoupl=Thermostat.VRESCALE,
            tc_grps="System",
            ref_t=temperature_end,
            tau_t=0.1,
            gen_vel=VelocityGeneration.YES,
            gen_temp=temperature_start,
            annealing="single",
            annealing_npoints=2,
            annealing_time=f"0 {sim_time_ps:.1f}",
            annealing_temp=f"{temperature_start:.1f} {temperature_end:.1f}",
            define=define_flags,
            nstlog=500,
            nstenergy=500,
            nstxout_compressed=5000,
        )
    )


# ---------------------------------------------------------------------------
# NPT equilibration (restrained and free)
# ---------------------------------------------------------------------------
def _barostat_block(*, membrane: bool, ref_p: float = 1.0) -> dict[str, Any]:
    """Return the pcoupl/pcoupltype/ref_p/compressibility block for NPT stages."""
    if membrane:
        return {
            "pcoupl": Barostat.CRESCALE,
            "pcoupltype": PCoupleType.SEMIISOTROPIC,
            "tau_p": 5.0,
            "ref_p": SemiisotropicValue(lateral=ref_p, normal=ref_p),
            "compressibility": SemiisotropicValue(lateral=4.5e-5, normal=4.5e-5),
        }
    return {
        "pcoupl": Barostat.CRESCALE,
        "pcoupltype": PCoupleType.ISOTROPIC,
        "tau_p": 2.0,
        "ref_p": ref_p,
        "compressibility": 4.5e-5,
    }


def npt_restrained(
    nsteps: int = 50_000,
    dt: float = 0.002,
    temperature: float = 300.0,
    *,
    membrane: bool = False,
    restrain_lipid: bool = False,
    ref_p: float = 1.0,
) -> GromacsParams:
    """NPT equilibration with backbone restraints.

    ``refcoord_scaling = com`` is required under position restraints with any
    barostat: without it grompp emits
    ``WARNING: refcoord-scaling = No is not compatible with position
    restraints and a non-zero compressibility`` and the restraint reference
    coordinates drift with the box, pulling the protein apart as the barostat
    equilibrates volume.  For a membrane run, ``restrain_lipid=True`` adds
    ``POSRES_LIPID`` so lipid heads stay anchored while the box relaxes.
    """
    define_flags = "-DPOSRES_BB"
    if restrain_lipid:
        define_flags += " -DPOSRES_LIPID"
    return GromacsParams(
        **_common(
            integrator=Integrator.LEAP_FROG,
            nsteps=nsteps,
            dt=dt,
            tcoupl=Thermostat.VRESCALE,
            tc_grps="System",
            ref_t=temperature,
            tau_t=0.1,
            refcoord_scaling="com",
            define=define_flags,
            nstlog=500,
            nstenergy=500,
            nstxout_compressed=5000,
            continuation=True,
            gen_vel=VelocityGeneration.NO,
            **_barostat_block(membrane=membrane, ref_p=ref_p),
        )
    )


def npt(
    nsteps: int = 50_000,
    dt: float = 0.002,
    temperature: float = 300.0,
    *,
    membrane: bool = False,
    ref_p: float = 1.0,
) -> GromacsParams:
    """Unrestrained NPT equilibration - last stage before production."""
    return GromacsParams(
        **_common(
            integrator=Integrator.LEAP_FROG,
            nsteps=nsteps,
            dt=dt,
            tcoupl=Thermostat.VRESCALE,
            tc_grps="System",
            ref_t=temperature,
            tau_t=0.1,
            nstlog=500,
            nstenergy=500,
            nstxout_compressed=5000,
            continuation=True,
            gen_vel=VelocityGeneration.NO,
            **_barostat_block(membrane=membrane, ref_p=ref_p),
        )
    )


# ---------------------------------------------------------------------------
# Production MD
# ---------------------------------------------------------------------------


def production(
    nsteps: int = 50_000,
    dt: float = 0.002,
    temperature: float = 300.0,
    *,
    membrane: bool = False,
    ref_p: float = 1.0,
    nstxout_compressed: int = 5000,
    nstenergy: int = 5000,
) -> GromacsParams:
    """Prodction MD.

    ''nsteps'' has no default - production lenth is always set by tge config. Trajectory and energy output cadences
    default to ecvery 10 ps at dt=2 fs.
    """
    return GromacsParams(
        **_common(
            integrator=Integrator.LEAP_FROG,
            nsteps=nsteps,
            dt=dt,
            tcoupl=Thermostat.VRESCALE,
            tc_grps="System",
            ref_t=temperature,
            tau_t=0.1,
            nstlog=nstenergy,
            nstenergy=nstenergy,
            nstxout_compressed=nstxout_compressed,
            continuation=True,
            gen_vel=VelocityGeneration.NO,
            **_barostat_block(membrane=membrane, ref_p=ref_p),
        )
    )
