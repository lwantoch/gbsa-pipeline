"""Creating custom protocol with one setter per GROMACS parameter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import BioSimSpace as BSS

# your enums live in another file → import them here (adjust module path!)
from gbsa_pipeline.change_defaults_enum import (
    Barostat,
    CommMode,
    ConstraintsAlgorithms,
    CoulombModifier,
    CoulombType,
    DispCorr,
    Integrator,
    NghCutoffScheme,
    VDWModifier,
    VDWType,
)
from gbsa_pipeline.change_params import change_default_params
from gbsa_pipeline.gmx_edit_defaults import apply_changes

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from enum import StrEnum

logger = logging.getLogger(__name__)


def _enum_values(enum_cls: type[StrEnum]) -> set[str]:
    return {m.value for m in enum_cls}


# ============================================================================
# Protocol wrapper
# ============================================================================


class GromacsCustom(BSS.Protocol.Custom):
    """Thin wrapper for a custom GROMACS protocol with one setter per mdp key."""

    def __init__(self, config: str | Path):
        """Initialise the protocol from a base mdp/config file."""
        super().__init__(config)

    # ------------------------------------------------------------------------
    # Core setter
    # ------------------------------------------------------------------------

    def _set_parameter(
        self,
        parameter: str,
        value: Any,
        allowed_values: Iterable[str] | None = None,
    ) -> GromacsCustom:
        """Internal: set an mdp parameter. Optional lightweight allowed-values check."""
        key = parameter.strip()

        if hasattr(value, "value"):  # Enum/StrEnum
            value = value.value

        if allowed_values is not None:
            if not isinstance(value, str):
                raise TypeError(f"{key} must be str when allowed_values is provided")
            v = value.strip().lower()
            canonical = next((a for a in allowed_values if a.lower() == v), None)
            if canonical is None:
                raise ValueError(f"Invalid value for {key}: {value!r}")
            value = canonical

        self._parameters[key] = value
        return self

    # =========================================================================
    # Run control
    # =========================================================================

    def set_integrator(self, value: str) -> GromacsCustom:
        """Set 'integrator' (integration algorithm)."""
        allowed = _enum_values(Integrator)
        return self._set_parameter("integrator", value, allowed)

    def set_tinit(self, value: float) -> GromacsCustom:
        """Set 'tinit' (starting time, ps)."""
        return self._set_parameter("tinit", value)

    def set_dt(self, value: float) -> GromacsCustom:
        """Set 'dt' (time step, ps)."""
        return self._set_parameter("dt", value)

    def set_nsteps(self, value: int) -> GromacsCustom:
        """Set 'nsteps' (number of steps)."""
        return self._set_parameter("nsteps", value)

    def set_init_step(self, value: int) -> GromacsCustom:
        """Set 'init-step' (initial step index)."""
        return self._set_parameter("init-step", value)

    def set_simulation_part(self, value: int) -> GromacsCustom:
        """Set 'simulation-part' (part index for split runs)."""
        return self._set_parameter("simulation-part", value)

    def set_mts(self, *, value: bool) -> GromacsCustom:
        """Set 'mts' (enable multiple time stepping)."""
        return self._set_parameter("mts", value)

    def set_mass_repartition_factor(self, value: float) -> GromacsCustom:
        """Set 'mass-repartition-factor' (HMR-like mass repartition factor)."""
        return self._set_parameter("mass-repartition-factor", value)

    def set_comm_mode(self, value: str) -> GromacsCustom:
        """Set 'comm-mode' (center-of-mass motion removal mode)."""
        allowed = _enum_values(CommMode)
        return self._set_parameter("comm-mode", value, allowed)

    def set_nstcomm(self, value: int) -> GromacsCustom:
        """Set 'nstcomm' (frequency of COM motion removal)."""
        return self._set_parameter("nstcomm", value)

    def set_bd_fric(self, value: float) -> GromacsCustom:
        """Set 'bd-fric' (Brownian dynamics friction)."""
        return self._set_parameter("bd-fric", value)

    def set_ld_seed(self, value: int) -> GromacsCustom:
        """Set 'ld-seed' (random seed for stochastic dynamics)."""
        return self._set_parameter("ld-seed", value)

    # =========================================================================
    # Minimisation
    # =========================================================================

    def set_emtol(self, value: float) -> GromacsCustom:
        """Set 'emtol' (EM convergence criterion)."""
        return self._set_parameter("emtol", value)

    def set_emstep(self, value: float) -> GromacsCustom:
        """Set 'emstep' (EM step size)."""
        return self._set_parameter("emstep", value)

    def set_niter(self, value: int) -> GromacsCustom:
        """Set 'niter' (max minimisation iterations)."""
        return self._set_parameter("niter", value)

    def set_fcstep(self, value: int) -> GromacsCustom:
        """Set 'fcstep' (flexible constraints update interval)."""
        return self._set_parameter("fcstep", value)

    def set_nstcgsteep(self, value: int) -> GromacsCustom:
        """Set 'nstcgsteep' (SD steps before CG)."""
        return self._set_parameter("nstcgsteep", value)

    def set_nbfgscorr(self, value: int) -> GromacsCustom:
        """Set 'nbfgscorr' (L-BFGS history size)."""
        return self._set_parameter("nbfgscorr", value)

    def set_rtpi(self, value: float) -> GromacsCustom:
        """Set 'rtpi' (reference temperature for special integrators)."""
        return self._set_parameter("rtpi", value)

    # =========================================================================
    # Output
    # =========================================================================

    def set_nstxout(self, value: int) -> GromacsCustom:
        """Set 'nstxout' (coordinate output frequency)."""
        return self._set_parameter("nstxout", value)

    def set_nstvout(self, value: int) -> GromacsCustom:
        """Set 'nstvout' (velocity output frequency)."""
        return self._set_parameter("nstvout", value)

    def set_nstfout(self, value: int) -> GromacsCustom:
        """Set 'nstfout' (force output frequency)."""
        return self._set_parameter("nstfout", value)

    def set_nstlog(self, value: int) -> GromacsCustom:
        """Set 'nstlog' (log output frequency)."""
        return self._set_parameter("nstlog", value)

    def set_nstcalcenergy(self, value: int) -> GromacsCustom:
        """Set 'nstcalcenergy' (energy calculation frequency)."""
        return self._set_parameter("nstcalcenergy", value)

    def set_nstenergy(self, value: int) -> GromacsCustom:
        """Set 'nstenergy' (energy write frequency)."""
        return self._set_parameter("nstenergy", value)

    def set_nstxout_compressed(self, value: int) -> GromacsCustom:
        """Set 'nstxout-compressed' (compressed trajectory output frequency)."""
        return self._set_parameter("nstxout-compressed", value)

    def set_compressed_x_precision(self, value: int) -> GromacsCustom:
        """Set 'compressed-x-precision' (XTC precision setting)."""
        return self._set_parameter("compressed-x-precision", value)

    # =========================================================================
    # PBC + neighbor list
    # =========================================================================

    def set_cutoff_scheme(self, value: str) -> GromacsCustom:
        """Set 'cutoff-scheme' (neighbor list scheme)."""
        allowed = _enum_values(NghCutoffScheme)
        return self._set_parameter("cutoff-scheme", value, allowed)

    def set_nstlist(self, value: int) -> GromacsCustom:
        """Set 'nstlist' (neighbor list update frequency)."""
        return self._set_parameter("nstlist", value)

    def set_pbc(self, value: str) -> GromacsCustom:
        """Set 'pbc' (periodic boundary conditions)."""
        allowed = {"xyz", "no", "xy", "screw"}
        return self._set_parameter("pbc", value, allowed)

    def set_periodic_molecules(self, *, value: bool) -> GromacsCustom:
        """Set 'periodic-molecules' (treat molecules as periodic)."""
        return self._set_parameter("periodic-molecules", value)

    def set_verlet_buffer_tolerance(self, value: float) -> GromacsCustom:
        """Set 'verlet-buffer-tolerance' (target energy drift for buffer sizing)."""
        return self._set_parameter("verlet-buffer-tolerance", value)

    def set_verlet_buffer_pressure_tolerance(self, value: float) -> GromacsCustom:
        """Set 'verlet-buffer-pressure-tolerance' (target pressure error for buffer sizing)."""
        return self._set_parameter("verlet-buffer-pressure-tolerance", value)

    def set_rlist(self, value: float) -> GromacsCustom:
        """Set 'rlist' (neighbor list cutoff, nm)."""
        return self._set_parameter("rlist", value)

    # =========================================================================
    # Short-range Coulomb
    # =========================================================================

    def set_coulomb_modifier(self, value: str) -> GromacsCustom:
        """Set 'coulomb-modifier' (Coulomb potential modifier near cutoff)."""
        allowed = _enum_values(CoulombModifier)
        return self._set_parameter("coulomb-modifier", value, allowed)

    def set_rcoulomb_switch(self, value: float) -> GromacsCustom:
        """Set 'rcoulomb-switch' (Coulomb switching radius, nm)."""
        return self._set_parameter("rcoulomb-switch", value)

    def set_rcoulomb(self, value: float) -> GromacsCustom:
        """Set 'rcoulomb' (Coulomb cutoff radius, nm)."""
        return self._set_parameter("rcoulomb", value)

    def set_epsilon_r(self, value: float) -> GromacsCustom:
        """Set 'epsilon-r' (relative dielectric constant)."""
        return self._set_parameter("epsilon-r", value)

    def set_epsilon_rf(self, value: str | float) -> GromacsCustom:
        """Set 'epsilon-rf' (reaction-field dielectric constant)."""
        return self._set_parameter("epsilon-rf", value)

    def set_table_extension(self, value: float) -> GromacsCustom:
        """Set 'table-extension' (extend table range beyond cutoffs, nm)."""
        return self._set_parameter("table-extension", value)

    # =========================================================================
    # Short-range VdW
    # =========================================================================

    def set_vdw_type(self, value: str) -> GromacsCustom:
        """Set 'vdw-type' (van der Waals interaction type)."""
        allowed = _enum_values(VDWType)
        return self._set_parameter("vdw-type", value, allowed)

    def set_vdw_modifier(self, value: str) -> GromacsCustom:
        """Set 'vdw-modifier' (LJ modifier near cutoff)."""
        allowed = _enum_values(VDWModifier)
        return self._set_parameter("vdw-modifier", value, allowed)

    def set_rvdw_switch(self, value: float) -> GromacsCustom:
        """Set 'rvdw-switch' (LJ switching radius, nm)."""
        return self._set_parameter("rvdw-switch", value)

    def set_rvdw(self, value: float) -> GromacsCustom:
        """Set 'rvdw' (LJ cutoff radius, nm)."""
        return self._set_parameter("rvdw", value)

    def set_dispcorr(self, value: str) -> GromacsCustom:
        """Set 'dispcorr' (long-range dispersion correction mode)."""
        allowed = _enum_values(DispCorr)
        return self._set_parameter("dispcorr", value, allowed)

    # =========================================================================
    # Long-range electrostatics / PME
    # =========================================================================

    def set_coulombtype(self, value: str) -> GromacsCustom:
        """Set 'coulombtype' (electrostatics method)."""
        allowed = _enum_values(CoulombType)
        return self._set_parameter("coulombtype", value, allowed)

    def set_fourierspacing(self, value: float) -> GromacsCustom:
        """Set 'fourierspacing' (PME grid spacing, nm)."""
        return self._set_parameter("fourierspacing", value)

    def set_fourier_nx(self, value: int) -> GromacsCustom:
        """Set 'fourier-nx' (PME grid size X)."""
        return self._set_parameter("fourier-nx", value)

    def set_fourier_ny(self, value: int) -> GromacsCustom:
        """Set 'fourier-ny' (PME grid size Y)."""
        return self._set_parameter("fourier-ny", value)

    def set_fourier_nz(self, value: int) -> GromacsCustom:
        """Set 'fourier-nz' (PME grid size Z)."""
        return self._set_parameter("fourier-nz", value)

    def set_pme_order(self, value: int) -> GromacsCustom:
        """Set 'pme-order' (PME interpolation order)."""
        return self._set_parameter("pme-order", value)

    # =========================================================================
    # Ewald / LJ-PME
    # =========================================================================

    def set_ewald_rtol(self, value: float) -> GromacsCustom:
        """Set 'ewald-rtol' (PME/Ewald relative tolerance)."""
        return self._set_parameter("ewald-rtol", value)

    def set_ewald_geometry(self, value: str) -> GromacsCustom:
        """Set 'ewald-geometry' (Ewald boundary geometry)."""
        return self._set_parameter("ewald-geometry", value)

    def set_epsilon_surface(self, value: float) -> GromacsCustom:
        """Set 'epsilon-surface' (surface dielectric for non-3D Ewald)."""
        return self._set_parameter("epsilon-surface", value)

    def set_ewald_rtol_lj(self, value: float) -> GromacsCustom:
        """Set 'ewald-rtol-lj' (LJ-PME relative tolerance)."""
        return self._set_parameter("ewald-rtol-lj", value)

    def set_lj_pme_comb_rule(self, value: str) -> GromacsCustom:
        """Set 'lj-pme-comb-rule' (combination rule for LJ-PME)."""
        return self._set_parameter("lj-pme-comb-rule", value)

    # =========================================================================
    # Pressure control
    # =========================================================================

    def set_pcoupl(self, value: str) -> GromacsCustom:
        """Set 'pcoupl' (pressure coupling algorithm)."""
        allowed = _enum_values(Barostat)
        return self._set_parameter("pcoupl", value, allowed)

    def set_refcoord_scaling(self, value: str) -> GromacsCustom:
        """Set 'refcoord-scaling' (reference coordinate scaling under P-coupling)."""
        return self._set_parameter("refcoord-scaling", value)

    # =========================================================================
    # Constraints
    # =========================================================================

    def set_constraint_algorithm(self, value: str) -> GromacsCustom:
        """Set 'constraint-algorithm' (constraints solver)."""
        allowed = _enum_values(ConstraintsAlgorithms)
        return self._set_parameter("constraint-algorithm", value, allowed)

    def set_continuation(self, *, value: bool) -> GromacsCustom:
        """Set 'continuation' (continue run from previous state)."""
        return self._set_parameter("continuation", value)

    def set_shake_sor(self, value: str) -> GromacsCustom:
        """Set 'shake-sor' (SHAKE SOR settings)."""
        return self._set_parameter("shake-sor", value)

    def set_shake_tol(self, value: float) -> GromacsCustom:
        """Set 'shake-tol' (SHAKE tolerance)."""
        return self._set_parameter("shake-tol", value)

    def set_lincs_order(self, value: int) -> GromacsCustom:
        """Set 'lincs-order' (LINCS expansion order)."""
        return self._set_parameter("lincs-order", value)

    def set_lincs_warnangle(self, value: float) -> GromacsCustom:
        """Set 'lincs-warnangle' (LINCS warning angle, degrees)."""
        return self._set_parameter("lincs-warnangle", value)

    # =========================================================================
    # Walls
    # =========================================================================

    def set_nwall(self, value: int) -> GromacsCustom:
        """Set 'nwall' (number of walls)."""
        return self._set_parameter("nwall", value)

    def set_wall_type(self, value: str) -> GromacsCustom:
        """Set 'wall-type' (wall potential type)."""
        return self._set_parameter("wall-type", value)

    def set_wall_r_linpot(self, value: float) -> GromacsCustom:
        """Set 'wall-r-linpot' (linear-potential wall parameter)."""
        return self._set_parameter("wall-r-linpot", value)

    def set_wall_ewald_zfac(self, value: float) -> GromacsCustom:
        """Set 'wall-ewald-zfac' (Ewald z scaling factor for walls)."""
        return self._set_parameter("wall-ewald-zfac", value)

    # =========================================================================
    # Biasing / restraints
    # =========================================================================

    def set_qmmm(self, *, value: bool) -> GromacsCustom:
        """Set 'qmmm' (enable QM/MM coupling)."""
        return self._set_parameter("qmmm", value)

    def set_pull(self, *, value: bool) -> GromacsCustom:
        """Set 'pull' (enable pulling code)."""
        return self._set_parameter("pull", value)

    def set_awh(self, *, value: bool) -> GromacsCustom:
        """Set 'awh' (enable AWH method)."""
        return self._set_parameter("awh", value)

    def set_rotation(self, *, value: bool) -> GromacsCustom:
        """Set 'rotation' (enable enforced rotation)."""
        return self._set_parameter("rotation", value)


# ============================================================================
# Run helper
# ============================================================================


def run_gro_custom(
    parameters: str | Path,
    system: BSS._SireWrappers.System,
    changes: Mapping[str, Any] | None = None,
    changes_file: str | Path | None = None,
) -> tuple[BSS._SireWrappers.System, BSS.Protocol]:
    """Create protocol, apply overrides (file + mapping), run GROMACS, return (system, protocol)."""
    custom_protocol = GromacsCustom(parameters)

    lines = custom_protocol.getConfig()

    parsed_from_file: dict[str, Any] | None = None
    if changes_file is not None:
        parsed_from_file = apply_changes(lines, changes_file)

    if changes is not None:
        change_default_params(lines, changes)

    custom_protocol.setConfig(lines)

    if parsed_from_file:
        logger.info(
            "Applied %d mdp overrides from file: %s",
            len(parsed_from_file),
            changes_file,
        )
    if changes:
        logger.info("Applied %d mdp overrides from mapping.", len(changes))

    logger.info("Starting GROMACS process.")
    process = BSS.Process.Gromacs(system, protocol=custom_protocol)
    process.start()
    process.wait()
    logger.info("Process finished.")

    customized = process.getSystem(block=True)
    return customized, custom_protocol
