"""Unit tests for the explicit stage builders in :mod:`gbsa_pipeline.protocol`.

Each test exercises one builder and asserts on the structural contract
documented in the module docstring: AMBER nonbonded block applied to
every stage, PME mesh cleared, membrane variants emit semiisotropic
C-rescale, position-restraint flags toggled through ``define``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gbsa_pipeline import protocol
from gbsa_pipeline.mdp import SemiisotropicValue
from gbsa_pipeline.mdp_enum import (
    Barostat,
    Constraints,
    CoulombModifier,
    CoulombType,
    DispCorr,
    Integrator,
    PCoupleType,
    Thermostat,
    VDWModifier,
    VDWType,
    VelocityGeneration,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gbsa_pipeline.mdp import GromacsParams

# ---------------------------------------------------------------------------
# AMBER nonbonded block applied by every builder
# ---------------------------------------------------------------------------

_ALL_BUILDERS = [
    ("steep", lambda: protocol.steep(nsteps=100)),
    ("cg", lambda: protocol.cg(nsteps=100)),
    ("relax", lambda: protocol.relax(nsteps=100)),
    ("heating", lambda: protocol.heating(nsteps=100)),
    ("npt_restrained", lambda: protocol.npt_restrained(nsteps=100)),
    ("npt", lambda: protocol.npt(nsteps=100)),
    ("prod", lambda: protocol.production(nsteps=100)),
]


@pytest.mark.parametrize(("name", "builder"), _ALL_BUILDERS)
def test_amber_nonbonded_block_everywhere(name: str, builder: Callable[[], GromacsParams]) -> None:
    """Every stage inherits the AMBER rcoulomb=rvdw=1.0 / Potential-shift / mts=no block."""
    p = builder()
    assert p.rcoulomb == pytest.approx(1.0), name
    assert p.rvdw == pytest.approx(1.0), name
    assert p.coulombtype is CoulombType.PME, name
    assert p.coulomb_modifier is CoulombModifier.POTENTIAL_SHIFT, name
    assert p.vdwtype is VDWType.CUT_OFF, name
    assert p.vdw_modifier is VDWModifier.POTENTIAL_SHIFT, name
    assert p.dispcorr is DispCorr.ENERGY_PRESSURE, name
    assert p.fourierspacing == pytest.approx(0.12), name
    assert p.mts is False, name


@pytest.mark.parametrize(("name", "builder"), _ALL_BUILDERS)
def test_pme_mesh_cleared_so_grompp_picks_from_fourierspacing(
    name: str,
    builder: Callable[[], GromacsParams],
) -> None:
    """fourier_nx/ny/nz = 0 lets grompp auto-pick mesh from fourierspacing."""
    p = builder()
    assert p.fourier_nx == 0, name
    assert p.fourier_ny == 0, name
    assert p.fourier_nz == 0, name


# ---------------------------------------------------------------------------
# Per-builder specifics
# ---------------------------------------------------------------------------


def test_steep_uses_steep_integrator_and_default_emtol() -> None:
    p = protocol.steep()
    assert p.integrator is Integrator.STEEP
    assert p.emtol == pytest.approx(1000.0)
    assert p.constraints is Constraints.HYDROGENS_BONDS
    assert p.nstxout_compressed == 0  # no trajectory for minimisation


def test_cg_disables_constraints_to_prevent_lincs_crash() -> None:
    """CG line search is unstable with LINCS → protocol disables constraints."""
    p = protocol.cg()
    assert p.integrator is Integrator.CONJUGATE_GRADIENT
    assert p.constraints is Constraints.NONE


def test_relax_uses_sd_integrator_and_posres_heavy_define() -> None:
    """Relax = short SD with POSRES_HEAVY to settle solvent clashes."""
    p = protocol.relax()
    assert p.integrator is Integrator.LEAP_FROG_STOCHASTIC
    assert "POSRES_HEAVY" in (p.define or "")
    assert p.gen_vel is VelocityGeneration.YES  # first stage after minimisation


def test_heating_emits_annealing_schedule_and_posres_bb() -> None:
    p = protocol.heating(nsteps=50_000, temperature_start=50.0, temperature_end=300.0)
    assert p.tcoupl is Thermostat.VRESCALE
    assert p.annealing == "single"
    assert p.annealing_npoints == 2
    assert p.annealing_temp == "50.0 300.0"
    assert "POSRES_BB" in (p.define or "")
    assert "POSRES_LIPID" not in (p.define or "")


def test_heating_restrain_lipid_adds_posres_lipid_flag() -> None:
    p = protocol.heating(restrain_lipid=True)
    assert "POSRES_BB" in (p.define or "")
    assert "POSRES_LIPID" in (p.define or "")


def test_npt_restrained_uses_crescale_and_refcoord_scaling_com() -> None:
    """refcoord-scaling=com is required with restraints + any barostat."""
    p = protocol.npt_restrained()
    assert p.pcoupl is Barostat.CRESCALE
    assert p.pcoupltype is PCoupleType.ISOTROPIC
    assert p.refcoord_scaling == "com"
    assert "POSRES_BB" in (p.define or "")
    assert p.continuation is True
    assert p.gen_vel is VelocityGeneration.NO


def test_npt_is_unrestrained_and_crescale() -> None:
    p = protocol.npt()
    assert p.pcoupl is Barostat.CRESCALE
    assert p.define is None
    assert p.continuation is True


def test_prod_requires_nsteps_and_defaults_to_isotropic_crescale() -> None:
    """prod() has no nsteps default — production length is always explicit."""
    p = protocol.production(nsteps=250_000)
    assert p.integrator is Integrator.LEAP_FROG
    assert p.nsteps == 250_000
    assert p.pcoupl is Barostat.CRESCALE
    assert p.pcoupltype is PCoupleType.ISOTROPIC


def test_prod_without_nsteps_raises_typeerror() -> None:
    """Nsteps is a required positional kwarg so callers never forget it."""
    with pytest.raises(TypeError):
        protocol.production()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Membrane variant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "builder"),
    [
        ("npt_restrained", lambda: protocol.npt_restrained(membrane=True)),
        ("npt", lambda: protocol.npt(membrane=True)),
        ("prod", lambda: protocol.production(nsteps=1_000, membrane=True)),
    ],
)
def test_membrane_variant_uses_semiisotropic_crescale(name: str, builder: Callable[[], GromacsParams]) -> None:
    """membrane=True → pcoupltype=semiisotropic + SemiisotropicValue ref_p/compressibility."""
    p = builder()
    assert p.pcoupl is Barostat.CRESCALE, name
    assert p.pcoupltype is PCoupleType.SEMIISOTROPIC, name
    assert isinstance(p.ref_p, SemiisotropicValue), name
    assert p.ref_p.lateral == pytest.approx(1.0), name
    assert p.ref_p.normal == pytest.approx(1.0), name
    assert isinstance(p.compressibility, SemiisotropicValue), name


def test_isotropic_variant_keeps_scalar_ref_p() -> None:
    """Default (membrane=False) uses a plain float for ref_p/compressibility."""
    p = protocol.npt()
    assert isinstance(p.ref_p, float)
    assert isinstance(p.compressibility, float)


# ---------------------------------------------------------------------------
# Rendered MDP sanity
# ---------------------------------------------------------------------------


def test_rendered_mdp_contains_core_amber_keys() -> None:
    """to_mdp() output has every AMBER-block key grompp expects, hyphenated."""
    text = protocol.npt(nsteps=1000).to_mdp()
    assert "rcoulomb = 1.0" in text
    assert "rvdw = 1.0" in text
    assert "coulombtype = PME" in text
    assert "dispcorr = EnerPres" in text
    assert "fourierspacing = 0.12" in text
    assert "pcoupl = C-rescale" in text


def test_prod_membrane_mdp_renders_two_values_for_ref_p() -> None:
    """Semiisotropic ref_p becomes two space-separated numbers in the MDP."""
    text = protocol.production(nsteps=1000, membrane=True).to_mdp()
    assert "ref-p = 1.0 1.0" in text
    assert "compressibility = 4.5e-05 4.5e-05" in text
    assert "pcoupltype = semiisotropic" in text
