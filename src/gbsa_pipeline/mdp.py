"""GROMACS MDP parameter model, serialisation helpers, and BSS protocol integration.

Public surface
--------------
``GromacsParams``
    Frozen Pydantic model covering the ~80 MDP parameters used by this pipeline.
    Keys are accepted in both Python (underscore) and GROMACS (hyphen) style;
    serialised output always uses hyphenated GROMACS keys.

``GromacsCustom``
    Thin subclass of ``BSS.Protocol.Custom`` that accepts a ``GromacsParams``
    instead of a raw MDP file path.

``run_gro_custom``
    One-shot helper: validate params, create a BSS custom protocol, run GROMACS,
    and return the result.

All enums (``Thermostat``, ``Barostat``, ``Integrator``, …) are re-exported from
this module so callers only need a single import::

    from gbsa_pipeline.mdp import GromacsParams, Thermostat, Barostat
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from enum import Enum
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import BioSimSpace as BSS
from pydantic import AliasGenerator, BaseModel, ConfigDict, field_validator

from gbsa_pipeline.mdp_enum import (
    Barostat,
    CommMode,
    Constraints,
    ConstraintsAlgorithms,
    CoulombModifier,
    CoulombType,
    DispCorr,
    Integrator,
    LJPMECombination,
    NghCutoffScheme,
    PCoupleType,
    Thermostat,
    VDWModifier,
    VDWType,
    VelocityGeneration,
)

# Re-export enums so consumers only need ``from gbsa_pipeline.mdp import …``.
__all__ = [
    "Barostat",
    "CommMode",
    "Constraints",
    "ConstraintsAlgorithms",
    "CoulombModifier",
    "CoulombType",
    "DispCorr",
    "GromacsCustom",
    "GromacsParams",
    "Integrator",
    "LJPMECombination",
    "NghCutoffScheme",
    "PCoupleType",
    "Thermostat",
    "VDWModifier",
    "VDWType",
    "VelocityGeneration",
    "field_to_mdp_key",
    "run_gro_custom",
    "set_mdp_key",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key conversion helpers
# ---------------------------------------------------------------------------


def _to_gmx_key(field_name: str) -> str:
    """Return the GROMACS MDP key for a Python field name (underscore → hyphen)."""
    return field_name.replace("_", "-")


def field_to_mdp_key(field: str) -> str:
    """Normalise any MDP key variant to a lowercase hyphen-separated string.

    Used for comparison and deduplication when merging BSS-generated configs
    with local overrides — ``DispCorr``, ``dispcorr``, and ``disp_corr`` all
    normalise to ``dispcorr``.
    """
    return field.strip().lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def format_gmx_value(value: Any) -> str:
    """Render a Python value as a GROMACS MDP token."""
    if hasattr(value, "value"):  # Enum / StrEnum — unwrap to plain string
        value = value.value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    raise TypeError(f"Unsupported .mdp value type: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Raw MDP line helpers  (used by md.py to patch BSS-generated configs)
# ---------------------------------------------------------------------------

# Matches a GROMACS MDP assignment line, capturing four groups:
#   1. leading whitespace   2. key (word chars + hyphens)
#   3. = with surrounding whitespace   4. optional inline comment
_MDP_ASSIGNMENT = re.compile(r"^(\s*)([\w-]+)(\s*=\s*)[^;#]*\s*([;#].*)?$")


def set_mdp_key(lines: list[str], key: str, value: Any) -> list[str]:
    """Return a copy of *lines* with ``key = value`` updated or appended.

    The first matching assignment is replaced, preserving its original
    indentation, ``=``-spacing, and inline comment.  When the key is absent
    it is appended in aligned ``key = value`` format.
    """
    mdp_value = format_gmx_value(value)
    wanted = key.strip()

    for i, ln in enumerate(lines):
        m = _MDP_ASSIGNMENT.match(ln)
        if m is None or m.group(2) != wanted:
            continue
        indent, _, eq_ws, comment = m.groups()
        suffix = f"  {comment}" if comment else ""
        return [*lines[:i], f"{indent}{wanted}{eq_ws}{mdp_value}{suffix}", *lines[i + 1 :]]

    return [*lines, f"{wanted:<28} = {mdp_value}"]


# ---------------------------------------------------------------------------
# GromacsParams
# ---------------------------------------------------------------------------


class GromacsParams(BaseModel):
    """Validated GROMACS MDP parameters with automatic key conversion.

    Fields use Python naming conventions (underscores, lowercase). Both Python
    and GROMACS hyphenated forms are accepted on construction — no separate
    ``from_mapping`` call is needed::

        # All three are equivalent:
        GromacsParams(nsteps=1000, cutoff_scheme=NghCutoffScheme.VERLET)
        GromacsParams(**{"nsteps": 1000, "cutoff-scheme": "Verlet"})
        GromacsParams.model_validate({"nsteps": 1000, "cutoff-scheme": "Verlet"})

    Since the model is frozen, use :meth:`override` to derive a modified copy::

        production = heating_params.override(nsteps=500_000, tcoupl=Thermostat.NO)
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
        populate_by_name=True,
        alias_generator=AliasGenerator(
            validation_alias=_to_gmx_key,
            serialization_alias=_to_gmx_key,
        ),
    )

    # Integrator
    integrator: Integrator = Integrator.LEAP_FROG
    tinit: float = 0.0
    dt: float = 0.001
    nsteps: int = 500
    init_step: int = 0
    simulation_part: int = 1
    mts: bool = True
    mass_repartition_factor: float = 1.0
    comm_mode: CommMode = CommMode.LINEAR
    nstcomm: int = 100
    bd_fric: float = 0.0
    ld_seed: int = 1845489648

    # Energy minimisation
    emtol: float = 10.0
    emstep: float = 0.01
    niter: int = 20
    fcstep: int = 0
    nstcgsteep: int = 1000
    nbfgscorr: int = 10
    rtpi: float = 0.05

    # Output
    nstxout: int = 0
    nstvout: int = 0
    nstfout: int = 0
    nstlog: int = 500
    nstcalcenergy: int = 100
    nstenergy: int = 500
    nstxout_compressed: int = 500
    compressed_x_precision: int = 1000

    # Neighbour searching
    cutoff_scheme: NghCutoffScheme = NghCutoffScheme.VERLET
    nstlist: int = 20
    pbc: str = "xyz"
    periodic_molecules: bool = False
    verlet_buffer_tolerance: float = 0.005
    verlet_buffer_pressure_tolerance: float = 0.5
    rlist: float = 1.221

    # Electrostatics
    coulombtype: CoulombType = CoulombType.PME
    coulomb_modifier: CoulombModifier = CoulombModifier.POTENTIAL_SHIFT
    rcoulomb_switch: float = 0.0
    rcoulomb: float = 1.2
    epsilon_r: float = 1.0
    epsilon_rf: float | str = "inf"
    table_extension: float = 1.0
    fourierspacing: float = 0.16
    fourier_nx: int = 52
    fourier_ny: int = 52
    fourier_nz: int = 52
    pme_order: int = 4
    ewald_rtol: float = 1e-5
    ewald_geometry: str = "3d"
    epsilon_surface: float = 0.0
    ewald_rtol_lj: float = 0.001
    lj_pme_comb_rule: LJPMECombination = LJPMECombination.GEOMETRIC

    # Van der Waals ---
    vdwtype: VDWType = VDWType.CUT_OFF
    vdw_modifier: VDWModifier = VDWModifier.FORCE_SWITCH
    rvdw_switch: float = 1.0
    rvdw: float = 1.2
    dispcorr: DispCorr = DispCorr.NO

    # Barostat and pressure coupling
    pcoupl: Barostat = Barostat.NO
    pcoupltype: PCoupleType = PCoupleType.ISOTROPIC
    tau_p: float = 2.0
    ref_p: float = 1.0
    compressibility: float = 4.5e-5
    refcoord_scaling: str = "No"

    # Thermostat
    tcoupl: Thermostat = Thermostat.NO
    tc_grps: str = "System"
    tau_t: float = 0.1
    ref_t: float = 300.0
    nhchainlength: int = 10

    # Annealing
    annealing: str | None = None
    annealing_npoints: int | None = None
    annealing_time: str | None = None
    annealing_temp: str | None = None

    # Position restraints and preprocessing
    define: str | None = None

    # Velocity generation
    gen_vel: VelocityGeneration = VelocityGeneration.NO
    gen_temp: float = 300.0
    gen_seed: int = -1

    # Bond constraints
    constraints: Constraints = Constraints.NONE
    constraint_algorithm: ConstraintsAlgorithms = ConstraintsAlgorithms.LINCS
    continuation: bool | str = False
    shake_sor: str = "no"
    shake_tol: float = 0.0001
    lincs_order: int = 4
    lincs_warnangle: float = 30
    nwall: int = 0
    wall_type: str = "9-3"
    wall_r_linpot: float = -1
    wall_ewald_zfac: float = 3
    qmmm: bool = False
    pull: bool = False
    awh: bool = False
    rotation: bool = False

    @field_validator("tcoupl", mode="before")
    @classmethod
    def _normalise_tcoupl(cls, value: Thermostat | str) -> Thermostat | str:
        """Accept mixed-case thermostat strings such as ``V-rescale``."""
        if isinstance(value, Thermostat):
            return value
        return str(value).lower().strip()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> GromacsParams:
        """Construct from a mapping of GROMACS-style or Python-style keys.

        Kept for backward compatibility; equivalent to
        ``GromacsParams.model_validate(mapping)``.
        """
        return cls.model_validate(mapping)

    def to_mapping(self) -> dict[str, Any]:
        """Return a ``{gromacs-key: value}`` dict suitable for MDP rendering.

        Keys use hyphenated GROMACS notation.  ``None`` fields are omitted so
        optional parameters are only written when explicitly set.  Enum values
        are serialised to their string ``.value``.
        """
        return {
            k: (v.value if isinstance(v, Enum) else v)
            for k, v in self.model_dump(by_alias=True).items()
            if v is not None
        }

    def override(self, **kwargs: Any) -> GromacsParams:
        """Return a new instance with the given fields replaced.

        Keyword arguments must use Python-style names (underscores). To apply
        a dict of GROMACS-style hyphenated keys use
        ``GromacsParams.model_validate({**self.to_mapping(), **overrides})``.

        Example::

            params = base.override(nsteps=500_000, tcoupl=Thermostat.NO)
        """
        return self.model_copy(update=kwargs)

    def to_mdp_lines(self) -> list[str]:
        """Render as a list of ``key = value`` MDP lines (no trailing newlines)."""
        return [f"{k} = {format_gmx_value(v)}" for k, v in self.to_mapping().items()]

    def to_mdp(self) -> str:
        """Render as newline-terminated MDP text ready to write to a file."""
        return "\n".join(self.to_mdp_lines()) + "\n"


# ---------------------------------------------------------------------------
# BioSimSpace bridge
# ---------------------------------------------------------------------------


class GromacsCustom(BSS.Protocol.Custom):
    """``BSS.Protocol.Custom`` constructed from a :class:`GromacsParams` instance.

    BioSimSpace requires a file path for custom protocols; this class writes a
    temporary MDP file transparently so callers never touch the filesystem.
    The resolved parameter mapping is stored on ``_parameters`` for inspection
    and testing.
    """

    def __init__(self, params: GromacsParams | Mapping[str, Any] | None = None) -> None:
        """Write a temporary MDP file from *params* and initialise the protocol."""
        self.params = (GromacsParams.from_mapping(params) if isinstance(params, Mapping) else params) or GromacsParams()

        with NamedTemporaryFile("w", suffix=".mdp", delete=False, encoding="utf-8") as tmp:
            tmp.write(self.params.to_mdp())
            mdp_path = tmp.name

        super().__init__(mdp_path)
        self._parameters = self.params.to_mapping()


def run_gro_custom(
    params: Mapping[str, Any] | GromacsParams | None,
    system: BSS._SireWrappers.System,
    work_dir: Path | None = None,
) -> tuple[BSS._SireWrappers.System, BSS.Protocol]:
    """Run a GROMACS custom protocol and return ``(system, protocol)``.

    *params* is validated through :class:`GromacsParams` before BioSimSpace is
    started, so unsupported keys and invalid enum values fail early.  Pass
    ``None`` to run with default parameters.

    Example::

        result, _ = run_gro_custom(
            GromacsParams(nsteps=50_000, tcoupl=Thermostat.VRESCALE),
            solvated_system,
            work_dir=Path("run/nvt"),
        )
    """
    gromacs_params = (GromacsParams.from_mapping(params) if isinstance(params, Mapping) else params) or GromacsParams()

    protocol = GromacsCustom(params=gromacs_params)

    logger.info(
        "Starting GROMACS custom protocol (%d parameters).",
        len(gromacs_params.to_mapping()),
    )
    kwargs: dict[str, Any] = {"work_dir": str(work_dir)} if work_dir else {}
    process = BSS.Process.Gromacs(system, protocol=protocol, **kwargs)
    process.start()
    process.wait()
    logger.info("GROMACS process finished.")

    return process.getSystem(block=True), protocol
