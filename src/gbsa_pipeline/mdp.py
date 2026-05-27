"""Creating custom protocol with one setter per GROMACS parameter."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import Enum
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import BioSimSpace as BSS
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

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

logger = logging.getLogger(__name__)


class GromacsParams(BaseModel):
    """MDP parameters with validated defaults and serialization helpers.

    The model is intentionally close to GROMACS MDP terminology because the
    rendered output is passed directly to ``BSS.Protocol.Custom``. Most fields
    use enums imported from ``change_defaults_enum`` so invalid common options
    are rejected before a GROMACS process is started. The ``Integrator`` enum
    includes both MD propagation integrators and minimization integrators such
    as ``steep`` and ``cg``. Optional fields such as ``define`` and annealing
    settings are emitted only when the caller supplies them, so the default MDP
    remains compact.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
    )

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
    emtol: float = 10.0
    emstep: float = 0.01
    niter: int = 20
    fcstep: int = 0
    nstcgsteep: int = 1000
    nbfgscorr: int = 10
    rtpi: float = 0.05
    nstxout: int = 0
    nstvout: int = 0
    nstfout: int = 0
    nstlog: int = 500
    nstcalcenergy: int = 100
    nstenergy: int = 500
    nstxout_compressed: int = 500
    compressed_x_precision: int = 1000
    cutoff_scheme: NghCutoffScheme = NghCutoffScheme.VERLET
    nstlist: int = 20
    pbc: str = "xyz"
    periodic_molecules: bool = False
    verlet_buffer_tolerance: float = 0.005
    verlet_buffer_pressure_tolerance: float = 0.5
    rlist: float = 1.221
    coulomb_modifier: CoulombModifier = CoulombModifier.POTENTIAL_SHIFT
    rcoulomb_switch: float = 0.0
    rcoulomb: float = 1.2
    epsilon_r: float = 1.0
    epsilon_rf: float | str = "inf"
    table_extension: float = 1.0
    vdwtype: VDWType = VDWType.CUT_OFF
    vdw_modifier: VDWModifier = VDWModifier.FORCE_SWITCH
    rvdw_switch: float = 1.0
    rvdw: float = 1.2
    dispcorr: DispCorr = DispCorr.NO
    coulombtype: CoulombType = CoulombType.PME
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
    pcoupl: Barostat = Barostat.NO
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

    # Barostat and pressure coupling
    pcoupltype: PCoupleType = PCoupleType.ISOTROPIC
    tau_p: float = 2.0
    ref_p: float = 1.0
    compressibility: float = 4.5e-5

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

    @model_validator(mode="before")
    @classmethod
    def _normalise_input_aliases(cls, values: Any) -> Any:
        """Normalise compatibility aliases before Pydantic validates fields.

        Direct construction with ``GromacsParams(**mapping)`` should behave like
        ``from_mapping`` for known compatibility aliases. In particular, older
        code may still pass ``vdw_type`` while the correct GROMACS MDP key is
        ``vdwtype``. If both forms are supplied, the explicit ``vdwtype`` value
        wins and the alias is ignored. Non-mapping input is returned unchanged
        so Pydantic can handle it normally.
        """
        if not isinstance(values, Mapping):
            return values

        normalised = dict(values)
        for alias, field_name in _FIELD_ALIASES.items():
            if alias in normalised and field_name not in normalised:
                normalised[field_name] = normalised.pop(alias)

        return normalised

    @field_validator("tcoupl", mode="before")
    @classmethod
    def _normalise_tcoupl(cls, value: Thermostat | str) -> Thermostat | str:
        """Normalise thermostat coupling strings before enum validation.

        GROMACS examples and human-written parameter blocks often use
        ``V-rescale`` while the enum stores the canonical lower-case value
        ``v-rescale``. This validator accepts that common spelling without
        weakening the field to arbitrary strings. Existing enum instances are
        passed through unchanged. Other invalid values still fail during enum
        validation.
        """
        if isinstance(value, Thermostat):
            return value

        return str(value).lower().strip()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> GromacsParams:
        """Instantiate from a mapping using GROMACS or Python-style keys.

        Hyphenated GROMACS keys such as ``cutoff-scheme`` are converted to the
        matching Python field name. Existing Python-style keys such as
        ``cutoff_scheme`` are accepted as well. The compatibility alias
        ``vdw_type`` is mapped to ``vdwtype`` because GROMACS uses the latter
        spelling in MDP files. Unknown keys raise ``KeyError`` so typos in
        parameter blocks are caught before launching a GROMACS process.
        """
        kwargs = {}

        for key, value in mapping.items():
            field_name = mdp_key_to_field(key)
            if field_name not in cls.model_fields:
                raise KeyError(f"Unknown parameter key: {key}")
            kwargs[field_name] = value

        return cls(**kwargs)

    def to_mapping(self) -> dict[str, Any]:
        """Return a GROMACS-style mapping.

        Most Python field names are converted by replacing underscores with
        hyphens, matching GROMACS MDP syntax. The ``vdwtype`` key is already in
        GROMACS spelling and is therefore emitted unchanged. Enum values are
        serialized through their ``.value`` attribute. Fields with value
        ``None`` are skipped so optional MDP parameters are emitted only when
        explicitly requested. The returned mapping is suitable for MDP rendering
        and for applying later validated overrides.
        """
        result = {}

        for field_name, field_value in self.model_dump().items():
            if field_value is None:
                continue

            serialized = field_value.value if isinstance(field_value, Enum) else field_value
            key = field_name if field_name == "vdwtype" else field_name.replace("_", "-")
            result[key] = serialized

        return result

    def to_mdp_lines(self) -> list[str]:
        """Render parameters as MDP lines without any base file.

        The order follows the model field order so generated files are stable
        and easy to inspect in integration-test output directories. Values are
        formatted through ``format_gmx_value`` so booleans, strings, numbers,
        and enum-derived strings are written consistently. No base MDP file is
        read or merged here; callers should merge parameter changes before
        constructing this model. The returned list is newline-free to make it
        convenient for tests and protocol construction.
        """
        lines: list[str] = []
        for key, value in self.to_mapping().items():
            lines.append(f"{key} = {format_gmx_value(value)}")
        return lines

    def to_mdp(self) -> str:
        """Render parameters as newline-terminated MDP text.

        This is a convenience wrapper around ``to_mdp_lines`` for callers that
        need a complete text block. The generated text is suitable for writing
        directly to a temporary or persistent ``.mdp`` file. It intentionally
        does not include comments because this module is used as a machine
        bridge to BioSimSpace/GROMACS. The final newline matches normal text
        file conventions.
        """
        return "\n".join(self.to_mdp_lines()) + "\n"


class GromacsCustom(BSS.Protocol.Custom):
    """Thin wrapper bridging ``GromacsParams`` to ``BSS.Protocol.Custom``.

    BioSimSpace accepts a custom GROMACS protocol from an MDP file path. This
    wrapper creates that MDP file from a validated ``GromacsParams`` object or a
    mapping accepted by ``GromacsParams.from_mapping``. The temporary file is
    intentionally retained because BioSimSpace needs the path after protocol
    construction. The rendered parameter mapping is also stored on
    ``_parameters`` for compatibility with callers that inspect the protocol.
    """

    def __init__(
        self,
        params: GromacsParams | Mapping[str, Any] | None = None,
    ) -> None:
        """Create a custom protocol from params only.

        No base MDP file is read, so every rendered parameter comes from the
        validated model. Mappings are passed through ``from_mapping`` to support
        both GROMACS-style and Python-style keys. The temporary MDP file uses a
        real filesystem path because ``BSS.Protocol.Custom`` expects a file
        rather than raw text. The object keeps the final parameter mapping for
        debugging and tests.
        """
        self.params = (GromacsParams.from_mapping(params) if isinstance(params, Mapping) else params) or GromacsParams()

        lines = self.params.to_mdp_lines()

        with NamedTemporaryFile("w", suffix=".mdp", delete=False, encoding="utf-8") as tmp:
            tmp.write("\n".join(lines) + "\n")
            mdp_path = tmp.name

        super().__init__(mdp_path)

        self._parameters = self.params.to_mapping()


def run_gro_custom(
    parameters: Mapping[str, Any] | GromacsParams | None,
    system: BSS._SireWrappers.System,
    changes: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | GromacsParams | None = None,
    work_dir: Path | None = None,
) -> tuple[BSS._SireWrappers.System, BSS.Protocol]:
    """Create a custom GROMACS protocol, run GROMACS, and return the result.

    ``parameters`` provides the base parameter set, while ``changes`` and
    ``params`` can apply later overrides. The final merged mapping is validated
    through ``GromacsParams`` before BioSimSpace is started, so unsupported keys
    and invalid enum values fail early. ``work_dir`` is passed directly to
    ``BSS.Process.Gromacs`` when supplied, which keeps integration-test artefacts
    in a predictable directory. The returned tuple contains the resulting system
    and the protocol object used to launch the process.
    """
    base_params = (
        GromacsParams.from_mapping(parameters) if isinstance(parameters, Mapping) else parameters
    ) or GromacsParams()

    merged = base_params.to_mapping()

    if changes is not None:
        merged.update(changes)

    if params is not None:
        merged.update(params if isinstance(params, Mapping) else params.to_mapping())

    final_params = GromacsParams.from_mapping(merged)

    custom_protocol = GromacsCustom(params=final_params)

    if changes:
        logger.info("Applied %d mdp overrides from mapping.", len(changes))

    logger.info("Starting GROMACS process.")
    kwargs = {"work_dir": str(work_dir)} if work_dir else {}
    process = BSS.Process.Gromacs(system, protocol=custom_protocol, **kwargs)
    process.start()
    process.wait()
    logger.info("Process finished.")

    customized = process.getSystem(block=True)
    return customized, custom_protocol


# Aliases for GROMACS parameter keys whose canonical MDP spelling differs from
# the Python field name used in GromacsParams.  Only a small number of
# historically inconsistent names are listed here.
_FIELD_ALIASES: dict[str, str] = {
    "vdw_type": "vdwtype",
}


def mdp_key_to_field(key: str) -> str:
    """Convert a GROMACS MDP key to a Python model field name.

    GROMACS uses hyphens (e.g. ``cutoff-scheme``); Python fields use
    underscores.  A small alias table handles the rare keys whose canonical
    GROMACS spelling differs from the field name used in ``GromacsParams``
    (e.g. ``vdwtype`` → ``vdw_type`` field).

    Used by ``GromacsParams.from_mapping`` so callers can pass both
    GROMACS-style and Python-style keys without silent failures.
    """
    field_name = key.replace("-", "_")
    return _FIELD_ALIASES.get(field_name, field_name)


def field_to_mdp_key(field: str) -> str:
    """Convert a Python field name to a comparison-safe GROMACS MDP key.

    Returns a lowercase, hyphen-separated key for comparison and deduplication.
    This is used when merging BSS-generated MDP configs with local overrides
    where the same logical parameter may appear with different spelling
    variants (e.g. ``DispCorr``, ``dispcorr``, ``disp_corr``).
    """
    return field.strip().lower().replace("_", "-")


def _leading_ws(s: str) -> str:
    """Return the leading whitespace of `s`."""
    i = 0
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    return s[:i]


def _split_inline_comment(s: str) -> tuple[str, str]:
    """Split `s` into (before_comment, comment) for inline ';' or '#'.

    Returns the earliest comment delimiter occurrence. If no delimiter exists,
    returns (s, "").
    """
    semi = s.find(";")
    hash_ = s.find("#")

    if semi == -1 and hash_ == -1:
        return s, ""
    if semi == -1:
        idx = hash_
    elif hash_ == -1:
        idx = semi
    else:
        idx = min(semi, hash_)

    return s[:idx], s[idx:]


def format_gmx_value(value: Any) -> str:
    """Convert Python values to GROMACS .mdp-compatible strings."""
    if hasattr(value, "value"):  # Enum / StrEnum
        value = value.value

    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()

    raise TypeError(f"Unsupported .mdp value type: {type(value).__name__}")


def is_comment(line: str) -> bool:
    """Return True if line is blank or a full-line comment."""
    stripped = line.lstrip()
    return not stripped or stripped.startswith(("#", ";"))


def set_mdp_key(lines: list[str], key: str, value: Any, *, inplace: bool = True) -> list[str]:
    """Update or append `key = value` in .mdp-like lines, preserving inline comments.

    If `inplace` is False, returns a modified copy and leaves `lines` unchanged.
    """
    mdp_value = format_gmx_value(value)
    out = lines if inplace else list(lines)
    wanted = key.strip()

    for i, ln in enumerate(out):
        if is_comment(ln):
            continue

        left, sep, right = ln.partition("=")
        if not sep or left.strip() != wanted:
            continue

        before_comment, comment = _split_inline_comment(right)
        prefix = _leading_ws(before_comment)  # preserve whitespace before old value
        out[i] = f"{left}={prefix}{mdp_value}{comment}"
        break
    else:
        # key not found -> append in aligned format
        out.append(f"{wanted:<28} = {mdp_value}")

    return out
