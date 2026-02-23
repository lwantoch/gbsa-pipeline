"""Creating custom protocol with one setter per GROMACS parameter."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import Enum
from tempfile import NamedTemporaryFile
from typing import Any

import BioSimSpace as BSS
from pydantic import BaseModel, ConfigDict

from gbsa_pipeline.change_defaults_enum import (
    Barostat,
    CommMode,
    ConstraintsAlgorithms,
    CoulombModifier,
    CoulombType,
    DispCorr,
    Integrator,
    LJPMECombination,
    NghCutoffScheme,
    VDWModifier,
    VDWType,
)
from gbsa_pipeline.change_params import format_gmx_value

logger = logging.getLogger(__name__)


class GromacsParams(BaseModel):
    """MDP parameters with validated defaults and serialization helpers."""

    model_config = ConfigDict(frozen=True, validate_default=True)

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
    vdw_type: VDWType = VDWType.CUT_OFF
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
    constraint_algorithm: ConstraintsAlgorithms = ConstraintsAlgorithms.LINCS
    continuation: bool = False
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

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> GromacsParams:
        """Instantiate from a mapping using hyphenated keys."""
        kwargs = {}

        for key, value in mapping.items():
            field_name = key.replace("-", "_")
            if field_name not in cls.model_fields:
                raise KeyError(f"Unknown parameter key: {key}")
            kwargs[field_name] = value

        return cls(**kwargs)

    def to_mapping(self) -> dict[str, Any]:
        """Return a GROMACS-style mapping (underscores -> hyphens)."""
        result = {}

        for field_name, field_value in self.model_dump().items():
            serialized = field_value.value if isinstance(field_value, Enum) else field_value

            key = field_name.replace("_", "-")
            result[key] = serialized

        return result

    def to_mdp_lines(self) -> list[str]:
        """Render parameters as mdp lines without any base file."""
        lines: list[str] = []
        for key, value in self.to_mapping().items():
            lines.append(f"{key} = {format_gmx_value(value)}")
        return lines

    def to_mdp(self) -> str:
        """Render parameters as mdp text (newline-terminated)."""
        return "\n".join(self.to_mdp_lines()) + "\n"


class GromacsCustom(BSS.Protocol.Custom):
    """Thin wrapper bridging `GromacsParams` to `BSS.Protocol.Custom`."""

    def __init__(
        self,
        params: GromacsParams | Mapping[str, Any] | None = None,
    ) -> None:
        """Create a custom protocol from params only (no base mdp)."""
        self.params = (GromacsParams.from_mapping(params) if isinstance(params, Mapping) else params) or GromacsParams()

        lines = self.params.to_mdp_lines()

        with NamedTemporaryFile("w", suffix=".mdp", delete=False, encoding="utf-8") as tmp:
            tmp.write("\n".join(lines) + "\n")
            mdp_path = tmp.name

        super().__init__(mdp_path)

        self._parameters = self.params.to_mapping()


# ============================================================================
# Run helper
# ============================================================================


def run_gro_custom(
    parameters: Mapping[str, Any] | GromacsParams | None,
    system: BSS._SireWrappers.System,
    changes: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | GromacsParams | None = None,
) -> tuple[BSS._SireWrappers.System, BSS.Protocol]:
    """Create protocol from params only, run GROMACS, return (system, protocol)."""
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
    process = BSS.Process.Gromacs(system, protocol=custom_protocol)
    process.start()
    process.wait()
    logger.info("Process finished.")

    customized = process.getSystem(block=True)
    return customized, custom_protocol
