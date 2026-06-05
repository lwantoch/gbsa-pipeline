"""Top-level RunConfig model for driving the pipeline from a TOML file."""

from __future__ import annotations

from pathlib import (
    Path,  # noqa: TC003 — Pydantic needs Path at runtime to resolve field types
)
from typing import Any

import tomllib
from pydantic import BaseModel, ConfigDict, Field

from gbsa_pipeline.mdp import GromacsParams
from gbsa_pipeline.parametrization import ParametrizationConfig, ParametrizationInput
from gbsa_pipeline.solvation_box import BoxShape, SolvationParams


class SystemConfig(BaseModel):
    """[system] section — input files and charge settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protein: Path
    ligand: Path | None = None
    net_charge: int | None = None


class SolvationConfig(SolvationParams):
    """[solvation] section — solvent box settings with pipeline defaults."""

    shape: BoxShape = BoxShape.TRUNCATED_OCTAHEDRON
    padding: float | None = Field(default=None, ge=0.0)
    ion_concentration: float | None = Field(default=0.15, ge=0.0)


class MinimizationConfig(BaseModel):
    """[minimization] section — energy minimization settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nsteps: int = 10_000
    emtol: float = 10.0


class EquilibrationConfig(BaseModel):
    """[equilibration] section — NVT heating settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_time_ps: float = 50.0


class NptConfig(BaseModel):
    """[npt_equilibration] section — NPT equilibration time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_time_ps: float = 100.0


class RunConfig(BaseModel):
    """Top-level configuration for a complete GBSA pipeline run.

    Load from a TOML file with :meth:`from_toml`. Each section maps to a
    nested model. The ``[md]`` section accepts any field of
    :class:`~gbsa_pipeline.mdp.GromacsParams`.

    Stages (in order):
    1. Parametrize  2. Solvate (BSS)  3. SD minimization  4. CG minimization
    5. NVT restrained heating  6. NPT restrained  7. NPT unrestrained
    8. Production MD

    Example:
    -------
    ```toml
    [system]
    protein = "protein.pdb"
    ligand  = "ligand.sdf"

    [solvation]
    water_model = "tip3p"
    padding = 1.0

    [equilibration]
    simulation_time_ps = 50.0

    [npt_equilibration]
    simulation_time_ps = 100.0

    [md]
    nsteps = 250000
    dt = 0.002
    tcoupl = "v-rescale"
    ref_t = 300.0
    ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: SystemConfig
    forcefield: ParametrizationConfig = Field(default_factory=ParametrizationConfig)
    solvation: SolvationConfig = Field(default_factory=SolvationConfig)
    minimization: MinimizationConfig = Field(default_factory=MinimizationConfig)
    equilibration: EquilibrationConfig = Field(default_factory=EquilibrationConfig)
    npt_equilibration: NptConfig = Field(default_factory=NptConfig)
    md: GromacsParams = Field(default_factory=GromacsParams)

    @classmethod
    def from_toml(cls, path: Path) -> RunConfig:
        """Load and validate a :class:`RunConfig` from a TOML file.

        Parameters
        ----------
        path:
            Path to the ``.toml`` configuration file.

        Returns:
        -------
        RunConfig
            Validated configuration object.
        """
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        return cls.model_validate(data)

    def to_parametrization_input(self, work_dir: Path) -> ParametrizationInput:
        """Build a :class:`~gbsa_pipeline.parametrization.ParametrizationInput` from this config.

        Parameters
        ----------
        work_dir:
            Directory where parametrization output files will be written.

        Returns:
        -------
        ParametrizationInput
            Ready to pass to :func:`~gbsa_pipeline.parametrization.parametrize`.

        Raises:
        ------
        ValueError
            If ``system.ligand`` is ``None`` (ligand is required for parametrization).
        """
        if self.system.ligand is None:
            raise ValueError(
                "system.ligand must be set to run the parametrization stage. "
                "Provide a ligand SDF path in the [system] section of your config."
            )
        return ParametrizationInput(
            protein_pdb=self.system.protein,
            ligand_sdf=self.system.ligand,
            config=self.forcefield,
            net_charge=self.system.net_charge,
            work_dir=work_dir,
        )
