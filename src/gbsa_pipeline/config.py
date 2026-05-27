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
from gbsa_pipeline.parametrization_enum import ChargeMethod, LigandFF, ProteinFF
from gbsa_pipeline.solvation_box import BoxShape, WaterModel


class SystemConfig(BaseModel):
    """[system] section — input files and charge settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protein: Path
    ligand: Path | None = None
    extra_ff_files: tuple[Path, ...] = ()
    net_charge: int | None = None


class ForceFieldConfig(BaseModel):
    """[forcefield] section — force field and charge method choices."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protein_ff: ProteinFF = ProteinFF.FF14SB
    ligand_ff: LigandFF = LigandFF.GAFF2
    charge_method: ChargeMethod = ChargeMethod.AM1BCC


class SolvationConfig(BaseModel):
    """[solvation] section — solvent box settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    water_model: WaterModel = WaterModel.TIP3P
    box_shape: BoxShape = BoxShape.TRUNCATED_OCTAHEDRON
    padding: float | None = None
    box_size: float | None = 8.0
    ion_concentration: float = 0.15
    neutralize: bool = True


class MinimizationConfig(BaseModel):
    """[minimization] section — energy minimization settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nsteps: int = 10_000
    emtol: float = 10.0


class EquilibrationConfig(BaseModel):
    """[equilibration] section — NVT heating settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_time_ps: float = 500.0


class RunConfig(BaseModel):
    """Top-level configuration for a complete GBSA pipeline run.

    Load from a TOML file with :meth:`from_toml`. Each section maps to a
    nested model. The ``[md]`` section accepts any field of
    :class:`~gbsa_pipeline.mdp.GromacsParams`.

    Example:
    -------
    ```toml
    [system]
    protein = "protein.pdb"
    ligand  = "ligand.sdf"

    [solvation]
    water_model = "tip3p"
    padding = 10.0

    [md]
    nsteps = 500000
    dt = 0.002
    tcoupl = "v-rescale"
    ref_t = 300.0
    ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: SystemConfig
    forcefield: ForceFieldConfig = Field(default_factory=ForceFieldConfig)
    solvation: SolvationConfig = Field(default_factory=SolvationConfig)
    minimization: MinimizationConfig = Field(default_factory=MinimizationConfig)
    equilibration: EquilibrationConfig = Field(default_factory=EquilibrationConfig)
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
            config=ParametrizationConfig(
                protein_ff=self.forcefield.protein_ff,
                ligand_ff=self.forcefield.ligand_ff,
                charge_method=self.forcefield.charge_method,
                extra_ff_files=self.system.extra_ff_files,
            ),
            net_charge=self.system.net_charge,
            work_dir=work_dir,
        )
