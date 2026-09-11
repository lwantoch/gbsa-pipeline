"""Top-level RunConfig model for driving the pipeline from a TOML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import tomllib
from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator

from gbsa_pipeline.mdp import GromacsParams
from gbsa_pipeline.parametrization import ParametrizationConfig, ParametrizationInput
from gbsa_pipeline.solvation_box import BoxShape, SolvationParams


class SystemConfig(BaseModel):
    """[system] section — input files and charge settings.

    Mutually exclusive with [membrane_system]: use this section for the
    normal path where the pipeline parametrizes a bare protein (+ ligand)
    itself; use [membrane_system] when the input is already a complete,
    solvated protein-in-bilayer system (e.g. from
    gbsa_pipeline.membrane.fetch_memprotmd_system) that only needs
    minimization/equilibration/production, not parametrization or solvation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    protein: Path
    ligand: Path | None = None
    net_charge: int | None = None


class MembraneSystemConfig(BaseModel):
    """[membrane_system] section — start from a pre-built protein-in-bilayer system.

    Structure/topology are already a complete, solvated GROMACS system (e.g.
    MemProtMD's atomistic output — see gbsa_pipeline.membrane), so setting
    this skips the parametrize and solvate stages entirely: the pipeline
    loads structure/topology directly and starts at SD minimization.
    Mutually exclusive with [system] — see RunConfig._validate_system_source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    structure: FilePath  # .pdb or .gro
    topology: FilePath  # .top


class SolvationConfig(SolvationParams):
    """[solvation] section — solvent box settings with pipeline defaults."""

    shape: BoxShape = BoxShape.TRUNCATED_OCTAHEDRON
    padding: float | None = Field(default=None, ge=0.0)
    ion_concentration: float | None = Field(default=0.15, ge=0.0)


class MinimizationConfig(BaseModel):
    """[minimization] section — energy minimization settings.

    ``define`` is passed straight through to the SD minimization stage's
    generated MDP as a GROMACS preprocessor define (e.g. ``"-DFLEX_SPC"``).
    Needed for systems assembled by external tools (e.g. MemProtMD's
    CG2AT-backmapped water) where the initial coordinates aren't precise
    enough for rigid SETTLE-constrained water: minimizing with flexible
    (bonded) water for this first pass avoids the resulting NaN potential
    energy at step 0. Confirmed against MemProtMD's own em.mdp, which sets
    exactly this define for the same reason. Leave unset for the ordinary
    protein-ligand path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nsteps: int = 10_000
    emtol: float = 10.0
    define: str | None = None


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

    Stages 1-2 are skipped when [membrane_system] is set instead of [system]
    (see MembraneSystemConfig): the structure/topology are already a
    complete, solvated system, so the pipeline loads them directly and starts
    at stage 3.

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

    A membrane-protein run replaces [system] with [membrane_system] and
    switches the barostat to semiisotropic — see docs/configuration.md and
    examples/membrane_1py6.toml for a complete worked example.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: SystemConfig | None = None
    membrane_system: MembraneSystemConfig | None = None
    forcefield: ParametrizationConfig = Field(default_factory=ParametrizationConfig)
    solvation: SolvationConfig = Field(default_factory=SolvationConfig)
    minimization: MinimizationConfig = Field(default_factory=MinimizationConfig)
    equilibration: EquilibrationConfig = Field(default_factory=EquilibrationConfig)
    npt_equilibration: NptConfig = Field(default_factory=NptConfig)
    md: GromacsParams = Field(default_factory=GromacsParams)

    @model_validator(mode="after")
    def _validate_system_source(self) -> Self:
        """Require exactly one of [system] or [membrane_system].

        [system] is the normal path: a bare protein (+ ligand) this pipeline
        parametrizes and solvates itself. [membrane_system] is for an
        already-complete, already-solvated protein-in-bilayer system (e.g.
        MemProtMD output) that skips straight to minimization. Requiring
        exactly one avoids a config that silently ignores whichever section
        it doesn't end up using.
        """
        if (self.system is None) == (self.membrane_system is None):
            raise ValueError("Exactly one of [system] or [membrane_system] must be set.")
        return self

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
            If ``system`` is ``None`` (this is a [membrane_system] run — there
            is nothing to parametrize) or ``system.ligand`` is ``None``
            (ligand is required for parametrization).
        """
        if self.system is None:
            raise ValueError(
                "to_parametrization_input() requires [system]; this config uses [membrane_system], "
                "which skips parametrization entirely (see RunConfig docstring)."
            )
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
