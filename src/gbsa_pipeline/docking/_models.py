"""Docking data models: DockingBox, DockingRequest, DockedPose, DockingResult, DockingEngine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Mapping


class DockingBox(BaseModel):
    """Docking-box center and size in Angstrom.

    This model exists so Vina box inputs stay explicit and typed instead of
    being passed around as anonymous tuples or dictionaries.
    The box is required because the docking engine cannot run without a search
    region, and making it a model helps catch malformed inputs early.
    We are currently storing only center and size because that is the minimum
    stable interface needed by the Vina command-line backend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    center: tuple[float, float, float]
    size: tuple[float, float, float]


class DockingRequest(BaseModel):
    """Normalized docking request for a receptor, one or more ligands, and a box.

    This model groups the core docking inputs into one validated object so the
    engine layer receives a stable, explicit contract instead of loosely coupled
    parameters.
    The receptor, ligands, and box are required because they define the actual
    docking problem, while seed, workdir, and parameters control runtime details.
    We are currently checking mostly filesystem validity and suffix support here,
    leaving chemistry-specific validation to the preparation and export helpers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    receptor: Path
    ligands: list[Path]
    box: DockingBox
    seed: int | None = None
    workdir: Path | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("receptor")
    @classmethod
    def _check_receptor_exists(cls, path: Path) -> Path:
        """Validate that the receptor exists and has a Vina-compatible suffix.

        The `path` parameter is needed because the docking engine supports both
        prebuilt receptor PDBQT files and plain PDB files that still need
        conversion.
        We are currently checking only file existence, file-ness, and suffix,
        because structural receptor correctness belongs to external preparation
        steps rather than to this thin adapter layer.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise ValueError(f"Receptor file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Receptor path is not a file: {path}")

        if path.suffix.lower() not in {".pdb", ".pdbqt"}:
            raise ValueError(f"Receptor must be .pdb or .pdbqt for vina docking: {path}")

        return path

    @field_validator("ligands")
    @classmethod
    def _check_ligands(cls, ligands: list[Path]) -> list[Path]:
        """Validate that one or more ligand files are present on disk.

        The `ligands` parameter is a list because the current engine API is
        allowed to dock multiple prepared ligands against one receptor request.
        We are currently checking that each path exists and is a file, but not
        whether each file is chemically valid, because that belongs to ligand
        preparation and downstream docking itself.
        """
        if not ligands:
            raise ValueError("At least one ligand file required.")

        checked: list[Path] = []

        for ligand_entry in ligands:
            ligand_path = Path(ligand_entry).resolve()

            if not ligand_path.exists():
                raise ValueError(f"Ligand file missing: {ligand_path}")

            if not ligand_path.is_file():
                raise ValueError(f"Ligand path is not a file: {ligand_path}")

            checked.append(ligand_path)

        return checked


@dataclass(frozen=True)
class DockedPose:
    """Single docked pose plus compact metadata about the docking run.

    This dataclass is intentionally small and stores only what downstream stages
    need immediately: which ligand was docked, where the pose file is, which
    score was parsed, and a compact metadata bundle.
    The metadata field exists because some run details are useful but not
    important enough to deserve top-level strongly typed fields yet.
    We are currently using this object as the bridge between docking execution
    and later export or reconstruction steps.
    """

    ligand: Path
    pose_path: Path
    score: float | None
    rank: int | None
    engine: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DockingResult:
    """Collection of produced poses and compact run metadata.

    This dataclass exists so one docking request returns a single structured
    object even when multiple ligands were processed in one call.
    The `parameters` field is retained because downstream code often needs to
    know with which runtime settings the poses were generated.
    We are currently keeping this result intentionally lightweight and not
    storing large parsed outputs, because the raw log files already exist on disk.
    """

    poses: list[DockedPose]
    engine: str
    parameters: Mapping[str, Any]
    raw_outputs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DockingValidation:
    """Post-docking geometry checks for one docked pose.

    All distances are in Å.  Each entry in the clash lists is a
    ``(description, distance)`` tuple; bridge entries are
    ``(water_res_id, lig_description, prot_description, lig_dist, prot_dist)``.
    """

    ligand_protein_clashes: list[tuple[str, float]]
    ligand_water_clashes: list[tuple[str, float]]
    water_protein_clashes: list[tuple[str, float]]
    water_bridges: list[tuple[str, str, str, float, float]]

    @property
    def has_clashes(self) -> bool:
        """Return True when any clash list is non-empty."""
        return bool(self.ligand_protein_clashes or self.ligand_water_clashes or self.water_protein_clashes)


@dataclass(frozen=True)
class DockingManifest:
    """Complete record of a dual docking run with and without active-site crystal waters.

    ``with_waters`` is ``None`` when no crystal waters passed the selection
    criteria (inside box + near ligand + no receptor clash).
    """

    without_waters: DockingResult
    with_waters: DockingResult | None
    score_without: float | None
    score_with: float | None
    retained_water_ids: list[str]
    receptor_without_waters: Path
    receptor_with_waters: Path | None
    validation_without: DockingValidation | None
    validation_with: DockingValidation | None


class DockingEngine(Protocol):
    """Protocol implemented by docking backends.

    This protocol keeps the rest of the codebase independent from one concrete
    docking backend implementation.
    The `dock()` signature is the minimal stable contract the runner or higher
    workflow code needs in order to call a docking engine generically.
    We are currently using only Vina, but this protocol avoids hard-coding that
    assumption into every consumer of the docking module.
    """

    name: str

    def dock(self, request: DockingRequest) -> DockingResult:
        """Run docking for one validated docking request.

        This method defines the minimal operation a docking backend must expose
        so the rest of the code can treat concrete engines uniformly.
        The `request` parameter is required because it bundles receptor, ligands,
        box, work directory, and runtime parameters into one validated object.
        We are currently requiring only a structured `DockingResult` return value,
        because downstream workflow code should not need to parse backend-specific
        subprocess results directly.
        """
