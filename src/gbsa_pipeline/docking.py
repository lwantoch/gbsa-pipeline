"""Lightweight docking adapter layer for Vina and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import TYPE_CHECKING, Any, Protocol

from meeko import MoleculePreparation, PDBQTWriterLegacy
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

if TYPE_CHECKING:
    from collections.abc import Mapping


class DockingBox(BaseModel):
    """Defines the docking box center and size."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    center: tuple[float, float, float]
    size: tuple[float, float, float]


class DockingRequest(BaseModel):
    """Normalized docking request for an engine run."""

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
        if not path.exists():
            raise ValueError(f"Receptor file does not exist: {path}")
        return path

    @field_validator("ligands")
    @classmethod
    def _check_ligands(cls, ligands: list[Path]) -> list[Path]:
        if not ligands:
            raise ValueError("At least one ligand file must be provided")

        missing = [path for path in ligands if not path.exists()]
        if missing:
            missing_list = ", ".join(str(p) for p in missing)
            raise ValueError(f"Ligand files do not exist: {missing_list}")

        return ligands


@dataclass(frozen=True)
class DockedPose:
    """Single docked pose with optional score and rank."""

    ligand: Path
    pose_path: Path
    score: float | None
    rank: int | None
    engine: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DockingResult:
    """Collection of docked poses and parameters used."""

    poses: list[DockedPose]
    engine: str
    parameters: Mapping[str, Any]
    raw_outputs: Mapping[str, Any] = field(default_factory=dict)


class DockingEngine(Protocol):
    """Protocol for pluggable docking engines."""

    name: str

    def dock(self, request: DockingRequest) -> DockingResult:
        """Run docking and return a normalized result."""
        raise NotImplementedError


def prepare_ligand_with_meeko(
    ligand: str | Chem.Mol,
    output_path: Path,
    name: str | None = None,
) -> Path:
    """Prepare a ligand (SMILES or RDKit Mol) to PDBQT using Meeko.

    Generates 3D coordinates if missing (ETKDG + UFF) before Meeko preparation.
    """
    if isinstance(ligand, str):
        mol = Chem.MolFromSmiles(ligand)
        if mol is None:
            raise ValueError("Failed to parse SMILES string for ligand preparation.")
        ligand_name = name or "LIG"
        mol.SetProp("_Name", ligand_name)
    else:
        mol = ligand

    mol = Chem.AddHs(mol)
    if mol.GetNumConformers() == 0:
        status = EmbedMolecule(mol)
        if status != 0:
            raise ValueError("Failed to embed ligand to generate 3D coordinates.")
        UFFOptimizeMolecule(mol, maxIters=200)

    preparer = MoleculePreparation()
    setups = preparer.prepare(mol)
    writer = PDBQTWriterLegacy()
    pdbqt = writer.write_string(setups[0])[0]

    output_path = output_path.resolve()
    output_path.write_text(pdbqt)
    return output_path


class _VinaLikeEngine:
    """Shared helpers for Vina-compatible command-line engines."""

    name = "vina-like"

    def __init__(self, binary: str):
        """Store the engine binary name or path."""
        self.binary = binary

    def _build_command(
        self,
        receptor: Path,
        ligand: Path,
        output: Path,
        box: DockingBox,
        seed: int | None,
        num_modes: int,
        exhaustiveness: int,
        energy_range: float | None,
        extra_flags: Mapping[str, str | int | float | None] | None = None,
    ) -> list[str]:
        """Assemble CLI arguments for a docking run."""
        cmd: list[str] = [self.binary]

        cmd.extend(
            [
                "--receptor",
                str(receptor),
                "--ligand",
                str(ligand),
                "--out",
                str(output),
                "--center_x",
                str(box.center[0]),
                "--center_y",
                str(box.center[1]),
                "--center_z",
                str(box.center[2]),
                "--size_x",
                str(box.size[0]),
                "--size_y",
                str(box.size[1]),
                "--size_z",
                str(box.size[2]),
                "--num_modes",
                str(num_modes),
                "--exhaustiveness",
                str(exhaustiveness),
            ]
        )

        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if energy_range is not None:
            cmd.extend(["--energy_range", str(energy_range)])

        if extra_flags:
            for flag, value in extra_flags.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(value)])

        return cmd

    def _run_one(
        self,
        receptor: Path,
        ligand: Path,
        request: DockingRequest,
        num_modes: int,
        exhaustiveness: int,
        energy_range: float | None,
        extra_flags: Mapping[str, str | int | float | None] | None,
    ) -> DockedPose:
        """Run docking for a single ligand and return pose info."""
        workdir = request.workdir.resolve() if request.workdir else Path.cwd()
        output = workdir / f"{ligand.stem}_{self.name}.pdbqt"

        cmd = self._build_command(
            receptor=receptor,
            ligand=ligand,
            output=output,
            box=request.box,
            seed=request.seed,
            num_modes=num_modes,
            exhaustiveness=exhaustiveness,
            energy_range=energy_range,
            extra_flags=extra_flags,
        )

        completed = self._run_command(cmd, workdir)
        return DockedPose(
            ligand=ligand,
            pose_path=output,
            score=None,
            rank=None,
            engine=self.name,
            metadata={
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            },
        )

    @staticmethod
    def _run_command(cmd: list[str], workdir: Path) -> CompletedProcess[str]:
        """Execute a docking command and capture output."""
        return run(cmd, cwd=workdir, capture_output=True, text=True, check=False)  # noqa: S603


class VinaEngine(_VinaLikeEngine):
    """Adapter for AutoDock Vina-compatible binaries."""

    name = "vina"

    def __init__(self, binary: str = "vina"):
        """Create a Vina-compatible engine wrapper."""
        super().__init__(binary=binary)

    def dock(
        self,
        request: DockingRequest,
        num_modes: int = 9,
        exhaustiveness: int = 8,
        energy_range: float | None = None,
        cpu: int | None = None,
    ) -> DockingResult:
        """Run AutoDock Vina docking."""
        flags: dict[str, str | int | float | None] = {}
        if cpu is not None:
            flags["--cpu"] = cpu

        poses = [
            self._run_one(
                receptor=request.receptor,
                ligand=Path(ligand),
                request=request,
                num_modes=num_modes,
                exhaustiveness=exhaustiveness,
                energy_range=energy_range,
                extra_flags=flags,
            )
            for ligand in request.ligands
        ]

        return DockingResult(
            poses=poses,
            engine=self.name,
            parameters={
                "num_modes": num_modes,
                "exhaustiveness": exhaustiveness,
                "energy_range": energy_range,
                "cpu": cpu,
                "seed": request.seed,
            },
            raw_outputs={},
        )
