"""AutoDock Vina engine."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import TYPE_CHECKING, Any

from gbsa_pipeline.docking._models import DockedPose, DockingBox, DockingRequest, DockingResult
from gbsa_pipeline.docking._receptor_prep import convert_receptor_pdb_to_pdbqt
from gbsa_pipeline.docking._utils import (
    _build_compact_pose_metadata,
    _parse_vina_best_score_from_log,
    _summarize_stderr,
    _write_process_log,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

LOGGER = logging.getLogger(__name__)


class VinaEngine:
    """Minimal AutoDock Vina wrapper.

    This class deliberately stays small and only covers what is needed to turn
    a validated request into one or more docked pose files plus compact metadata.
    It does not own ligand preparation, chemistry reconstruction, or box
    generation, because those are separate concerns in this module.
    We are currently keeping the engine narrowly focused on command construction,
    receptor readiness, subprocess execution, score parsing, and result assembly.
    """

    name = "vina"

    def __init__(
        self,
        binary: str = "vina",
    ) -> None:
        """Initialize the Vina executable name.

        Receptor preparation is done via the Meeko Python API and does not
        require a separate binary.
        """
        if shutil.which(binary) is None:
            LOGGER.warning("Vina binary not found in PATH: %s", binary)

        self.binary = binary

    def _build_command(
        self,
        *,
        receptor: Path,
        ligand: Path,
        output: Path,
        box: DockingBox,
        seed: int | None = None,
        num_modes: int | None = None,
        exhaustiveness: int | None = None,
        energy_range: float | None = None,
        extra_flags: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Build the Vina command line for a single ligand.

        This helper exists so command construction is isolated from the rest of
        the execution logic and can be tested directly.
        The receptor, ligand, output, and box parameters are the minimal Vina
        inputs, while the optional parameters expose common runtime controls.
        We are currently checking only the flags needed by the present workflow
        and passing any extra flags through transparently rather than validating them.

        Reference:
        https://autodock-vina.readthedocs.io/en/latest/docking_basic.html
        """
        cmd: list[str] = [
            self.binary,
            "--receptor",
            str(Path(receptor).resolve()),
            "--ligand",
            str(Path(ligand).resolve()),
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
            "--out",
            str(Path(output).resolve()),
        ]

        if seed is not None:
            cmd += ["--seed", str(seed)]

        if num_modes is not None:
            cmd += ["--num_modes", str(num_modes)]

        if exhaustiveness is not None:
            cmd += ["--exhaustiveness", str(exhaustiveness)]

        if energy_range is not None:
            cmd += ["--energy_range", str(energy_range)]

        if extra_flags:
            for key, value in extra_flags.items():
                cmd.append(str(key))
                if value is not None:
                    cmd.append(str(value))

        return cmd

    def _prepare_receptor_for_docking(
        self,
        receptor: Path,
        workdir: Path,
        cofactor_sdfs: list[Path] | None = None,
    ) -> Path:
        """Return the receptor PDBQT to use for docking.

        Accepts either a preprepared PDBQT or a plain PDB. If ``cofactor_sdfs``
        is provided, cofactors are merged into the receptor PDB before meeko
        runs so AutoDock atom types are assigned correctly for everything in one
        pass. Requires a PDB receptor when cofactors are given.
        """
        receptor = Path(receptor).resolve()

        if cofactor_sdfs and receptor.suffix.lower() != ".pdb":
            raise ValueError(
                "cofactor_sdfs requires a PDB receptor so meeko can assign atom types "
                f"to both protein and cofactors in one pass. Got: {receptor}"
            )

        if receptor.suffix.lower() == ".pdbqt":
            return receptor
        if receptor.suffix.lower() == ".pdb":
            return convert_receptor_pdb_to_pdbqt(
                receptor,
                output_path=workdir / f"{receptor.stem}.pdbqt",
                cofactor_sdfs=cofactor_sdfs or [],
            )
        raise ValueError(f"Unsupported receptor file type for docking: {receptor}")

    def dock(self, request: DockingRequest) -> DockingResult:
        """Run docking for all ligands in the request.

        This is the one method that actually executes the backend and turns a
        validated request into structured pose outputs.
        The `request` parameter is required because it carries the complete
        docking contract: receptor, ligands, box, working directory, and runtime
        options in one validated object.
        We are currently checking a minimal but useful workflow: prepare the
        receptor if needed, run Vina ligand by ligand, parse the top score from
        the persisted log file, write full logs, and return structured pose records.
        """
        workdir = request.workdir or Path.cwd()
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        receptor_for_docking = self._prepare_receptor_for_docking(
            request.receptor,
            workdir,
            cofactor_sdfs=request.cofactor_sdfs or None,
        )

        poses: list[DockedPose] = []
        raw_outputs: dict[str, Any] = {}

        for ligand_entry in request.ligands:
            ligand_path = Path(ligand_entry).resolve()
            out_file = workdir / f"{ligand_path.stem}_vina_out.pdbqt"
            log_file = workdir / f"{ligand_path.stem}_vina.log"

            cmd = self._build_command(
                receptor=receptor_for_docking,
                ligand=ligand_path,
                output=out_file,
                box=request.box,
                seed=request.seed,
                num_modes=request.parameters.get("num_modes"),
                exhaustiveness=request.parameters.get("exhaustiveness"),
                energy_range=request.parameters.get("energy_range"),
                extra_flags=request.parameters.get("extra_flags"),
            )

            process: CompletedProcess[str] = run(  # noqa: S603
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
            )

            _write_process_log(
                log_file,
                process,
                command=cmd,
                title=f"Vina docking log for {ligand_path.name}",
            )

            score = _parse_vina_best_score_from_log(log_file)
            rank = 1 if score is not None else None

            pose_metadata = _build_compact_pose_metadata(
                returncode=process.returncode,
                log_file=log_file,
                output_exists=out_file.exists(),
                receptor_used=receptor_for_docking,
            )

            poses.append(
                DockedPose(
                    ligand=ligand_path,
                    pose_path=out_file,
                    score=score,
                    rank=rank,
                    engine=self.name,
                    metadata=pose_metadata,
                )
            )

            raw_outputs[str(ligand_path)] = pose_metadata

            if process.returncode != 0:
                LOGGER.error(
                    "Vina failed for %s: return code %s; stderr summary: %s",
                    ligand_path.name,
                    process.returncode,
                    _summarize_stderr(process.stderr),
                )
            elif not out_file.exists():
                LOGGER.warning(
                    "Vina returned success but output is missing for %s",
                    ligand_path.name,
                )

        return DockingResult(
            poses=poses,
            engine=self.name,
            parameters=request.parameters,
            raw_outputs=raw_outputs,
        )
