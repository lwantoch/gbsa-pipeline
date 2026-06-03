"""Receptor preparation: PDB → PDBQT conversion and crystal-water merging."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from subprocess import CompletedProcess, run

from gbsa_pipeline.docking._utils import _require_file, _summarize_stderr, _write_process_log

LOGGER = logging.getLogger(__name__)


def _strip_hetatm(receptor_pdb: Path, dest: Path) -> Path:
    """Write a copy of receptor_pdb with HETATM records removed.

    Meeko processes ATOM and HETATM identically, so modified residues stored
    as HETATM (e.g. CSD, PTR) that share a residue number with a protein ATOM
    residue cause ``each residue key must have exactly 1 resname`` errors.
    Docking receptors should only contain protein atoms anyway.
    """
    lines = receptor_pdb.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith("HETATM")]
    dest.write_text("".join(kept), encoding="utf-8")
    return dest


def convert_receptor_pdb_to_pdbqt(
    receptor_pdb: Path,
    output_path: Path | None = None,
    *,
    mk_prepare_receptor_binary: str = "mk_prepare_receptor.py",
) -> Path:
    """Convert a receptor PDB to rigid receptor PDBQT using Meeko.

    This helper exists because docking often starts from a receptor PDB even
    when Vina requires a receptor PDBQT for execution.
    The `mk_prepare_receptor_binary` parameter is configurable because different
    environments may expose Meeko command-line tools through wrappers or explicit
    executable names.
    We currently use Meeko's `--read_pdb` path for plain PDB input so this helper
    does not require ProDy for the simple rigid-receptor workflow tested here.
    Receptor hydrogen addition, protonation-state decisions, and structural
    cleanup are still expected to happen upstream.
    """
    receptor_pdb = _require_file(Path(receptor_pdb), "Receptor PDB")

    if receptor_pdb.suffix.lower() != ".pdb":
        raise ValueError(f"Expected a .pdb receptor input, got: {receptor_pdb}")

    if shutil.which(mk_prepare_receptor_binary) is None:
        raise RuntimeError(f"Meeko receptor executable not found in PATH: {mk_prepare_receptor_binary}")

    if output_path is None:
        output_path = receptor_pdb.with_suffix(".pdbqt")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_base = output_path.with_suffix("")
    log_path = output_path.with_suffix(".meeko_receptor.log")

    # Strip HETATM before Meeko so modified residues (CSD, PTR, …) stored as
    # HETATM with the same residue number as a protein ATOM don't cause errors.
    pdb_for_meeko = _strip_hetatm(receptor_pdb, output_path.parent / f"{receptor_pdb.stem}_protein_only.pdb")

    def _run_meeko(extra_args: list[str] = []) -> tuple[CompletedProcess[str], list[str]]:  # noqa: B006
        _cmd = [
            mk_prepare_receptor_binary,
            "--read_pdb",
            str(pdb_for_meeko),
            "-o",
            str(output_base),
            "-p",
            *extra_args,
        ]
        return run(_cmd, capture_output=True, text=True, check=False), _cmd  # noqa: S603

    LOGGER.info(
        "Preparing receptor with Meeko: %s -> %s",
        receptor_pdb.name,
        output_path.name,
    )

    process, cmd = _run_meeko()

    # Cross-chain disulfide bonds cause "Expected N paddings … got 0". Retry
    # with --set_template {chain}:{res}=CYX so Meeko uses the cystine template.
    if process.returncode != 0 and "paddings" in process.stderr:
        residues = re.findall(r"matched with excess inter-residue bond\(s\): (\S+)", process.stderr)
        if residues:
            template_arg = ",".join(f"{r}=CYX" for r in residues)
            LOGGER.warning(
                "Meeko: cross-chain CYS bonds detected (%s), retrying with --set_template %s",
                ", ".join(residues), template_arg,
            )
            process, cmd = _run_meeko(["--set_template", template_arg])

    _write_process_log(
        log_path,
        process,
        command=cmd,
        title=f"Meeko receptor preparation log for {receptor_pdb.name}",
    )

    if process.returncode != 0:
        raise RuntimeError(
            "Meeko receptor preparation failed.\n"
            f"Receptor: {receptor_pdb}\n"
            f"Log: {log_path}\n"
            f"stderr summary: {_summarize_stderr(process.stderr)}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"Meeko reported success but receptor PDBQT output is missing.\nExpected: {output_path}\nLog: {log_path}"
        )

    if process.stderr.strip():
        LOGGER.warning(
            "Meeko receptor preparation finished with warnings; full details in %s",
            log_path.name,
        )
    else:
        LOGGER.info("Meeko receptor PDBQT written: %s", output_path.name)

    return output_path


def prepare_receptor_with_crystal_waters(
    receptor_pdb: Path,
    crystal_waters_pdb: Path,
    output_pdb: Path,
) -> Path:
    """Merge selected crystal waters into a receptor PDB for docking.

    Waters are appended after the protein records so Meeko and Vina treat them
    as part of the rigid receptor.
    """
    protein_lines = [
        line for line in receptor_pdb.read_text(encoding="utf-8").splitlines() if not line.startswith(("TER", "END"))
    ]
    water_lines = [
        line for line in crystal_waters_pdb.read_text(encoding="utf-8").splitlines() if not line.startswith("END")
    ]
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(protein_lines + water_lines) + "\nEND\n", encoding="utf-8")
    return output_pdb
