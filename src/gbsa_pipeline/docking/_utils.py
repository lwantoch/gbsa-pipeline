"""Internal utilities: subprocess logging, Meeko result extraction, Vina score parsing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from subprocess import CompletedProcess

VINA_SCORE_COLUMN_COUNT = 2
VINA_RANK_COLUMN_INDEX = 0
VINA_SCORE_COLUMN_INDEX = 1
VINA_TOP_RANK = "1"


def _require_file(path: Path, label: str = "File") -> Path:
    """Resolve *path* and raise if it is missing or not a regular file."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} path is not a file: {path}")
    return path


def _summarize_stderr(stderr: str, max_lines: int = 4) -> str:
    """Return a short preview of stderr suitable for terminal output.

    This helper exists because full subprocess stderr is still written to log
    files, but a compact preview is more useful for immediate terminal feedback.
    The `stderr` parameter is required because multiple external tools in this
    module can fail, and the same summarization logic should apply to all of them.
    We are currently checking only a few leading non-empty lines because this
    function is meant for human scanability, not for archival completeness.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "no stderr output"

    preview = lines[:max_lines]
    text = " | ".join(preview)

    if len(lines) > max_lines:
        text += " | ..."

    return text


def _write_process_log(
    log_path: Path,
    process: CompletedProcess[str],
    *,
    command: list[str],
    title: str,
) -> None:
    """Write complete subprocess stdout and stderr to a plain-text log file.

    This helper is deliberately separate from `_summarize_stderr()` because the
    module needs both a short terminal summary and a complete on-disk record.
    The `log_path`, `process`, `command`, and `title` parameters are all needed
    to produce a log file that is self-contained and reviewable later.
    We are currently writing command, return code, stdout, and stderr verbatim
    because that is the minimum useful forensic record for external tool failures.
    """
    resolved_log_path = Path(log_path).resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    text = (
        f"{title}\n"
        f"{'=' * len(title)}\n\n"
        f"Command:\n{' '.join(command)}\n\n"
        f"Return code:\n{process.returncode}\n\n"
        f"STDOUT\n"
        f"------\n"
        f"{process.stdout or ''}\n\n"
        f"STDERR\n"
        f"------\n"
        f"{process.stderr or ''}\n"
    )
    resolved_log_path.write_text(text, encoding="utf-8")


def _extract_pdbqt_string_from_meeko_result(result: Any) -> str:
    """Extract the PDBQT string from Meeko's version-dependent return value.

    This helper exists because Meeko's `write_string()` return shape is not
    always uniform across versions or calling contexts.
    The `result` parameter is intentionally typed as `Any` because the function
    is explicitly about normalizing a loosely specified third-party return value.
    We are currently checking only the cases needed by the observed Meeko API:
    a plain string or a tuple whose first element is the PDBQT string.
    """
    if isinstance(result, str):
        return result

    if isinstance(result, tuple):
        if not result:
            raise ValueError("Meeko returned an empty tuple from write_string().")

        pdbqt_string = result[0]

        if not isinstance(pdbqt_string, str):
            raise TypeError(
                f"Expected first element of Meeko write_string() result to be a str, got {type(result[0]).__name__}."
            )

        return pdbqt_string

    raise TypeError(f"Unexpected return type from Meeko write_string(): {type(result).__name__}")


def _parse_vina_best_score_from_log(log_path: Path) -> float | None:
    """Parse the best affinity from the written Vina log file.

    This helper exists because docking output is already persisted to disk and
    downstream parsing should use that durable record rather than in-memory
    subprocess stdout.
    The `log_path` parameter is required because the score now comes from the
    archived process log written by `_write_process_log()`.
    We are currently checking only the first-ranked pose because the current
    adapter exposes a minimal top-pose view instead of a full ranked table.

    Reference:
    https://autodock-vina.readthedocs.io/en/latest/
    """
    resolved_log_path = Path(log_path).resolve()

    if not resolved_log_path.exists():
        return None

    log_text = resolved_log_path.read_text(encoding="utf-8")

    for line in log_text.splitlines():
        columns = line.strip().split()

        if len(columns) < VINA_SCORE_COLUMN_COUNT:
            continue

        if columns[VINA_RANK_COLUMN_INDEX] != VINA_TOP_RANK:
            continue

        try:
            return float(columns[VINA_SCORE_COLUMN_INDEX])
        except ValueError:
            continue

    return None


def _build_compact_pose_metadata(
    *,
    returncode: int,
    log_file: Path,
    output_exists: bool,
    receptor_used: Path,
) -> dict[str, Any]:
    """Build minimal per-pose metadata.

    This helper exists to keep the metadata payload uniform wherever poses are
    created inside the docking engine.
    The parameters are intentionally explicit because they are the smallest set
    of run facts that are still useful for debugging and downstream checks.
    We are currently storing filesystem- and process-level facts only, not
    chemistry-level annotations, because those belong elsewhere in the pipeline.
    """
    return {
        "returncode": returncode,
        "log_file": str(Path(log_file).resolve()),
        "output_exists": output_exists,
        "receptor_used": str(Path(receptor_used).resolve()),
    }
