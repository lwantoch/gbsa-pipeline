"""Adapting the general change funtion for GROMACS formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_value(raw: str) -> Any:
    """Parse values from key=value text into bool/int/float/str."""
    text = raw.strip()
    low = text.lower()

    if low in {"yes", "true", "on"}:
        return True
    if low in {"no", "false", "off"}:
        return False

    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _read_changes_file(path: str | Path) -> dict[str, Any]:
    """Read non-default parameters from a key=value file."""
    changes_path = Path(path)
    changes: dict[str, Any] = {}

    for line in changes_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", ";")):
            continue

        key, has_eq, raw_value = s.partition("=")
        if not has_eq:
            continue

        changes[key.strip()] = _parse_value(raw_value)

    return changes
