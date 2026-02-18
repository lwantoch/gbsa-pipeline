"""General Parameyter Change Function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def _format_gmx_value(value: Any) -> str:
    """Convert Python values to GROMACS mdp-compatible strings."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    raise TypeError(f"Unsupported mdp value type: {type(value)}")


def _set_key(lines: list[str], key: str, value: Any) -> None:
    """Replace only the value of an existing 'key = value' line, preserving inline comments.

    If the key is not present, append a new line in a standard aligned format.
    """
    mdp_value = _format_gmx_value(value)

    def _leading_ws(s: str) -> str:
        """Return leading whitespace of s."""
        return s[: len(s) - len(s.lstrip())]

    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if not s or s.startswith(("#", ";")):
            continue

        left, sep, right = ln.partition("=")
        if not sep:
            continue
        if left.strip() != key:
            continue

        # Preserve inline comments and whitespace before the value.
        for delim in (";", "#"):
            idx = right.find(delim)
            if idx != -1:
                prefix = _leading_ws(right[:idx])  # whitespace before value
                comment = right[idx:]  # including delimiter
                break
        else:
            prefix = _leading_ws(right)
            comment = ""

        lines[i] = f"{left}={prefix}{mdp_value}{comment}"
        return

    # key not found → append (no comment to preserve)
    lines.append(f"{key:<28} = {mdp_value}")


def change_default_params(lines: list[str], params: Mapping[str, Any]) -> None:
    """Apply multiple mdp parameter changes to a config (in-place)."""
    for key, value in params.items():
        _set_key(lines, key, value)
