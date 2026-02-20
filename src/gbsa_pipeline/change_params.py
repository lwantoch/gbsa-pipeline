"""General parameter change helpers for GROMACS .mdp files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def _leading_ws(s: str) -> str:
    """Return the leading whitespace of `s`."""
    i = 0
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    return s[:i]


def _split_inline_comment(s: str) -> tuple[str, str]:
    """Split `s` into (before_comment, comment) for inline ';' or '#'.

    Returns the earliest comment delimiter occurrence. If no delimiter exists,
    returns (s, "").
    """
    semi = s.find(";")
    hash_ = s.find("#")

    if semi == -1 and hash_ == -1:
        return s, ""
    if semi == -1:
        idx = hash_
    elif hash_ == -1:
        idx = semi
    else:
        idx = min(semi, hash_)

    return s[:idx], s[idx:]


def format_gmx_value(value: Any) -> str:
    """Convert Python values to GROMACS .mdp-compatible strings."""
    if hasattr(value, "value"):  # Enum / StrEnum
        value = value.value

    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()

    raise TypeError(f"Unsupported .mdp value type: {type(value).__name__}")


def is_comment(line: str) -> bool:
    """Return True if line is blank or a full-line comment."""
    stripped = line.lstrip()
    return not stripped or stripped.startswith(("#", ";"))


def set_mdp_key(lines: list[str], key: str, value: Any, *, inplace: bool = True) -> list[str]:
    """Update or append `key = value` in .mdp-like lines, preserving inline comments.

    If `inplace` is False, returns a modified copy and leaves `lines` unchanged.
    """
    mdp_value = format_gmx_value(value)
    out = lines if inplace else list(lines)
    wanted = key.strip()

    for i, ln in enumerate(out):
        if is_comment(ln):
            continue

        left, sep, right = ln.partition("=")
        if not sep or left.strip() != wanted:
            continue

        before_comment, comment = _split_inline_comment(right)
        prefix = _leading_ws(before_comment)  # preserve whitespace before old value
        out[i] = f"{left}={prefix}{mdp_value}{comment}"
        break
    else:
        # key not found -> append in aligned format
        out.append(f"{wanted:<28} = {mdp_value}")

    return out


def change_default_params(lines: list[str], params: Mapping[str, Any]) -> None:
    """Apply multiple .mdp parameter changes to `lines` (in-place)."""
    for k, v in params.items():
        set_mdp_key(lines, k, v, inplace=True)
