"""General parameter change helpers for GROMACS .mdp files."""

from __future__ import annotations

from typing import Any

# Aliases for GROMACS parameter keys whose canonical MDP spelling differs from
# the Python field name used in GromacsParams.  Only a small number of
# historically inconsistent names are listed here.
_FIELD_ALIASES: dict[str, str] = {
    "vdw_type": "vdwtype",
}


def mdp_key_to_field(key: str) -> str:
    """Convert a GROMACS MDP key to a Python model field name.

    GROMACS uses hyphens (e.g. ``cutoff-scheme``); Python fields use
    underscores.  A small alias table handles the rare keys whose canonical
    GROMACS spelling differs from the field name used in ``GromacsParams``
    (e.g. ``vdwtype`` → ``vdw_type`` field).

    Used by ``GromacsParams.from_mapping`` so callers can pass both
    GROMACS-style and Python-style keys without silent failures.
    """
    field_name = key.replace("-", "_")
    return _FIELD_ALIASES.get(field_name, field_name)


def field_to_mdp_key(field: str) -> str:
    """Convert a Python field name to a comparison-safe GROMACS MDP key.

    Returns a lowercase, hyphen-separated key for comparison and deduplication.
    This is used when merging BSS-generated MDP configs with local overrides
    where the same logical parameter may appear with different spelling
    variants (e.g. ``DispCorr``, ``dispcorr``, ``disp_corr``).
    """
    return field.strip().lower().replace("_", "-")


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
