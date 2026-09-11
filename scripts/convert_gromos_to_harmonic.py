#!/usr/bin/env python3
r"""Convert GROMOS G96 bonds/angles (funct=2) to harmonic ones (funct=1).

Standalone data-prep utility, deliberately kept OUT of gbsa_pipeline: it is a
one-off fix for a specific incompatibility, not something the pipeline should
run on every membrane_system load.

Why this exists
----------------
MemProtMD's atomistic output (and GROMOS force fields generally) writes every
bond and angle in the G96 functional form (``funct = 2``), referenced by a
``#define``d macro name (e.g. ``gb_21``, ``ga_10``) rather than inline
numbers. Real ``gmx grompp`` handles this fine. BioSimSpace/Sire can *read*
G96 terms (with a "NOT HARMONIC!" warning) but its GROMACS-topology *writer*
cannot write them back out::

    Sire cannot yet interpret bonds that are not in a standard harmonic
    format! (93.8098 [r^2 - 1]^2)
    Sire cannot yet interpret angles that are not in a standard harmonic
    format! (63.3365 [cos(theta) + 0.358368]^2)

``BSS.Process.Gromacs`` always re-serializes its in-memory system through
that writer before invoking any GROMACS tool, so both of these fail on every
MD stage for a GROMOS-parametrized system, not just at load time — confirmed
against MemProtMD's 1py6 (bonds first, then angles once bonds were fixed).

The fix: replace each G96 term with a harmonic one that matches its
curvature at the equilibrium value.

* Bonds — G96: V = (k/4)(r^2 - r0^2)^2, harmonic: V = (k/2)(r - r0)^2.
  Matching second derivatives at r = r0 gives::

      k_harmonic = 2 * k_g96 * r0^2

* Angles — G96: V = (k/2)(cos(theta) - cos(theta0))^2, harmonic:
  V = (k/2)(theta - theta0)^2 (theta in radians internally, though both
  forms list theta0 in degrees in the .itp/.top text). Matching second
  derivatives at theta = theta0 gives::

      k_harmonic = k_g96 * sin(theta0)^2

Both are the standard approximations used whenever a G96 term needs to be
represented in an engine that only supports harmonic forms; AMBER/CHARMM/GAFF
force fields never use G96 terms in the first place, so this only matters
when starting from a GROMOS-parametrized system (as here). Dihedrals are left
untouched: GROMOS dihedrals use the ordinary periodic (funct=1) and harmonic
improper (funct=2) forms already, both of which Sire's writer supports.

Usage
-----
::

    python scripts/convert_gromos_to_harmonic.py \\
        tests/testdata/membrane/1py6 tests/testdata/membrane/1py6_harmonic

Copies the whole source directory (so #include paths between topol.top and
its itp/ files keep working unchanged) and rewrites the [bonds]/[angles]
sections of every .top/.itp file under the destination that has funct=2
entries. The gb_*/ga_* macro definitions come from gromos53a6.ff/ffbonded.itp,
which ships with GROMACS itself (found by locating `gmx` on PATH, unless
--ffbonded is given). Point [membrane_system].topology at the converted
topol.top afterwards.

Known limitation -- necessary but not sufficient
--------------------------------------------------
This fixes bonds/angles so BioSimSpace/Sire can load *and* re-save the
topology without erroring. It does NOT fix a second, separate BioSimSpace/
Sire bug: its GROTOP writer silently drops `[nonbond_params]`/`[pairtypes]`
override tables when re-serializing (confirmed: MemProtMD's
itp/lipid-gmx53a6.itp has 541 + 105 such entries for its OPLS-style lipid
atom types; 0 survive a BSS.IO round-trip). BSS.Process.Gromacs always
re-serializes before invoking any GROMACS tool, so for MemProtMD's 1py6 this
loses enough of the lipid's nonbonded parameters that minimization diverges
to a NaN potential energy at step 0 -- confirmed even though the identical
post-conversion files minimize cleanly (3 steps, finite energy) under plain
`gmx grompp`/`mdrun` outside BioSimSpace entirely. That data loss is a
BioSimSpace/Sire library bug, out of scope for this script or for
gbsa_pipeline to work around.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFINE_RE = re.compile(r"^#define\s+(\S+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*(?:;.*)?$")

# [ bonds ]: ai aj funct <macro | b0 kb>
_BOND_MACRO_RE = re.compile(r"^(\s*\d+\s+\d+\s+)2(\s+)(\S+)(\s*;.*)?$")
_BOND_NUMERIC_RE = re.compile(r"^(\s*\d+\s+\d+\s+)2(\s+)([-\d.eE+]+)\s+([-\d.eE+]+)(\s*;.*)?$")

# [ angles ]: ai aj ak funct <macro | theta0 ka>
_ANGLE_MACRO_RE = re.compile(r"^(\s*\d+\s+\d+\s+\d+\s+)2(\s+)(\S+)(\s*;.*)?$")
_ANGLE_NUMERIC_RE = re.compile(r"^(\s*\d+\s+\d+\s+\d+\s+)2(\s+)([-\d.eE+]+)\s+([-\d.eE+]+)(\s*;.*)?$")


def find_ffbonded_itp(gmx: str = "gmx") -> Path:
    """Locate gromos53a6.ff/ffbonded.itp next to the `gmx` executable on PATH.

    GROMACS ships its bundled force fields under <prefix>/share/gromacs/top/,
    where <prefix> is two directories up from the `gmx` binary itself (e.g.
    .../envs/dev/bin.AVX2_256/gmx -> .../envs/dev/share/gromacs/top/...).
    Raises FileNotFoundError with a clear message if gmx isn't on PATH or the
    expected file isn't where it should be.
    """
    gmx_path = shutil.which(gmx)
    if gmx_path is None:
        raise FileNotFoundError(f"'{gmx}' not found on PATH; pass --ffbonded explicitly.")
    prefix = Path(gmx_path).resolve().parent.parent
    ffbonded = prefix / "share" / "gromacs" / "top" / "gromos53a6.ff" / "ffbonded.itp"
    if not ffbonded.exists():
        raise FileNotFoundError(
            f"Expected {ffbonded} next to '{gmx}' ({gmx_path}) but it doesn't exist; pass --ffbonded explicitly."
        )
    return ffbonded


def load_g96_macros(ffbonded_itp: Path, prefix: str) -> dict[str, tuple[float, float]]:
    """Parse `#define <prefix>_XX v0 k` lines into {macro_name: (v0, k)}.

    Only ``#define`` lines are read (a simple line-by-line regex, not a full
    C-preprocessor); ffbonded.itp defines bond/angle macros as flat top-level
    ``#define`` statements with no conditionals around them, so this is
    sufficient. ``prefix`` is ``"gb_"`` for bonds or ``"ga_"`` for angles.
    """
    macros: dict[str, tuple[float, float]] = {}
    for line in ffbonded_itp.read_text().splitlines():
        m = _DEFINE_RE.match(line.strip())
        if m and m.group(1).startswith(prefix):
            macros[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return macros


def _harmonic_bond_k(b0: float, kb_g96: float) -> float:
    """k_harmonic = 2 * k_g96 * b0^2 -- matches curvature at r = b0."""
    return 2.0 * kb_g96 * b0**2


def _harmonic_angle_k(theta0_deg: float, ka_g96: float) -> float:
    """k_harmonic = k_g96 * sin(theta0)^2 -- matches curvature at theta = theta0."""
    return ka_g96 * math.sin(math.radians(theta0_deg)) ** 2


def _convert_section(
    lines: list[str],
    macros: dict[str, tuple[float, float]],
    macro_re: re.Pattern[str],
    numeric_re: re.Pattern[str],
    harmonic_k: Callable[[float, float], float],
) -> tuple[list[str], int]:
    """Rewrite funct=2 lines in one [ bonds ] or [ angles ] block to funct=1.

    Shared by both sections: they differ only in column count (2 atoms vs.
    3) and the curvature-matching formula, both handled via the passed-in
    regexes and ``harmonic_k``. Lines with funct != 2, blank lines, and
    comments pass through unchanged.
    """
    out: list[str] = []
    n_converted = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            out.append(line)
            continue

        m_macro = macro_re.match(line)
        m_numeric = numeric_re.match(line)
        if m_macro is not None and m_macro.group(3) in macros:
            prefix, gap, macro_name, comment = m_macro.groups()
            v0, k_g96 = macros[macro_name]
            note = f"; g96->harmonic (was {macro_name})"
        elif m_numeric is not None:
            prefix, gap, v0_str, k_str, comment = m_numeric.groups()
            v0, k_g96 = float(v0_str), float(k_str)
            note = "; g96->harmonic"
        else:
            out.append(line)
            continue

        k_harmonic = harmonic_k(v0, k_g96)
        comment_text = comment.strip() if comment else ""
        out.append(f"{prefix}1{gap}{v0:.5f}  {k_harmonic:.6e}  {note} {comment_text}\n".rstrip() + "\n")
        n_converted += 1

    return out, n_converted


def convert_file(
    path: Path,
    bond_macros: dict[str, tuple[float, float]],
    angle_macros: dict[str, tuple[float, float]],
) -> int:
    """Rewrite every [ bonds ] and [ angles ] section in *path* in place.

    Returns the total number of terms converted.
    """
    section_handlers = {
        "[ bonds ]": (bond_macros, _BOND_MACRO_RE, _BOND_NUMERIC_RE, _harmonic_bond_k),
        "[ angles ]": (angle_macros, _ANGLE_MACRO_RE, _ANGLE_NUMERIC_RE, _harmonic_angle_k),
    }

    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    total_converted = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        handler = section_handlers.get(line.strip())
        if handler is None:
            continue
        macros, macro_re, numeric_re, harmonic_k = handler
        section_start = i
        while i < len(lines) and not lines[i].strip().startswith("["):
            i += 1
        converted, n = _convert_section(lines[section_start:i], macros, macro_re, numeric_re, harmonic_k)
        out.extend(converted)
        total_converted += n

    if total_converted:
        path.write_text("".join(out))
    return total_converted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="Directory containing topol.top and its included .itp files")
    parser.add_argument("dest_dir", type=Path, help="Where to write the converted copy (created if missing)")
    parser.add_argument(
        "--ffbonded",
        type=Path,
        default=None,
        help="Path to gromos53a6.ff/ffbonded.itp (default: auto-detect next to `gmx` on PATH)",
    )
    args = parser.parse_args(argv)

    ffbonded = args.ffbonded or find_ffbonded_itp()
    bond_macros = load_g96_macros(ffbonded, "gb_")
    angle_macros = load_g96_macros(ffbonded, "ga_")
    print(f"Loaded {len(bond_macros)} gb_* and {len(angle_macros)} ga_* macros from {ffbonded}")

    if args.dest_dir.exists():
        parser.error(f"{args.dest_dir} already exists; remove it first or choose a different destination.")
    shutil.copytree(args.source_dir, args.dest_dir)

    total = 0
    for path in sorted(args.dest_dir.rglob("*")):
        if path.suffix not in (".top", ".itp"):
            continue
        n = convert_file(path, bond_macros, angle_macros)
        if n:
            print(f"  {path.relative_to(args.dest_dir)}: converted {n} bond/angle terms")
        total += n

    print(f"Converted {total} G96 bond/angle terms to harmonic under {args.dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
