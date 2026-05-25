"""BioSimSpace MD protocol helpers for gbsa-pipeline.

This module contains the BioSimSpace building blocks for the MD stage.  The
helpers assume that the caller already provides a valid parametrized
BioSimSpace/Sire system; parametrization, solvation, file conversion, and
workflow orchestration belong to separate pipeline layers.

Each public function creates a normal BioSimSpace protocol object first so that
BSS owns the base GROMACS setup (restraint topology, pressure-coupling flags,
trajectory output, etc.).  A validated set of MDP overrides is then applied on
top to address failure modes discovered during integration testing.  Callers
can supply their own ``params`` dict to replace the module defaults entirely;
the two-layer design (BSS base + MDP override) is preserved in both cases.

Module-level override dicts
---------------------------
``_HEATING_STABILITY_PARAMS``
    Applied to the NVT heating stage when the caller passes ``params=None``.
    Disables MTS and relaxes the LINCS warning angle.  Crystal waters must be
    excluded from the input structure (see ``parametrization.py``) to avoid
    SETTLE failures caused by crystallographic clashes.

``_NPT_STABILITY_PARAMS``
    Applied to restrained NPT, unrestrained NPT, and production stages when the
    caller passes ``params=None``.  Uses a 2 fs timestep (safe with h-bonds
    constraints) and the relaxed LINCS warning angle.  Velocity generation and
    continuation flags are NOT included here because they are controlled by the
    ``checkpoint_path`` mechanism in ``_run_bss_protocol``: when a checkpoint is
    supplied, ``continuation=yes`` and ``gen_vel=no`` are injected automatically
    so mdrun reads velocities from the checkpoint instead of regenerating them.

Checkpoint continuity
---------------------
BSS does not carry velocities between stages.  ``process.getSystem()`` returns a
Sire system built from the final GROMACS GRO file, which BSS writes with
coordinates only (no velocity columns).  This means that without intervention
every stage starts at T≈0 K (``continuation=yes, gen_vel=no``) or with freshly
drawn Maxwell-Boltzmann velocities (``gen_vel=yes``) that can land on bad
contacts and trigger SETTLE crashes.

The correct approach is to pass the GROMACS checkpoint file (``gromacs.cpt``)
written by each stage to the next stage via ``gmx mdrun -cpi``.  The checkpoint
carries the exact velocity state at the end of the previous stage.  When
``checkpoint_path`` is supplied to ``run_npt_equilibration`` or
``run_production``, ``_run_bss_protocol`` passes it to ``BSS.Process.Gromacs``
as ``checkpoint_file``, which makes BSS add ``-t <checkpoint>`` to the internal
``gmx grompp`` call.  This embeds the checkpoint velocities into the TPR so
mdrun starts with the correct velocity state without needing ``-cpi``.  Using
``grompp -t`` instead of ``mdrun -cpi`` is essential when the integrator changes
between stages (NVT ``sd`` to NPT ``md``): ``-cpi`` causes GROMACS to abort
with "Cannot change integrator during a checkpoint restart".

``run_heating`` does not accept a ``checkpoint_path`` because NVT heating always
starts from a minimized structure with no prior velocities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import BioSimSpace as BSS

from gbsa_pipeline.change_defaults import GromacsParams
from gbsa_pipeline.change_params import set_mdp_key
from gbsa_pipeline.md_diagnostics import analyze_crash_frames, check_posre_consistency

if TYPE_CHECKING:
    from collections.abc import Mapping

    import sire

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validated default MDP overrides
# ---------------------------------------------------------------------------

# These dicts contain only the parameters that BSS.Protocol.Equilibration and
# BSS.Protocol.Production do NOT generate correctly by default for a solvated
# protein-ligand system.  Parameters that BSS already handles well (barostat
# type, pressure reference, trajectory output cadence, etc.) are intentionally
# absent so that BSS remains the authoritative source for those settings.

_HEATING_STABILITY_PARAMS: dict[str, Any] = {
    # ------------------------------------------------------------------
    # Multiple time-stepping: disabled
    # ------------------------------------------------------------------
    # BSS.Protocol.Equilibration enables mts = yes in the generated MDP.
    # GROMACS requires integrator = md for mts; disable it to keep the
    # MDP valid regardless of which integrator the caller selects.
    "mts": "no",
    # ------------------------------------------------------------------
    # LINCS warning angle: 90° instead of the GROMACS default 30°
    # ------------------------------------------------------------------
    # BSS emits lincs-warnangle = 30.  For a structure that has residual LJ
    # stress after minimisation (ligand force-field parameters, tight pockets),
    # individual bonds can rotate more than 30° in the first picoseconds of
    # heating.  The GROMACS manual explicitly recommends 90° for strained
    # starting structures.
    "lincs_warnangle": 90.0,
    # h-bonds constraints: GROMACS standard for protein MD; allows a 2 fs
    # timestep in subsequent NPT stages and prevents the highest-frequency
    # O-H and N-H vibrations from dominating the timestep limit.
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "lincs_order": 4,
}

_NPT_STABILITY_PARAMS: dict[str, Any] = {
    # ------------------------------------------------------------------
    # Timestep: 2 fs
    # ------------------------------------------------------------------
    # BSS.Protocol.Equilibration defaults to the timestep supplied by the
    # caller (or 1 fs in run_npt_equilibration below).  2 fs is the
    # standard production timestep for protein MD with h-bonds constraints
    # and halves the wall-clock time of the NPT and production stages
    # relative to 1 fs.  The h-bonds constraints below make 2 fs stable.
    "dt": 0.002,
    # Same LINCS rationale as the NVT stage: residual stress in a
    # solvated complex can cause bond rotations > 30° in the first NPT
    # steps before the barostat has equilibrated the box.
    "lincs_warnangle": 90.0,
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "lincs_order": 4,
}

_SOLVENT_RELAX_PARAMS: dict[str, Any] = {
    # sd integrator: damps velocity spikes in waters that were placed too
    # close to protein atoms by the solvation algorithm.  Heavy-atom
    # restraints keep the protein/ligand fixed while waters and ions find
    # their equilibrium positions before NVT heating begins.
    "integrator": "sd",
    "dt": 0.001,
    "tcoupl": "no",
    "mts": "no",
    "lincs_warnangle": 90.0,
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "lincs_order": 4,
    "gen_vel": "yes",
    "gen_temp": 300.0,
}


def _gromacs_log_finished(work_dir: Path) -> bool:
    """Return True if the GROMACS log contains the 'Finished mdrun' marker."""
    log_path = work_dir / "gromacs.log"
    if not log_path.exists():
        return False
    return "Finished mdrun" in log_path.read_text(encoding="utf-8", errors="replace")


def _tail_text_file(path: Path, max_lines: int = 80) -> str:
    """Return the final lines of a process text file.

    GROMACS failures are usually diagnosed from the end of ``gromacs.out``,
    ``gromacs.log``, or the generated ``.mdp`` file. This helper keeps the
    failure-report formatting local to the MD helper module and avoids adding a
    broader logging abstraction. Missing files are represented explicitly so a
    failed BioSimSpace setup still gives useful context. Files are decoded with
    replacement enabled because external tools can occasionally write unusual
    characters to log files.
    """
    if not path.exists():
        return f"<missing: {path}>"

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if len(lines) <= max_lines:
        return text

    return "\n".join(lines[-max_lines:])


def _format_gromacs_failure_report(work_dir: Path | None, stage_name: str) -> str:
    """Build a compact diagnostic report for a failed GROMACS process.

    BioSimSpace can return ``None`` from ``getSystem`` after an ``mdrun``
    failure, which otherwise makes callers fail later with an unhelpful
    assertion error. This report includes the working directory and the tails of
    the most relevant files produced by the process. It intentionally does not
    try to parse or classify the chemistry problem because this helper only
    owns process execution. The goal is to expose the original GROMACS failure
    quickly enough for the integration test to be actionable.
    """
    if work_dir is None:
        return (
            f"{stage_name} finished without a readable output system. "
            "No work_dir was supplied, so no process logs can be summarized."
        )

    report_parts = [
        f"{stage_name} finished without a readable output system.",
        f"Work directory: {work_dir}",
    ]

    for filename in (
        "gromacs.out",
        "gromacs.log",
        "gromacs.err",
        "gromacs.mdp",
        "gromacs.out.mdp",
    ):
        path = work_dir / filename
        if path.exists():
            report_parts.append(f"\n--- tail: {path} ---\n{_tail_text_file(path)}")

    lincs_pdbs = sorted(work_dir.glob("step*.pdb"))
    if lincs_pdbs:
        report_parts.append("\nLINCS diagnostic PDB files:")
        report_parts.extend(str(path) for path in lincs_pdbs[-10:])
        crash_report = analyze_crash_frames(work_dir)
        if crash_report:
            report_parts.append("\n--- crash frame analysis ---")
            report_parts.append(crash_report)

    return "\n".join(report_parts)


def _normalise_mdp_key(key: str) -> str:
    """Return a comparison-safe GROMACS MDP key.

    BioSimSpace and local parameter models do not always write MDP keys with
    identical spelling. For example, one source can write ``DispCorr`` while
    another writes ``dispcorr``; similarly, Python-facing code may use
    underscores while GROMACS normally uses hyphens. GROMACS treats those as the
    same logical parameter often enough that exact string matching is unsafe for
    config merging. This helper normalises only for comparison and does not
    decide how the final key should be written.
    """
    return key.strip().lower().replace("_", "-")


def _mdp_key_from_line(line: str) -> str | None:
    """Extract a normalised MDP key from one config line.

    GROMACS MDP files use ``key = value`` assignments and may include comments.
    This parser is intentionally small because it only needs to identify
    existing assignment lines before local overrides are applied. Blank lines,
    comment-only lines, and non-assignment lines are left untouched by returning
    ``None``. Inline comments are ignored so a line such as ``dt = 0.001 ;
    timestep`` is still recognised as the ``dt`` key.
    """
    body = line.split(";", 1)[0].strip()

    if not body or "=" not in body:
        return None

    key, _value = body.split("=", 1)
    return _normalise_mdp_key(key)


def _remove_existing_mdp_key(config: list[str], key: str) -> list[str]:
    """Remove existing assignments for one MDP key.

    This keeps BioSimSpace-generated configs valid when explicit overrides are
    applied afterwards. ``set_mdp_key`` can update exact key matches, but it may
    not catch spelling variants such as ``DispCorr`` versus ``dispcorr`` or
    underscore versus hyphen forms. Removing all equivalent assignments first
    guarantees that the later write produces a single final value for the key.
    Comments and unrelated config lines are preserved.
    """
    target_key = _normalise_mdp_key(key)

    return [line for line in config if _mdp_key_from_line(line) != target_key]


def _apply_gromacs_params_to_config(
    config: list[str],
    params: GromacsParams | Mapping[str, Any],
) -> list[str]:
    """Apply validated GROMACS parameters to a generated BioSimSpace config.

    BioSimSpace generates the initial MDP from the selected protocol object,
    which keeps stage-specific setup such as restraint topology generation tied
    to the normal BSS code path. This helper then updates or appends individual
    MDP keys using the serialized ``GromacsParams`` mapping. Before each key is
    written, any existing equivalent assignment is removed using a
    case-insensitive and hyphen/underscore-tolerant comparison. This prevents
    GROMACS errors such as ``Parameter "dispcorr" doubly defined`` when BSS and
    the local parameter model spell the same MDP key differently.
    """
    final_params: GromacsParams
    final_params = params if isinstance(params, GromacsParams) else GromacsParams.from_mapping(params)

    updated_config = list(config)

    for key, value in final_params.to_mapping().items():
        updated_config = _remove_existing_mdp_key(updated_config, key)
        set_mdp_key(updated_config, key, value, inplace=True)

    return updated_config


def _check_stage_posre(work_dir: Path, stage_name: str) -> None:
    """Run the position-restraint consistency check for a stage work directory.

    BSS generates ``posre_*.itp`` files during process setup.  This helper
    finds the first one and validates that every restrained atom maps to a
    backbone heavy atom in the stage GRO.  Results are logged at DEBUG level
    when the check passes and at WARNING level when unexpected atoms are found.
    The check is advisory: it does not abort the stage, only logs.
    """
    gro = work_dir / "gromacs.gro"
    posre_files = sorted(work_dir.glob("posre_*.itp"))
    if not posre_files:
        return

    for posre_path in posre_files:
        result = check_posre_consistency(gro, posre_path)
        if not result.ok:
            logger.warning(
                "%s: posre validation FAILED for %s -- "
                "%d unexpected restrained atoms (first 5: %s). "
                "Check that the restraint file matches the GRO atom order.",
                stage_name,
                posre_path.name,
                len(result.unexpected),
                result.unexpected[:5],
            )
        else:
            logger.debug(
                "%s: posre validation OK -- %d backbone atoms restrained in %s.",
                stage_name,
                result.n_restrained,
                posre_path.name,
            )


_GRO_WATER_RESNAMES = {"SOL", "HOH", "WAT", "TIP3", "TIP3P"}


def _parse_gro_atom_line(
    line: str,
) -> tuple[int, str, str, int, tuple[float, float, float]]:
    """Parse the fixed-width fields needed from one GROMACS GRO atom line.

    GRO atom lines are fixed-width records with residue number, residue name,
    atom name, atom number, and coordinates in nanometres. The cleanup helper
    only needs those fields, so velocities and other trailing content are left
    untouched when the file is rewritten. A small local parser is used instead
    of a structural dependency because this helper must keep the GROMACS
    topology and coordinate files in lockstep. Parsing errors are raised with
    the original line content so malformed intermediate files fail explicitly.
    """
    try:
        residue_number = int(line[0:5])
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        atom_number = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse GRO atom line: {line!r}") from exc

    return residue_number, residue_name, atom_name, atom_number, (x, y, z)


def _water_residue_key(line: str, water_resnames: set[str]) -> tuple[int, str] | None:
    """Return a residue key for solvent water atom lines.

    The key combines residue number and residue name because GROMACS residue
    numbers can wrap in very large systems. This integration workflow has far
    fewer than 100000 atoms, but keeping the residue name in the key makes the
    helper more defensive. Only common three-site water residue names are
    treated as removable solvent. Protein, ligand, ion, and unknown residues are
    deliberately excluded so the cleanup cannot remove chemically meaningful
    solute atoms.
    """
    residue_number, residue_name, _atom_name, _atom_number, _xyz = _parse_gro_atom_line(line)
    if residue_name in water_resnames:
        return (residue_number, residue_name)
    return None


def _find_clashing_water_residues(
    atom_lines: list[str],
    cutoff_nm: float,
    water_resnames: set[str],
) -> set[tuple[int, str]]:
    """Identify whole water residues with impossible solute contacts.

    A water residue is flagged when any of its atoms is closer than
    ``cutoff_nm`` to any non-water atom. The distance check uses a simple
    spatial grid over non-water atoms, which keeps the operation cheap for the
    20k-atom integration-test systems while avoiding an O(Nwater*Nsolute)
    full pair loop. The helper intentionally uses atom-atom distances rather
    than only oxygen-heavy-atom distances because the observed SETTLE failures
    involved water hydrogens and solute hydrogens/heavy atoms. The returned set
    contains whole water residue keys so all atoms of each bad water molecule
    are removed together.
    """
    if cutoff_nm <= 0:
        raise ValueError("cutoff_nm must be positive.")

    cutoff2 = cutoff_nm * cutoff_nm
    cell_size = cutoff_nm
    grid: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    water_atoms: list[tuple[tuple[int, str], tuple[float, float, float]]] = []

    def cell_for(coords: tuple[float, float, float]) -> tuple[int, int, int]:
        return (
            int(coords[0] // cell_size),
            int(coords[1] // cell_size),
            int(coords[2] // cell_size),
        )

    for line in atom_lines:
        residue_number, residue_name, _atom_name, _atom_number, coords = _parse_gro_atom_line(line)
        if residue_name in water_resnames:
            water_atoms.append(((residue_number, residue_name), coords))
            continue
        grid.setdefault(cell_for(coords), []).append(coords)

    clashing: set[tuple[int, str]] = set()
    neighbour_offsets = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))

    for water_key, coords in water_atoms:
        if water_key in clashing:
            continue
        cx, cy, cz = cell_for(coords)
        wx, wy, wz = coords
        for dx, dy, dz in neighbour_offsets:
            for sx, sy, sz in grid.get((cx + dx, cy + dy, cz + dz), ()):
                ddx = wx - sx
                ddy = wy - sy
                ddz = wz - sz
                if (ddx * ddx) + (ddy * ddy) + (ddz * ddz) < cutoff2:
                    clashing.add(water_key)
                    break
            if water_key in clashing:
                break

    return clashing


def _renumber_gro_atom_line(line: str, atom_number: int) -> str:
    """Return a GRO atom line with an updated five-column atom number."""
    if atom_number <= 0:
        raise ValueError("atom_number must be positive.")
    if atom_number > 99999:  # noqa: PLR2004
        atom_number = atom_number % 100000
        if atom_number == 0:
            atom_number = 99999
    return f"{line[:15]}{atom_number:5d}{line[20:]}"


def _write_cleaned_gro(
    input_gro: Path,
    output_gro: Path,
    cutoff_nm: float,
    water_resnames: set[str],
) -> dict[str, int]:
    """Write a GRO file with clashing solvent waters removed.

    The input file is expected to be a normal GROMACS coordinate file with a
    title line, atom-count line, atom records, and one final box line. Whole
    solvent residues are removed when any atom of that water molecule is closer
    than the requested cutoff to any non-water atom. The atom count and atom
    serial numbers are updated in the output file. The returned dictionary maps
    water residue names to the number of removed molecules, which is later used
    to update the matching GROMACS topology.
    """
    lines = input_gro.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:  # noqa: PLR2004
        raise ValueError(f"GRO file is too short: {input_gro}")

    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"Could not read GRO atom count from {input_gro}") from exc

    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"GRO file ended before all atom records were read: {input_gro}")

    box_line = lines[2 + atom_count]
    clashing_waters = _find_clashing_water_residues(atom_lines, cutoff_nm, water_resnames)

    cleaned_atom_lines: list[str] = []
    for line in atom_lines:
        key = _water_residue_key(line, water_resnames)
        if key is not None and key in clashing_waters:
            continue
        cleaned_atom_lines.append(line)

    removed_counts: dict[str, int] = {}
    for _resnr, resname in clashing_waters:
        removed_counts[resname] = removed_counts.get(resname, 0) + 1

    output_lines = [lines[0], f"{len(cleaned_atom_lines):5d}"]
    output_lines.extend(_renumber_gro_atom_line(line, i) for i, line in enumerate(cleaned_atom_lines, start=1))
    output_lines.append(box_line)
    output_gro.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    return removed_counts


def _update_topology_water_counts(
    input_top: Path,
    output_top: Path,
    removed_counts: dict[str, int],
) -> None:
    """Write a topology with [ molecules ] water counts reduced.

    GROMACS requires the topology molecule counts to match the coordinate file.
    Removing water coordinates without changing the ``[ molecules ]`` section
    would create a broken system, so this helper updates the first matching
    water molecule count for each removed residue name. The function is strict:
    if waters were removed but no matching topology molecule count can be
    updated, it raises an error instead of writing an inconsistent topology. The
    parser intentionally only touches the ``[ molecules ]`` section and keeps
    includes, force-field definitions, comments, and all other molecule counts
    unchanged.
    """
    if not removed_counts:
        output_top.write_text(input_top.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        return

    lines = input_top.read_text(encoding="utf-8", errors="replace").splitlines()
    in_molecules = False
    remaining = dict(removed_counts)
    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section_name = stripped.strip("[]").strip().lower()
            in_molecules = section_name == "molecules"
            output_lines.append(line)
            continue

        if not in_molecules or not stripped or stripped.startswith((";", "#")):
            output_lines.append(line)
            continue

        body, *comment_parts = line.split(";", 1)
        comment = ";" + comment_parts[0] if comment_parts else ""
        fields = body.split()
        if len(fields) < 2:  # noqa: PLR2004
            output_lines.append(line)
            continue

        molecule_name = fields[0]
        if molecule_name not in remaining:
            output_lines.append(line)
            continue

        try:
            old_count = int(fields[1])
        except ValueError as exc:
            raise ValueError(f"Could not parse molecule count in topology line: {line!r}") from exc

        new_count = old_count - remaining[molecule_name]
        if new_count < 0:
            raise ValueError(
                f"Cannot remove {remaining[molecule_name]} {molecule_name} waters from topology count {old_count}."
            )

        prefix = line[: line.find(molecule_name)] if molecule_name in line else ""
        output_lines.append(f"{prefix}{molecule_name:<16} {new_count}{(' ' + comment) if comment else ''}".rstrip())
        del remaining[molecule_name]

    if remaining:
        raise ValueError(
            "Removed solvent waters but could not update matching topology molecule counts: "
            + ", ".join(f"{name}={count}" for name, count in sorted(remaining.items()))
        )

    output_top.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def remove_clashing_solvent_waters(
    system: sire.System,
    *,
    work_dir: Path,
    cutoff_angstrom: float = 1.5,
    water_resnames: tuple[str, ...] = ("SOL", "HOH", "WAT", "TIP3", "TIP3P"),
) -> sire.System:
    """Remove impossible bulk-solvent contacts from a solvated GROMACS system.

    This is a defensive cleanup for automated protein-ligand workflows before
    NVT heating. Solvation and minimization usually remove bad solvent contacts,
    but some systems can retain a single bulk water with an impossible contact
    to the protein or ligand. During constrained dynamics, SETTLE may then fail
    and write a crash frame where the offending water atoms become NaN. This
    helper saves the current BSS/Sire system as GRO/TOP, removes whole solvent
    water molecules whose atoms are closer than ``cutoff_angstrom`` to any
    non-water atom, updates the topology molecule counts, reloads the cleaned
    system, and writes a small report in ``work_dir``.

    The cleanup intentionally targets only common bulk water residue names. It
    does not remove protein atoms, ligand atoms, ions, metals, cofactors, or any
    residue with an unknown name. The cutoff is deliberately conservative for
    impossible contacts, not for normal hydration-shell pruning; a value around
    1.4--1.5 A removes contacts like the observed 1.13 A water-oxygen to
    protein-carbon clash without deleting chemically reasonable waters. If no
    clashing waters are found, the saved GRO/TOP are still copied into the
    cleanup directory and the returned system is reloaded from those files.
    """
    if cutoff_angstrom <= 0:
        raise ValueError("cutoff_angstrom must be positive.")

    work_dir.mkdir(parents=True, exist_ok=True)
    input_prefix = work_dir / "input"
    output_gro = work_dir / "cleaned.gro"
    output_top = work_dir / "cleaned.top"

    saved = BSS.IO.saveMolecules(str(input_prefix), system, ["gro87", "grotop"])
    input_paths = [Path(path) for path in saved] if saved is not None else []
    input_gro = next(
        (path for path in input_paths if path.suffix == ".gro"),
        input_prefix.with_suffix(".gro"),
    )
    input_top = next(
        (path for path in input_paths if path.suffix == ".top"),
        input_prefix.with_suffix(".top"),
    )

    if not input_gro.exists() or not input_top.exists():
        raise RuntimeError(
            f"Could not save a GROMACS GRO/TOP pair for solvent-water cleanup: gro={input_gro}, top={input_top}"
        )

    cutoff_nm = cutoff_angstrom / 10.0
    resname_set = set(water_resnames)
    removed_counts = _write_cleaned_gro(input_gro, output_gro, cutoff_nm, resname_set)
    _update_topology_water_counts(input_top, output_top, removed_counts)

    total_removed = sum(removed_counts.values())
    report_lines = [
        "remove_clashing_solvent_waters",
        f"cutoff_angstrom = {cutoff_angstrom}",
        f"input_gro = {input_gro}",
        f"input_top = {input_top}",
        f"output_gro = {output_gro}",
        f"output_top = {output_top}",
        f"removed_waters_total = {total_removed}",
    ]
    for resname, count in sorted(removed_counts.items()):
        report_lines.append(f"removed_{resname} = {count}")
    (work_dir / "water_cleanup.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if total_removed:
        logger.info(
            "Removed %d clashing solvent waters before MD heating; details written to %s.",
            total_removed,
            work_dir / "water_cleanup.txt",
        )
    else:
        logger.info("No clashing solvent waters found before MD heating.")

    return BSS.IO.readMolecules([str(output_gro), str(output_top)])


def _run_bss_protocol(
    system: sire.System,
    protocol: Any,
    work_dir: Path | None = None,
    params: GromacsParams | Mapping[str, Any] | None = None,
    *,
    ignore_warnings: bool = True,
    stage_name: str = "GROMACS stage",
    max_time: int | None = None,
    checkpoint_path: Path | None = None,
) -> sire.System:
    """Run a BioSimSpace GROMACS protocol with optional MDP overrides.

    The process is created from a normal BioSimSpace protocol first, rather than
    from a standalone custom MDP. This matters for restrained stages because BSS
    can prepare the topology and reference-coordinate machinery associated with
    the protocol. If ``params`` is provided, the generated MDP config is then
    modified key-by-key and passed back into the process before execution. A
    failed process raises a diagnostic error with log tails instead of returning
    ``None`` to the caller.

    When ``checkpoint_path`` points to a GROMACS ``.cpt`` file from the previous
    stage, it is passed to ``BSS.Process.Gromacs`` as ``checkpoint_file``.  BSS
    adds ``-t <checkpoint>`` to its internal ``gmx grompp`` call, which embeds
    the checkpoint velocities directly into the TPR.  mdrun then starts with
    those velocities without needing ``-cpi``.

    This approach (``grompp -t``) is correct for cross-stage velocity continuity
    where the integrator changes between stages (e.g. NVT ``sd`` to NPT ``md``).
    The alternative, ``mdrun -cpi``, locks the integrator and GROMACS refuses to
    start with "Cannot change integrator during a checkpoint restart".  The MDP
    is overridden with ``continuation = yes`` and ``gen_vel = no`` so grompp
    uses the checkpoint velocities rather than regenerating them.  See the module
    docstring for the full explanation of the BSS checkpoint gap.
    """
    # Use 1 MPI rank + all available OpenMP threads so mdrun fully uses the
    # machine's CPU cores without spawning multiple MPI processes that would
    # each try to own the GPU/memory bus.
    # -cpt 1: write a checkpoint every 1 minute of wall time.  BSS terminates
    # mdrun when the simulated time matches the protocol runtime; GROMACS may
    # not reach the default 15-minute checkpoint interval before BSS kills it,
    # leaving no checkpoint file for the next stage to read velocities from.
    # Writing every minute ensures a recent checkpoint always exists.
    extra_args: dict[str, str] = {"-ntmpi": "1", "-ntomp": "12", "-cpt": "1"}

    kwargs: dict[str, object] = {
        "ignore_warnings": ignore_warnings,
        "extra_args": extra_args,
    }
    if work_dir:
        kwargs["work_dir"] = str(work_dir)

    # When a checkpoint is supplied, pass it to BSS as checkpoint_file so BSS
    # adds -t <checkpoint> to its internal grompp call.  grompp embeds the
    # checkpoint velocities directly into the TPR; mdrun then starts with those
    # velocities without needing -cpi.
    #
    # This is the correct approach for cross-stage velocity continuity:
    # -cpi (mdrun restart) locks the integrator -- GROMACS refuses to switch from
    # sd (NVT) to md (NPT) with "Cannot change integrator during a checkpoint
    # restart".  grompp -t has no such restriction and is designed exactly for
    # reading velocities when starting a new simulation from a previous state.
    if checkpoint_path is not None:
        kwargs["checkpoint_file"] = str(checkpoint_path)

    process = BSS.Process.Gromacs(
        protocol=protocol,
        system=system,
        **kwargs,
    )

    # Apply caller-supplied MDP overrides first.  None is treated as "no
    # overrides", so each public stage function substitutes the validated
    # module-level default before reaching this point.
    if params is not None:
        config = _apply_gromacs_params_to_config(
            config=list(process.getConfig()),
            params=params,
        )
        process.setConfig(config)

    # When a checkpoint is supplied, force continuation=yes and gen_vel=no
    # AFTER any caller overrides so the checkpoint always wins.
    #
    # Do NOT use _apply_gromacs_params_to_config here: that function goes through
    # GromacsParams.from_mapping → to_mapping, which emits ALL model defaults
    # including nsteps=500 (the GromacsParams default).  Those defaults would
    # overwrite any nsteps the caller set in the first setConfig call.  Instead,
    # remove the two keys directly and append the fixed values so only those two
    # MDP lines change and everything else (including nsteps/dt) is preserved.
    if checkpoint_path is not None:
        config = list(process.getConfig())
        for _key in ("continuation", "gen-vel"):
            config = _remove_existing_mdp_key(config, _key)
        set_mdp_key(config, "continuation", "yes", inplace=True)
        set_mdp_key(config, "gen-vel", "no", inplace=True)
        process.setConfig(config)

    # Validate position restraints before mdrun when the stage uses them.
    # BSS writes posre_*.itp during setup; the indices must map to backbone
    # atoms in the starting GRO to avoid restraining the wrong atoms.
    if work_dir is not None:
        _check_stage_posre(work_dir, stage_name)

    process.start()
    process.wait(max_time=max_time)

    result = process.getSystem(block=True)
    if result is None:
        raise RuntimeError(_format_gromacs_failure_report(work_dir, stage_name))

    if work_dir is not None:
        if _gromacs_log_finished(work_dir):
            (work_dir / "success.txt").write_text("", encoding="utf-8")
        else:
            logger.warning(
                "%s: 'Finished mdrun' not found in gromacs.log -- run may have been aborted or log is missing.",
                stage_name,
            )

    return result


def run_minimization(
    system: sire.System,
    work_dir: Path | None = None,
    params: GromacsParams | Mapping[str, Any] | None = None,
    *,
    ignore_warnings: bool = True,
    max_time: int | None = None,
) -> sire.System:
    """Run a BioSimSpace energy minimization.

    This helper is the first small MD-stage building block and only handles
    minimization of an already prepared BioSimSpace/Sire system. The input
    ``system`` is expected to be parametrized already, because this function
    does not assign force-field parameters, solvate, neutralize, or prepare
    topology files. A normal ``BSS.Protocol.Minimisation`` object is always
    created first so BioSimSpace owns the base GROMACS setup. When ``params`` is
    provided, the generated MDP config is modified before execution so staged
    workflows can still request explicit steepest-descent or conjugate-gradient
    settings.
    """
    minimization_protocol = BSS.Protocol.Minimisation()

    return _run_bss_protocol(
        system=system,
        protocol=minimization_protocol,
        work_dir=work_dir,
        params=params,
        ignore_warnings=ignore_warnings,
        stage_name="GROMACS minimization",
        max_time=max_time,
    )


def run_solvent_relaxation(
    system: sire.System,
    work_dir: Path | None = None,
    params: GromacsParams | Mapping[str, Any] | None = None,
    simulation_time: BSS.Types.Time | None = None,
    *,
    ignore_warnings: bool = True,
    max_time: int | None = None,
) -> sire.System:
    """Run a short NVT with all heavy atoms restrained to relax the solvent.

    BSS solvation can place water molecules within van der Waals overlap of
    protein side-chain atoms.  Energy minimisation removes the worst clashes
    but does not always give waters enough freedom to escape tight pockets.
    Running a short NVT (default 20 ps) with all protein and ligand heavy
    atoms restrained lets waters and ions find their equilibrium positions
    before the full NVT heating ramp starts, preventing SETTLE failures that
    originate from bad protein-water contacts.

    When ``params`` is ``None`` the module-level ``_SOLVENT_RELAX_PARAMS``
    overrides are applied (sd integrator, 1 fs timestep, no annealing,
    T = 300 K, h-bonds constraints, lincs_warnangle = 90°).
    """
    runtime = simulation_time if simulation_time is not None else 20 * BSS.Units.Time.picosecond
    protocol = BSS.Protocol.Equilibration(
        timestep=1 * BSS.Units.Time.femtosecond,
        runtime=runtime,
        temperature=300 * BSS.Units.Temperature.kelvin,
        restraint="heavy",
    )
    return _run_bss_protocol(
        system=system,
        protocol=protocol,
        work_dir=work_dir,
        params=params if params is not None else _SOLVENT_RELAX_PARAMS,
        ignore_warnings=ignore_warnings,
        stage_name="GROMACS solvent relaxation",
        max_time=max_time,
    )


def run_heating(
    simulation_time: BSS.Types.Time,
    minimized: sire.System,
    work_dir: Path | None = None,
    params: GromacsParams | Mapping[str, Any] | None = None,
    temperature_start: BSS.Types.Temperature = 50 * BSS.Units.Temperature.kelvin,
    temperature_end: BSS.Types.Temperature = 300 * BSS.Units.Temperature.kelvin,
    restraint: str | None = "backbone",
    *,
    ignore_warnings: bool = True,
    max_time: int | None = None,
) -> sire.System:
    """Run a BioSimSpace/GROMACS restrained or unrestrained NVT stage.

    BioSimSpace creates the base Equilibration protocol including restraint
    topology files.  When ``params`` is ``None`` the module-level
    ``_HEATING_STABILITY_PARAMS`` overrides are applied to the generated MDP:
    they switch the integrator to Langevin (``sd``), disable MTS, and relax
    the LINCS warning angle.  See the module docstring for the full rationale.

    When ``params`` is supplied the caller owns the complete MDP configuration.
    In that case BSS uses a single target temperature (``temperature_end``)
    rather than a linear ramp, so any annealing schedule in the caller's params
    is not overridden by a BSS-generated one.

    The ``restraint`` argument controls backbone position restraints and is
    honoured regardless of which params path is taken.
    """
    if params is None:
        # Use BSS temperature-ramp Equilibration so that BSS generates an
        # annealing schedule scaled to simulation_time automatically.  The
        # _HEATING_STABILITY_PARAMS overrides (integrator=sd, mts=no, etc.)
        # are then applied on top without touching the annealing timing.
        heating_protocol = BSS.Protocol.Equilibration(
            timestep=1 * BSS.Units.Time.femtosecond,
            runtime=simulation_time,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            restraint=restraint,
        )
        effective_params: GromacsParams | Mapping[str, Any] | None = _HEATING_STABILITY_PARAMS
    else:
        # Caller supplies a full params dict which may include an explicit
        # annealing schedule.  Use single-temperature BSS protocol to avoid
        # BSS generating an annealing schedule that would conflict with it.
        heating_protocol = BSS.Protocol.Equilibration(
            timestep=1 * BSS.Units.Time.femtosecond,
            runtime=simulation_time,
            temperature=temperature_end,
            restraint=restraint,
        )
        effective_params = params

    return _run_bss_protocol(
        system=minimized,
        protocol=heating_protocol,
        work_dir=work_dir,
        params=effective_params,
        ignore_warnings=ignore_warnings,
        stage_name="GROMACS NVT heating",
        max_time=max_time,
    )


def run_npt_equilibration(
    simulation_time: BSS.Types.Time,
    heated: sire.System,
    work_dir: Path | None = None,
    params: GromacsParams | Mapping[str, Any] | None = None,
    restraint: str | None = "backbone",
    *,
    ignore_warnings: bool = True,
    max_time: int | None = None,
    checkpoint_path: Path | None = None,
) -> sire.System:
    """Run a BioSimSpace/GROMACS NPT equilibration procedure.

    BioSimSpace creates the base NPT Equilibration protocol, including position
    restraint files when ``restraint`` is set.  When ``params`` is ``None`` the
    module-level ``_NPT_STABILITY_PARAMS`` overrides are applied: a 2 fs
    timestep and the relaxed LINCS warning angle.

    When ``checkpoint_path`` is supplied it must point to the ``gromacs.cpt``
    file written by the preceding stage (NVT for restrained NPT, restrained NPT
    for unrestrained NPT).  The checkpoint is passed to ``gmx mdrun`` via
    ``-cpi`` and the MDP is forced to ``continuation = yes, gen_vel = no`` so
    that mdrun uses the checkpoint velocities rather than generating new ones.
    This avoids the T≈0 K start (BSS drops velocities) and the SETTLE crashes
    that can result from freshly drawn Maxwell-Boltzmann velocities landing on
    residual bad contacts.

    This function is used for both restrained and unrestrained NPT; the
    ``restraint`` argument selects the appropriate topology and MDP define flag
    via BSS's normal restraint machinery.
    """
    equilibration_protocol = BSS.Protocol.Equilibration(
        timestep=1 * BSS.Units.Time.femtosecond,
        runtime=simulation_time,
        temperature=300 * BSS.Units.Temperature.kelvin,
        pressure=1 * BSS.Units.Pressure.atm,
        restraint=restraint,
    )

    return _run_bss_protocol(
        system=heated,
        protocol=equilibration_protocol,
        work_dir=work_dir,
        params=params if params is not None else _NPT_STABILITY_PARAMS,
        ignore_warnings=ignore_warnings,
        stage_name="GROMACS NPT equilibration",
        max_time=max_time,
        checkpoint_path=checkpoint_path,
    )


def run_production(
    simulation_time: BSS.Types.Time,
    equilibrated: sire.System,
    work_dir: Path | None = None,
    params: GromacsParams | Mapping[str, Any] | None = None,
    *,
    ignore_warnings: bool = True,
    max_time: int | None = None,
    checkpoint_path: Path | None = None,
) -> sire.System:
    """Run a BioSimSpace production MD procedure.

    BioSimSpace creates the base Production protocol.  When ``params`` is
    ``None`` the module-level ``_NPT_STABILITY_PARAMS`` overrides are applied
    (2 fs timestep, relaxed LINCS warning angle).

    When ``checkpoint_path`` is supplied it must point to the ``gromacs.cpt``
    file from the final NPT equilibration stage.  The checkpoint is passed to
    ``gmx mdrun`` via ``-cpi`` and the MDP is forced to
    ``continuation = yes, gen_vel = no`` so that production starts with the
    exact velocities at the end of NPT rather than drawing new ones.  This is
    essential: freshly drawn Maxwell-Boltzmann velocities at 300 K can cause
    immediate SETTLE failures when the system still has residual contacts from
    equilibration (observed at step 568 in production when gen_vel=yes was used).
    """
    production_protocol = BSS.Protocol.Production(
        runtime=simulation_time,
        temperature=300 * BSS.Units.Temperature.kelvin,
        pressure=1 * BSS.Units.Pressure.atm,
    )

    return _run_bss_protocol(
        system=equilibrated,
        protocol=production_protocol,
        work_dir=work_dir,
        params=params if params is not None else _NPT_STABILITY_PARAMS,
        ignore_warnings=ignore_warnings,
        stage_name="GROMACS production",
        max_time=max_time,
        checkpoint_path=checkpoint_path,
    )
