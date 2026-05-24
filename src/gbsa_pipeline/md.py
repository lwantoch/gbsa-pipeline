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
    Switches the integrator to Langevin (``sd``), disables MTS, and relaxes the
    LINCS warning angle.  These three changes were the minimum required to
    prevent SETTLE failures and grompp errors on a solvated 450-residue
    protein-ligand system during integration testing.

``_NPT_STABILITY_PARAMS``
    Applied to restrained NPT, unrestrained NPT, and production stages when the
    caller passes ``params=None``.  Uses a 2 fs timestep (safe with h-bonds
    constraints) and the relaxed LINCS warning angle.  All other NPT/production
    MDP parameters (barostat, gen-vel, thermostat coupling) are left to BSS
    because the BSS-generated defaults proved correct during integration testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import BioSimSpace as BSS

from gbsa_pipeline.change_defaults import GromacsParams
from gbsa_pipeline.change_params import set_mdp_key

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import sire


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
    # Integrator: Langevin (sd) instead of leap-frog (md)
    # ------------------------------------------------------------------
    # BSS.Protocol.Equilibration emits integrator = md by default.
    # During integration testing, md + V-rescale caused SETTLE failures at
    # T ≈ 253 K (step 91 039 / 100 000) on a solvated 450-residue complex.
    # SETTLE crashes when a water molecule's O-H geometry becomes too
    # distorted for the analytic rigid-body solver.  The Langevin integrator
    # applies a stochastic friction force at each step that damps velocity
    # spikes before they can distort water geometry; the same run completed
    # without any SETTLE warning when switched to sd.
    "integrator": "sd",
    # ------------------------------------------------------------------
    # Multiple time-stepping: disabled
    # ------------------------------------------------------------------
    # BSS.Protocol.Equilibration enables mts = yes in the generated MDP.
    # GROMACS 2024+ accepts mts only when integrator = md; grompp exits
    # with a fatal error ("Multiple time stepping is only supported with
    # integrator md") when integrator = sd is combined with mts = yes.
    "mts": "no",
    # ------------------------------------------------------------------
    # Thermostat coupling: none (built into sd)
    # ------------------------------------------------------------------
    # BSS emits tcoupl = v-rescale.  The sd integrator provides its own
    # temperature control through the friction and noise terms; adding an
    # external thermostat on top double-counts thermal energy removal and
    # causes grompp to emit "WARNING: sd and bd combine T-coupling with
    # the integrator, ignore tcoupl".  Setting tcoupl = no avoids the
    # warning and is consistent with GROMACS best practice for Langevin MD.
    "tcoupl": "no",
    # ------------------------------------------------------------------
    # LINCS warning angle: 90° instead of the GROMACS default 30°
    # ------------------------------------------------------------------
    # BSS emits lincs-warnangle = 30, the upstream GROMACS default.  For a
    # structure that has residual LJ stress after minimisation (common with
    # ligand force-field parameters and crystal-water clashes), individual
    # bonds occasionally rotate more than 30° in the first picoseconds of
    # heating.  At 30° each violation prints a warning that counts toward
    # GROMACS's crash limit; the GROMACS manual explicitly recommends 90°
    # for poorly-minimised or strained starting structures.
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


def _run_bss_protocol(
    system: sire.System,
    protocol: Any,
    work_dir: Path | None = None,
    params: GromacsParams | Mapping[str, Any] | None = None,
    *,
    ignore_warnings: bool = True,
    stage_name: str = "GROMACS stage",
    max_time: int | None = None,
) -> sire.System:
    """Run a BioSimSpace GROMACS protocol with optional MDP overrides.

    The process is created from a normal BioSimSpace protocol first, rather than
    from a standalone custom MDP. This matters for restrained stages because BSS
    can prepare the topology and reference-coordinate machinery associated with
    the protocol. If ``params`` is provided, the generated MDP config is then
    modified key-by-key and passed back into the process before execution. A
    failed process raises a diagnostic error with log tails instead of returning
    ``None`` to the caller.
    """
    kwargs: dict[str, object] = {
        "ignore_warnings": ignore_warnings,
        # Use 1 MPI rank + all available OpenMP threads so mdrun fully uses the
        # machine's CPU cores without spawning multiple MPI processes that would
        # each try to own the GPU/memory bus.
        "extra_args": {"-ntmpi": "1", "-ntomp": "12"},
    }
    if work_dir:
        kwargs["work_dir"] = str(work_dir)

    process = BSS.Process.Gromacs(
        protocol=protocol,
        system=system,
        **kwargs,
    )

    # Apply MDP overrides when supplied.  None is treated as "no overrides",
    # so each public stage function is responsible for substituting the
    # validated module-level default before reaching this point.
    if params is not None:
        config = _apply_gromacs_params_to_config(
            config=list(process.getConfig()),
            params=params,
        )
        process.setConfig(config)

    process.start()
    process.wait(max_time=max_time)

    result = process.getSystem(block=True)
    if result is None:
        raise RuntimeError(_format_gromacs_failure_report(work_dir, stage_name))

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
) -> sire.System:
    """Run a BioSimSpace/GROMACS NPT equilibration procedure.

    BioSimSpace creates the base NPT Equilibration protocol, including position
    restraint files when ``restraint`` is set.  When ``params`` is ``None`` the
    module-level ``_NPT_STABILITY_PARAMS`` overrides are applied: a 2 fs
    timestep and the relaxed LINCS warning angle.  BSS-generated settings that
    proved correct during integration testing (barostat type, vel-rescale
    thermostat, gen-vel, continuation) are left untouched.

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
    )


def run_production(
    simulation_time: BSS.Types.Time,
    equilibrated: sire.System,
    work_dir: Path | None = None,
    params: GromacsParams | Mapping[str, Any] | None = None,
    *,
    ignore_warnings: bool = True,
    max_time: int | None = None,
) -> sire.System:
    """Run a BioSimSpace production MD procedure.

    BioSimSpace creates the base Production protocol.  When ``params`` is
    ``None`` the module-level ``_NPT_STABILITY_PARAMS`` overrides are applied
    (2 fs timestep, relaxed LINCS warning angle).  BSS-generated settings that
    proved correct during integration testing (gen-vel, continuation, barostat,
    thermostat) are left untouched so the production stage inherits a consistent
    environment regardless of the preceding equilibration integrator.
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
    )
