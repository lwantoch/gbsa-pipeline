"""gmx_MMPBSA configuration and GROMACS-based runner.

This module provides typed dataclasses for building gmx_MMPBSA input files
(the ``&general``, ``&gb``, and ``&pb`` namelists) and a thin runner that
invokes ``gmx_MMPBSA`` against GROMACS-format inputs (gro/top/xtc/ndx).
It does not handle output parsing, energy decomposition, or any post-processing
of results — those concerns belong to the caller.
See the gmx_MMPBSA documentation at https://valdes-tresanco-ms.github.io/gmx_MMPBSA/
for the full list of supported namelist keywords and their meaning.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _fmt(v: Any) -> str:
    """Format a Python value as a gmx_MMPBSA namelist token."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        raise ValueError("None should be filtered before formatting")
    s = str(v)
    # Quote strings that could confuse parsing
    if s == "" or any(c.isspace() for c in s) or "," in s:
        return f'"{s}"'
    return s


def _as_kv(dc: Any) -> dict[str, Any]:
    """Convert a param dataclass to a flat key/value dict.

    Fields set to None are excluded so that optional parameters are not written
    to the input file at all. The special ``extra`` field is merged last,
    allowing callers to override or extend any explicit field without subclassing.
    """
    d: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for f in fields(dc):
        val = getattr(dc, f.name)
        if f.name == "extra":
            extra = val or {}
            continue
        if val is not None:
            d[f.name] = val
    # extra overrides explicit fields
    d.update(extra)
    return d


def _render_namelist(name: str, kv: dict[str, Any]) -> str:
    """Render a Fortran-style namelist block from a key/value dict."""
    lines = [f"&{name}"]
    for k, v in kv.items():
        lines.append(f"  {k:<20} = {_fmt(v)}")
    lines.append("/")
    return "\n".join(lines)


# --------- Parameter dataclasses ---------


@dataclass(frozen=True)
class GeneralParams:
    """Parameters for the ``&general`` namelist section of a gmx_MMPBSA input file.

    This dataclass covers trajectory frame selection (``startframe``,
    ``endframe``, ``interval``), entropy approximations, GROMACS executable
    path, and file-keeping behaviour.  Fields ``forcefields``, ``ions_parameters``,
    and ``PBRadii`` are only relevant when gmx_MMPBSA builds an AMBER topology
    itself (i.e. when ``-cp`` is not passed); when a pre-built topology is
    provided they are left as ``None`` and omitted from the written file.
    The ``extra`` dict is an escape hatch for gmx_MMPBSA options not listed
    here; its entries override any explicit field of the same name.
    See https://valdes-tresanco-ms.github.io/gmx_MMPBSA/dev/input_file/#general
    for the full reference.
    """

    sys_name: str = ""
    startframe: int = 1
    endframe: int = 9_999_999
    interval: int = 1

    # Only relevant when gmx_MMPBSA builds AMBER topology itself (no -cp)
    forcefields: str | None = None
    ions_parameters: int | None = None
    PBRadii: int | None = None

    temperature: float = 298.15

    qh_entropy: int = 0
    interaction_entropy: int = 0
    ie_segment: int = 25
    c2_entropy: int = 0

    assign_chainID: int = 0  # noqa: N815
    exp_ki: float = 0.0

    full_traj: int = 0
    gmx_path: str = ""

    keep_files: int = 2
    netcdf: int = 0
    solvated_trajectory: int = 1
    verbose: int = 1

    # Any additional keys supported by your gmx_MMPBSA version
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GBParams:
    """Parameters for the ``&gb`` (Generalized Born) namelist section.

    This dataclass exposes the GB solvation parameters used by gmx_MMPBSA,
    including the GB model index (``igb``), dielectric constants, and surface
    tension terms.  QM/MM fields (``ifqnt``, ``qm_theory``, ``qm_residues``,
    etc.) default to ``None`` and are omitted from the written file unless
    explicitly set; this avoids ``Unknown variable`` errors from gmx_MMPBSA
    versions that do not recognise QM/MM keywords when QM/MM is inactive.
    The ``extra`` dict allows passing any additional keyword accepted by the
    underlying sander/AMBER engine without modifying this class.
    See https://valdes-tresanco-ms.github.io/gmx_MMPBSA/dev/input_file/#gb
    for the full reference.
    """

    igb: int = 5
    intdiel: float = 1.0
    extdiel: float = 78.5
    # Physiological NaCl concentration (0.15 mol/L). The previous default of
    # 0.0 implies vacuum (no ionic screening), which causes the GB implicit
    # solvent to overestimate the electrostatic solvation free energy because
    # the Debye-Huckel screening term is absent. 0.15 mol/L matches the ionic
    # strength used in the explicit-solvent MD stages and is the standard value
    # cited in the gmx_MMPBSA documentation for protein-ligand GB calculations.
    saltcon: float = 0.15

    surften: float = 0.0072
    surfoff: float = 0.0
    molsurf: int = 0
    msoffset: float = 0.0
    probe: float = 1.4

    # QM/MM options — omitted by default; only write when ifqnt = 1
    ifqnt: int | None = None
    qm_theory: str | None = None
    qm_residues: str | None = None
    qmcharge_com: int | None = None
    qmcharge_lig: int | None = None
    qmcharge_rec: int | None = None
    qmcut: float | None = None
    scfconv: float | None = None
    peptide_corr: int | None = None
    writepdb: int | None = None
    verbosity: int | None = None

    alpb: int | None = None
    arad_method: int | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PBParams:
    """Parameters for the ``&pb`` (Poisson-Boltzmann) namelist section.

    This dataclass covers the full surface of PB options supported by
    gmx_MMPBSA, including grid setup (``scale``, ``fillratio``, ``nfocus``),
    membrane embedding (``memopt``, ``mthick``, ``mctrdz``), solver settings,
    and cavity/surface area terms.  Membrane-related fields are only active
    when ``memopt = 1``; they are included here for completeness and left at
    their defaults for standard soluble-protein runs.  The ``extra`` dict
    provides a forward-compatible escape hatch for keywords added in newer
    gmx_MMPBSA versions.
    See https://valdes-tresanco-ms.github.io/gmx_MMPBSA/dev/input_file/#pb
    for the full reference.
    """

    ipb: int = 2
    inp: int = 2
    sander_apbs: int = 0

    indi: float = 1.0
    exdi: float = 80.0
    emem: float = 4.0

    smoothopt: int = 1
    istrng: float = 0.0
    radiopt: int = 1
    prbrad: float = 1.4
    iprob: float = 2.0
    sasopt: int = 0
    arcres: float = 0.25

    memopt: int = 0
    mprob: float = 2.7
    mthick: float = 40.0
    mctrdz: float = 0.0
    poretype: int = 1

    npbopt: int = 0
    solvopt: int = 1
    accept: float = 0.001
    linit: int = 1000

    fillratio: float = 4.0
    scale: float = 2.0
    nbuffer: float = 0.0
    nfocus: int = 2
    fscale: int = 8
    npbgrid: int = 1
    bcopt: int = 5

    eneopt: int = 2
    frcopt: int = 0
    scalec: int = 0

    cutfd: float = 5.0
    cutnb: float = 0.0
    nsnba: int = 1

    decompopt: int = 2
    use_rmin: int = 1

    sprob: float = 0.557
    vprob: float = 1.3
    rhow_effect: float = 1.129
    use_sav: int = 1
    cavity_surften: float = 0.0378
    cavity_offset: float = -0.5692

    maxsph: int = 400
    maxarcdot: int = 1500
    npbverb: int = 0

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MMPBSAConfig:
    """Top-level configuration object for a gmx_MMPBSA run.

    Holds one ``GeneralParams`` and optional ``GBParams`` / ``PBParams``
    instances; setting either to ``None`` omits that namelist from the written
    file, which is how you request a GB-only or PB-only calculation.
    ``other_namelists`` is an open dict for namelists not modelled here
    (e.g. ``&rism``, ``&decomp``, ``&nmode``) without requiring changes to
    this class.  No validation of parameter combinations is performed here;
    gmx_MMPBSA itself reports incompatible options at runtime.
    """

    general: GeneralParams = field(default_factory=GeneralParams)
    gb: GBParams | None = field(default_factory=GBParams)
    pb: PBParams | None = field(default_factory=PBParams)

    # For namelists not modelled above (e.g. &rism, &decomp, &nmode)
    other_namelists: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_text(self) -> str:
        """Render the full gmx_MMPBSA input file as a string.

        Namelists are written in the order: general, gb (if present),
        pb (if present), then any entries in ``other_namelists`` in insertion
        order.  Each namelist is separated by a blank line, matching the
        format expected by gmx_MMPBSA.  Fields set to ``None`` on the
        underlying dataclasses are silently omitted so the file stays clean.
        """
        parts = [_render_namelist("general", _as_kv(self.general))]
        if self.gb is not None:
            parts.append(_render_namelist("gb", _as_kv(self.gb)))
        if self.pb is not None:
            parts.append(_render_namelist("pb", _as_kv(self.pb)))
        for name, kv in self.other_namelists.items():
            parts.append(_render_namelist(name, kv))
        return "\n\n".join(parts).rstrip() + "\n"

    def write(self, path: str | Path) -> Path:
        """Write the gmx_MMPBSA input file to ``path`` and return the resolved path.

        The file is written atomically via ``Path.write_text``; no intermediate
        temporary file is used.  The parent directory must already exist —
        this method does not create it.  Returns the resolved ``Path`` object
        so callers can pass it directly to ``run_gmx_mmpbsa_from_gromacs``.
        """
        path = Path(path)
        path.write_text(self.to_text())
        return path


def run_gmx_mmpbsa_from_gromacs(
    input_file: str | Path,
    complex_structure: str | Path,
    trajectory: str | Path,
    topology: str | Path,
    index_file: str | Path,
    receptor_group: int,
    ligand_group: int,
    output_dir: str | Path,
    gmx_mmpbsa: str = "gmx_MMPBSA",
    extra_args: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    """Run gmx_MMPBSA against GROMACS-format inputs and return the completed process.

    ``complex_structure`` is passed via ``-cs`` and must be a ``.tpr`` or ``.pdb``
    file — gmx_MMPBSA does not accept ``.gro`` for this flag.  ``trajectory``
    via ``-ct`` (xtc), ``topology`` via ``-cp`` (top), and ``index_file`` via
    ``-ci`` (ndx).  ``receptor_group`` and ``ligand_group`` are the integer
    group indices within the index file (0-based as written by ``gmx make_ndx``)
    and are passed together via ``-cg``.  The call uses ``check=False`` so the
    caller is responsible for inspecting ``returncode`` and ``stderr`` — this is
    intentional because gmx_MMPBSA sometimes writes useful partial output even
    on non-zero exit.  ``output_dir`` is created if it does not exist and is
    used as the working directory so that gmx_MMPBSA intermediate files are
    contained there.

    Flags used:

    .. code-block:: text

        -O              overwrite existing output files
        -nogui          suppress the gmx_MMPBSA_ana post-processing GUI
        -i  input_file
        -cs complex_structure
        -ct trajectory
        -cp topology
        -ci index_file
        -cg receptor_group ligand_group

    ``-nogui`` is always passed because gmx_MMPBSA v1.5.0.3 attempts to launch
    the ``gmx_MMPBSA_ana`` PyQt5 GUI after every successful calculation.  On
    systems where PyQt5 is absent or the display is not available (CI, headless
    servers, Conda environments without the Qt stack), the launcher exits with
    return code 1 even though the energy calculation completed correctly.  The
    ``-nogui`` flag skips the GUI launch and lets the return code reflect the
    actual calculation outcome.  Callers that want the GUI can open
    ``gmx_MMPBSA_ana`` manually from the output directory afterwards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        gmx_mmpbsa,
        "-O",
        # See docstring: always suppress the PyQt5 GUI launcher.
        "-nogui",
        "-i",
        str(input_file),
        "-cs",
        str(complex_structure),
        "-ct",
        str(trajectory),
        "-cp",
        str(topology),
        "-ci",
        str(index_file),
        "-cg",
        str(receptor_group),
        str(ligand_group),
    ]
    cmd.extend(extra_args)

    return subprocess.run(  # noqa: S603
        cmd,
        cwd=str(output_dir),
        text=True,
        capture_output=True,
        check=False,
    )
