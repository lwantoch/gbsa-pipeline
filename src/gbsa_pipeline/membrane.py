"""Membrane-protein systems: fetch from MemProtMD, measure bilayer geometry.

This module has two independent jobs:

``fetch_memprotmd_system``
    Download a pre-built, already-solvated protein-in-bilayer system from
    MemProtMD (https://memprotmd.bioch.ox.ac.uk/), a database of coarse-grained
    self-assembly simulations (converted back to atomistic detail) for every
    transmembrane protein in the PDB. The result is a GROMACS structure+topology
    pair that plugs into this pipeline *after* parametrize/solvate — see
    ``RunConfig.membrane_system`` — because MemProtMD has already placed the
    protein in a bilayer, solvated it, and added ions.

``estimate_membrane_geometry``
    Measure the ``mthick``/``mctrdz`` parameters that ``mmbsa.PBParams`` needs
    for its implicit-membrane PB model (``memopt=1``) directly from a lipid's
    phosphate atoms, so the values fed into the GBSA/PBSA calculation reflect
    the system that was actually simulated rather than a guess.

``canonicalize_gromacs_system``
    Round-trip a structure+topology pair through ``gmx grompp``/``editconf``
    to fix two legacy-GROMACS issues that break BioSimSpace/Sire's loader —
    see its docstring. Needed before ``BSS.IO.readMolecules`` for MemProtMD's
    ``atomistic-system.pdb`` (written by GROMACS 4.6.2's pdb2gmx).

These are independent by design: ``estimate_membrane_geometry`` works on any
structure with recognisable lipid headgroups, not only MemProtMD output.

Known limitation: MemProtMD's default output is GROMOS53a6-parametrized
(G96 bonds/angles), which needs converting to harmonic form before
BioSimSpace/Sire can even round-trip it — see
``scripts/convert_gromos_to_harmonic.py`` (deliberately a standalone script,
not part of this module: it's a one-off data-prep fix, not something to run
on every load). Converting the bonds/angles is necessary but *not*
sufficient to run a full MD stage through this pipeline: BioSimSpace/Sire's
GROTOP writer also silently drops ``[nonbond_params]``/``[pairtypes]``
override tables on re-serialization (confirmed: 541 + 105 entries in
MemProtMD's own ``itp/lipid-gmx53a6.itp``, 0 in what Sire writes back out),
which the OPLS-style lipid parameters need for correct nonbonded
interactions. For MemProtMD's 1py6 example this data loss is enough that
minimization diverges to a NaN potential energy at step 0, even though the
identical (harmonic-converted) files minimize cleanly under plain
``gmx grompp``/``mdrun``. This is a BioSimSpace/Sire topology-writer bug, not
fixable from this pipeline's code — loading (``canonicalize_gromacs_system``)
and geometry measurement work regardless, but a full run needs either a
membrane system whose force field doesn't rely on ``[nonbond_params]``
overrides, or an upstream fix to Sire's GROTOP writer.

This is specific to that override-table class of force field, not to
membrane systems generally: an **AMBER** ``.prmtop``/``.inpcrd`` membrane
system (Lipid21, no ``[nonbond_params]``-equivalent mechanism) needs neither
``canonicalize_gromacs_system`` nor a bond/angle conversion, and is confirmed
working end-to-end (load, SD minimization) against a real
packmol-memgen-built system — see ``config.MembraneSystemConfig``'s
docstring. Prefer AMBER format when you have a choice of how to build the
input system.
"""

from __future__ import annotations

import logging
import subprocess
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gemmi

from gbsa_pipeline.mmbsa import PBParams

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

# Common phospholipid residue names. PDB's 3-character resname field truncates
# 4-letter names (MemProtMD's own DPPC becomes "DPP" — confirmed against a
# real downloaded system, see tests/testdata/membrane/), so both spellings are
# listed where relevant. This list is a convenience default, not a promise:
# always check the resnames in *your* structure and pass lipid_resnames=
# explicitly if they differ.
DEFAULT_LIPID_RESNAMES: frozenset[str] = frozenset(
    {
        "DPP",
        "DPPC",
        "POP",
        "POPC",
        "POPE",
        "POPG",
        "POPS",
        "POPI",
        "DOP",
        "DOPC",
        "DOPE",
        "DMP",
        "DMPC",
        "DLPC",
    }
)

# A symmetric bilayer needs a meaningful number of lipids per leaflet for the
# mean phosphate z-position to be a stable estimate of that leaflet's plane.
_MIN_PHOSPHATES_PER_LEAFLET = 5

_MEMPROTMD_AT_URL = (
    "https://memprotmd.bioch.ox.ac.uk/data/memprotmd/simulations/{pdb_code}_default_{lipid}/files/run/at.zip"
)


# ---------------------------------------------------------------------------
# Fetching from MemProtMD
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemProtMDSystem:
    """Paths into a downloaded MemProtMD atomistic (CG2AT-converted) system.

    ``structure``/``topology`` are ready to load with
    ``BSS.IO.readMolecules([str(structure), str(topology)])``. The topology's
    force-field ``#include`` directives are relative paths resolved against
    its own directory, so ``structure``, ``topology``, and any sibling
    ``itp/``/``*.ff/`` directories must stay together on disk — they all live
    under ``root``, exactly as unpacked from MemProtMD's zip.
    """

    structure: Path
    topology: Path
    index: Path | None
    root: Path


def fetch_memprotmd_system(
    pdb_code: str,
    dest_dir: Path,
    lipid: str = "dppc",
    *,
    timeout: float = 120.0,
) -> MemProtMDSystem:
    """Download and unpack a MemProtMD atomistic system for *pdb_code*.

    ``lipid`` selects which of MemProtMD's simulations to fetch (most PDB
    entries were only run with DPPC, the tool's default lipid for its initial
    coarse-grained self-assembly step; a handful of entries have alternative
    lipid runs too). The downloaded archive already contains a complete
    protein-in-bilayer-in-water(-and-ions) system plus the GROMACS force field
    files it depends on — nothing further needs to be built.

    The REST endpoint used here isn't documented on MemProtMD's own (JS
    rendered) ``/api/`` page; it was recovered from the ``biobb_io`` package
    (``biobb_io.api.common.get_memprotmd_sim``), which downloads the same
    file. It may change without notice since it's unofficial.

    Raises ``FileNotFoundError`` if MemProtMD has no matching entry (HTTP 404,
    the ordinary case when the PDB code or lipid is wrong) or the archive
    doesn't contain the expected files, and re-raises other
    ``urllib.error.HTTPError``/``URLError`` failures unchanged.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    pdb_code_lower = pdb_code.strip().lower()
    lipid_lower = lipid.strip().lower()
    url = _MEMPROTMD_AT_URL.format(pdb_code=pdb_code_lower, lipid=lipid_lower)
    zip_path = dest_dir / f"{pdb_code_lower}_{lipid_lower}_at.zip"

    logger.info("Downloading MemProtMD atomistic system: %s (lipid=%s) from %s", pdb_code, lipid, url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 — fixed https host
            zip_path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # noqa: PLR2004
            raise FileNotFoundError(
                f"No MemProtMD entry for pdb_code={pdb_code!r} lipid={lipid!r} (404 from {url}). "
                "Browse https://memprotmd.bioch.ox.ac.uk/ to confirm the PDB code and available lipid."
            ) from exc
        raise
    logger.debug("Downloaded %s (%d bytes)", zip_path, zip_path.stat().st_size)

    extract_dir = dest_dir / pdb_code_lower
    _safe_extract(zip_path, extract_dir)

    structure = extract_dir / "atomistic-system.pdb"
    topology = extract_dir / "topol.top"
    index = extract_dir / "atomistic-system.ndx"
    if not structure.exists() or not topology.exists():
        found = sorted(p.name for p in extract_dir.iterdir())
        raise FileNotFoundError(
            f"MemProtMD archive for {pdb_code} did not contain the expected "
            f"atomistic-system.pdb/topol.top (found: {found})."
        )

    return MemProtMDSystem(
        structure=structure,
        topology=topology,
        index=index if index.exists() else None,
        root=extract_dir,
    )


def _safe_extract(zip_path: Path, dest_dir: Path) -> None:
    """Extract *zip_path* into *dest_dir*, rejecting members that would escape it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest_dir / member.filename).resolve()
            if not target.is_relative_to(resolved_dest):
                raise ValueError(f"Refusing to extract unsafe zip member: {member.filename!r}")
        zf.extractall(dest_dir)  # members validated above


# ---------------------------------------------------------------------------
# Canonicalizing legacy GROMACS systems for BioSimSpace/Sire
# ---------------------------------------------------------------------------

# grompp only needs to succeed here, not produce a physically meaningful
# .tpr -- nothing is ever simulated from it -- so electrostatics/cutoff
# settings are irrelevant to correctness; generous cutoffs just avoid
# unrelated grompp warnings for whatever box size the input happens to have.
_CANONICALIZE_MDP = """\
integrator    = steep
nsteps        = 0
cutoff-scheme = Verlet
coulombtype   = cut-off
rcoulomb      = 1.2
rvdw          = 1.2
rlist         = 1.2
"""


def canonicalize_gromacs_system(
    structure: Path,
    topology: Path,
    work_dir: Path,
    *,
    maxwarn: int = 50,
    gmx: str = "gmx",
) -> Path:
    """Round-trip *structure*+*topology* through GROMACS into a fresh ``.gro``.

    BioSimSpace/Sire's own GROMACS reader is stricter than ``gmx grompp``
    about two issues found in older systems (e.g. MemProtMD's
    ``atomistic-system.pdb``, written by GROMACS 4.6.2's ``pdb2gmx`` — see
    its ``topol.top`` header) and in large ones — both confirmed against that
    exact file:

    * Branched-hydrogen atom names with the counting digit *before* the name
      in the coordinate file (``"1HH1"``) but *after* it in the topology
      (``"HH11"``). ``grompp`` accepts this with an "atom name ... does not
      match" warning and uses the topology's names; Sire raises "Could not
      find a matching atom record" and ``BSS.IO.readMolecules`` fails outright.
    * PDB's 4-digit residue-number field wrapping once a system has more than
      9999 residues (this pipeline's 1py6 example has ~14000, wrapping
      9999 -> 0). ``grompp`` doesn't care — it matches coordinates to the
      topology positionally, by atom count, never by residue number — but
      Sire's loader does key off residue number and fails the same way as
      above, on an arbitrary water molecule wherever the wrap lands.

    Running the pair through ``gmx grompp`` (a consistency check only —
    ``nsteps = 0``, nothing is simulated) and then ``gmx editconf`` writes a
    fresh ``.gro`` using the topology's own atom names and a 5-digit residue
    field, sidestepping both issues at once. Safe to call even when neither
    issue applies (e.g. an input that is already a clean ``.gro`) — the
    round-trip is then a no-op beyond regenerating an equivalent file.

    Requires a real GROMACS install: ``gmx`` on ``PATH``, or pass its full
    path via ``gmx=``. Raises ``RuntimeError`` (with the captured stderr) if
    either ``grompp`` or ``editconf`` fails.
    """
    # Resolve to absolute paths before setting cwd=work_dir below: a relative
    # structure/topology (e.g. from a config file loaded elsewhere) would
    # otherwise be reinterpreted relative to work_dir instead of the caller's
    # original cwd, and a relative work_dir would double up on itself the
    # same way for mdp/tpr/canonical_gro.
    work_dir = work_dir.resolve()
    structure = structure.resolve()
    topology = topology.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    mdp = work_dir / "canonicalize.mdp"
    mdp.write_text(_CANONICALIZE_MDP)
    tpr = work_dir / "canonicalize.tpr"

    grompp = subprocess.run(  # noqa: S603
        [
            gmx,
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(structure),
            "-p",
            str(topology),
            "-o",
            str(tpr),
            "-maxwarn",
            str(maxwarn),
        ],
        capture_output=True,
        cwd=work_dir,
        check=False,
    )
    if grompp.returncode != 0:
        raise RuntimeError(f"gmx grompp failed while canonicalizing {structure}:\n{grompp.stderr.decode()}")

    canonical_gro = work_dir / "canonical.gro"
    editconf = subprocess.run(  # noqa: S603
        [gmx, "editconf", "-f", str(tpr), "-o", str(canonical_gro)],
        capture_output=True,
        cwd=work_dir,
        check=False,
    )
    if editconf.returncode != 0:
        raise RuntimeError(f"gmx editconf failed while canonicalizing {structure}:\n{editconf.stderr.decode()}")

    return canonical_gro


# ---------------------------------------------------------------------------
# Measuring bilayer geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MembraneGeometry:
    """Bilayer geometry measured from a lipid's phosphate atoms.

    ``mctrdz`` is an *absolute* z-coordinate in the same frame as the
    structure it was measured from — that is what gmx_MMPBSA's
    ``PBParams.mctrdz`` expects (it is not an offset from the protein or box
    center); see https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA/discussions/436.
    ``mthick`` is the headgroup-to-headgroup (phosphate-to-phosphate)
    thickness, the standard bilayer thickness metric.
    """

    mctrdz: float
    mthick: float
    n_phosphates: int

    def pb_params(self, **overrides: Any) -> PBParams:
        """Build a membrane-ready ``PBParams(memopt=1, ...)`` from this geometry.

        Sets ``mctrdz``/``mthick`` from the measurement and ``eneopt=1``
        (required alongside ``memopt=1`` — see ``PBParams.__post_init__``).
        Any keyword in ``overrides`` (e.g. ``emem``, ``poretype``) takes
        precedence, including over ``eneopt``/``mctrdz``/``mthick`` themselves
        if the caller has a reason to override the measurement.
        """
        kwargs: dict[str, Any] = {
            "memopt": 1,
            "mctrdz": self.mctrdz,
            "mthick": self.mthick,
            "eneopt": 1,
        }
        kwargs.update(overrides)
        return PBParams(**kwargs)


def estimate_membrane_geometry(
    structure: Path,
    lipid_resnames: Sequence[str] = tuple(DEFAULT_LIPID_RESNAMES),
) -> MembraneGeometry:
    """Measure bilayer ``mthick``/``mctrdz`` from a structure's lipid phosphate atoms.

    ``structure`` must be a ``.pdb`` or ``.cif``/``.mmcif`` file — whatever
    gemmi (this function's parser) reads. It is *not* a GROMACS ``.gro``:
    gemmi doesn't parse that format, so a production-stage ``.gro`` must be
    converted first, e.g. ``BSS.IO.saveMolecules(prefix, system, "pdb")``
    (the same conversion the mmbsa integration test does before calling
    gmx_MMPBSA, which also needs ``-cs`` as ``.pdb``/``.tpr``, never ``.gro``).

    Collects the z-coordinate of every atom whose residue name is in
    ``lipid_resnames`` and whose atom name starts with ``"P"`` (the phosphate
    — true across the CHARMM ``"P"``, Amber Lipid21 ``"P31"``, and GROMOS/
    Berger ``"P8"`` naming conventions). A symmetric bilayer's phosphates
    cluster into two well-separated bands, so splitting at the overall mean z
    reliably separates the two leaflets without needing clustering.

    Raises ``ValueError`` if fewer than ``_MIN_PHOSPHATES_PER_LEAFLET`` atoms
    land in either leaflet — usually a sign that ``lipid_resnames`` doesn't
    match the structure (check the actual residue names first) or that the
    structure isn't a bilayer at all — and if ``structure`` is a ``.gro`` file.
    """
    if str(structure).lower().endswith(".gro"):
        raise ValueError(
            f"estimate_membrane_geometry() cannot read GROMACS .gro files ({structure}); "
            "gemmi doesn't parse that format. Convert to PDB first, e.g. "
            'BSS.IO.saveMolecules(prefix, system, "pdb").'
        )

    resnames = frozenset(lipid_resnames)
    struct = (
        gemmi.read_pdb(str(structure))
        if str(structure).lower().endswith(".pdb")
        else gemmi.read_structure(str(structure))
    )

    phosphate_z: list[float] = [
        atom.pos.z
        for model in struct
        for chain in model
        for residue in chain
        if residue.name.strip() in resnames
        for atom in residue
        if atom.name.strip().upper().startswith("P")
    ]

    if not phosphate_z:
        raise ValueError(
            f"No phosphate atoms found for lipid_resnames={sorted(resnames)} in {structure}. "
            "Check the actual lipid residue names in the structure (PDB truncates 4-letter "
            "resnames like DPPC to 3 characters) and pass lipid_resnames= explicitly."
        )

    mean_z = sum(phosphate_z) / len(phosphate_z)
    upper = [z for z in phosphate_z if z >= mean_z]
    lower = [z for z in phosphate_z if z < mean_z]

    if len(upper) < _MIN_PHOSPHATES_PER_LEAFLET or len(lower) < _MIN_PHOSPHATES_PER_LEAFLET:
        raise ValueError(
            f"Phosphate atoms did not split into two comparable leaflets "
            f"({len(lower)} below, {len(upper)} above the mean z={mean_z:.2f}); "
            "this structure may not be a symmetric bilayer, or lipid_resnames is wrong."
        )

    mthick = abs(sum(upper) / len(upper) - sum(lower) / len(lower))

    return MembraneGeometry(mctrdz=mean_z, mthick=mthick, n_phosphates=len(phosphate_z))
