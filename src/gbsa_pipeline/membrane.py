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

These two are independent by design: ``estimate_membrane_geometry`` works on
any structure with recognisable lipid headgroups, not only MemProtMD output.
"""

from __future__ import annotations

import logging
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
