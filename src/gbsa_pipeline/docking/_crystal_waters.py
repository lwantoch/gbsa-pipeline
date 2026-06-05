"""Crystal-water selection and post-docking validation/orchestration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gemmi
import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from gbsa_pipeline.docking._models import DockingManifest, DockingValidation
from gbsa_pipeline.docking._receptor_prep import merge_pdb_structures

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from gbsa_pipeline.docking._models import (
        DockingBox,
        DockingEngine,
        DockingRequest,
        DockingResult,
    )

LOGGER = logging.getLogger(__name__)

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "TIP3", "TIP3P", "SOL"})


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _read_pdb_like(path: Path) -> gemmi.Structure:
    """Read a PDB or PDBQT file; only the first MODEL block is kept.

    Vina writes all poses into one PDBQT (MODEL 1…N). Gemmi raises on
    duplicate model numbers when the full file is parsed, so we truncate to
    the first ENDMDL before handing off to the parser.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    endmdl = text.find("ENDMDL")
    if endmdl != -1:
        text = text[: endmdl + len("ENDMDL")] + "\nEND\n"
    return gemmi.read_pdb_string(text)


def _atom_pos(atom: Any) -> np.ndarray:
    """Convert a Gemmi atom position to a (3,) numpy array."""
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z])


def _first_heavy_atom(residue: Any) -> Any | None:
    """Return the first non-hydrogen atom in a residue, or None if all are hydrogens."""
    return next((a for a in residue if not a.is_hydrogen()), None)


def _pdb_heavy_atom_coords(
    pdb_path: Path,
    *,
    exclude_residues: frozenset[str] = _WATER_RESNAMES,
) -> np.ndarray:
    """Return (N, 3) heavy-atom coordinates from the first model of a PDB/PDBQT file."""
    structure = _read_pdb_like(pdb_path)
    if not structure:
        return np.empty((0, 3))
    coords = [
        _atom_pos(atom)
        for chain in structure[0]
        for residue in chain
        if residue.name.strip().upper() not in exclude_residues
        for atom in residue
        if not atom.is_hydrogen()
    ]
    return np.array(coords) if coords else np.empty((0, 3))


def _sdf_heavy_atom_coords(sdf_path: Path) -> np.ndarray:
    """Return (N, 3) heavy-atom coordinates from the first SDF conformer."""
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return np.empty((0, 3))
    conf = mol.GetConformer(0)
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def _pose_heavy_atom_coords(pose_path: Path) -> np.ndarray:
    """Return (N, 3) heavy-atom coordinates for a PDB/PDBQT or SDF pose file."""
    if pose_path.suffix.lower() in (".pdb", ".pdbqt"):
        return _pdb_heavy_atom_coords(pose_path, exclude_residues=frozenset())
    return _sdf_heavy_atom_coords(pose_path)


def _iter_water_oxygens(pdb_path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(res_id, xyz)`` of the first heavy atom for each residue in the PDB."""
    structure = _read_pdb_like(pdb_path)
    if not structure:
        return
    for chain in structure[0]:
        for residue in chain:
            atom = _first_heavy_atom(residue)
            if atom is not None:
                yield str(residue.seqid.num), _atom_pos(atom)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _best_score(result: DockingResult) -> float | None:
    """Return the score of the first pose that has one, or None."""
    return next((p.score for p in result.poses if p.score is not None), None)


def _best_pose_path(result: DockingResult) -> Path | None:
    """Return the pose_path of the rank-1 pose, or None."""
    best = next((p for p in result.poses if p.rank == 1), None)
    return best.pose_path if best is not None else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_docking_crystal_waters(
    crystal_waters_pdb: Path,
    box: DockingBox,
    ligand_sdf: Path,
    receptor_pdb: Path,
    output_pdb: Path,
    *,
    ligand_cutoff_angstrom: float = 5.0,
    receptor_clash_cutoff_angstrom: float = 2.2,
) -> tuple[Path | None, list[str]]:
    """Select crystal waters that pass three spatial filters.

    A water oxygen must be: (1) inside the docking box, (2) within
    ``ligand_cutoff_angstrom`` of a ligand heavy atom, and (3) not within
    ``receptor_clash_cutoff_angstrom`` of any receptor heavy atom.

    Returns ``(output_pdb_path, retained_residue_ids)``, or ``(None, [])``
    when no waters survive all three filters.
    """
    if not crystal_waters_pdb.exists():
        return None, []

    structure = _read_pdb_like(crystal_waters_pdb)
    if not structure:
        return None, []

    lig_coords = _pose_heavy_atom_coords(ligand_sdf)
    rec_coords = _pdb_heavy_atom_coords(receptor_pdb)
    lig_tree = cKDTree(lig_coords) if lig_coords.size > 0 else None
    rec_tree = cKDTree(rec_coords) if rec_coords.size > 0 else None

    cx, cy, cz = box.center
    hx, hy, hz = box.size[0] / 2.0, box.size[1] / 2.0, box.size[2] / 2.0

    out_chain = gemmi.Chain("W")
    retained_ids: list[str] = []

    for chain in structure[0]:
        for residue in chain:
            if (_ha := _first_heavy_atom(residue)) is None:
                continue
            ow = _atom_pos(_ha)
            if not (cx - hx <= ow[0] <= cx + hx and cy - hy <= ow[1] <= cy + hy and cz - hz <= ow[2] <= cz + hz):
                continue
            if lig_tree is not None and not lig_tree.query_ball_point(ow, r=ligand_cutoff_angstrom):
                continue
            if rec_tree is not None and rec_tree.query_ball_point(ow, r=receptor_clash_cutoff_angstrom):
                LOGGER.debug("Crystal water %s discarded (receptor clash).", residue.seqid.num)
                continue
            out_chain.add_residue(residue.clone())
            retained_ids.append(str(residue.seqid.num))

    if not retained_ids:
        LOGGER.debug("No crystal waters survived selection.")
        return None, []

    out_model = gemmi.Model("1")
    out_model.add_chain(out_chain)
    out_st = gemmi.Structure()
    out_st.add_model(out_model)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_st.write_pdb(str(output_pdb))

    LOGGER.info("Selected %d crystal waters: %s", len(retained_ids), retained_ids)
    return output_pdb, retained_ids


def validate_docked_pose(
    pose_sdf: Path,
    receptor_pdb: Path,
    retained_waters_pdb: Path | None = None,
    *,
    clash_cutoff_angstrom: float = 2.0,
    bridge_min_angstrom: float = 2.6,
    bridge_max_angstrom: float = 3.2,
) -> DockingValidation:
    """Check clashes and water bridges for a docked pose.

    Uses ``cKDTree`` for all distance queries.  A "water bridge" is a crystal
    water whose oxygen is within ``[bridge_min, bridge_max]`` Å of both a
    ligand heavy atom and a receptor heavy atom simultaneously.
    """
    lig_coords = _pose_heavy_atom_coords(pose_sdf)
    rec_coords = _pdb_heavy_atom_coords(receptor_pdb)
    lig_tree = cKDTree(lig_coords) if lig_coords.size > 0 else None
    rec_tree = cKDTree(rec_coords) if rec_coords.size > 0 else None

    lig_protein_clashes: list[tuple[str, float]] = []
    lig_water_clashes: list[tuple[str, float]] = []
    water_protein_clashes: list[tuple[str, float]] = []
    water_bridges: list[tuple[str, str, str, float, float]] = []

    if lig_coords.size > 0 and rec_tree is not None:
        for i, lpos in enumerate(lig_coords):
            for j in rec_tree.query_ball_point(lpos, r=clash_cutoff_angstrom):
                lig_protein_clashes.append(
                    (
                        f"lig_atom{i}<->rec_atom{j}",
                        float(np.linalg.norm(lpos - rec_coords[j])),
                    )
                )

    if retained_waters_pdb is not None and retained_waters_pdb.exists():
        for res_id, ow in _iter_water_oxygens(retained_waters_pdb):
            if rec_tree is not None:
                for j in rec_tree.query_ball_point(ow, r=clash_cutoff_angstrom):
                    water_protein_clashes.append(
                        (
                            f"HOH{res_id}<->rec_atom{j}",
                            float(np.linalg.norm(ow - rec_coords[j])),
                        )
                    )
            if lig_tree is not None:
                for i in lig_tree.query_ball_point(ow, r=clash_cutoff_angstrom):
                    lig_water_clashes.append(
                        (
                            f"HOH{res_id}<->lig_atom{i}",
                            float(np.linalg.norm(ow - lig_coords[i])),
                        )
                    )
            if lig_tree is not None and rec_tree is not None:
                for i in lig_tree.query_ball_point(ow, r=bridge_max_angstrom):
                    dl = float(np.linalg.norm(ow - lig_coords[i]))
                    if dl < bridge_min_angstrom:
                        continue
                    for j in rec_tree.query_ball_point(ow, r=bridge_max_angstrom):
                        dr = float(np.linalg.norm(ow - rec_coords[j]))
                        if dr >= bridge_min_angstrom:
                            water_bridges.append((f"HOH{res_id}", f"lig_atom{i}", f"rec_atom{j}", dl, dr))

    return DockingValidation(
        ligand_protein_clashes=lig_protein_clashes,
        ligand_water_clashes=lig_water_clashes,
        water_protein_clashes=water_protein_clashes,
        water_bridges=water_bridges,
    )


def dock_with_and_without_crystal_waters(
    engine: DockingEngine,
    request: DockingRequest,
    crystal_waters_pdb: Path | None,
    ligand_sdf: Path,
    work_dir: Path,
    *,
    use_crystal_waters: bool = True,
    ligand_cutoff_angstrom: float = 5.0,
    receptor_clash_cutoff_angstrom: float = 2.2,
) -> DockingManifest:
    """Run Vina twice — without and with selected active-site crystal waters.

    Crystal water selection applies three filters: inside docking box, within
    ``ligand_cutoff_angstrom`` of a ligand heavy atom, and not clashing with
    the receptor.  Set ``use_crystal_waters=False`` to skip the second run.

    The receptor in ``request`` must be a ``.pdb`` file so the water-augmented
    version can be assembled.  Both poses are validated post-hoc for clashes
    and water bridges.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Run 1: no crystal waters -------------------------------------------
    result_no_water = engine.dock(request.model_copy(update={"workdir": work_dir / "no_water"}))
    score_no_water = _best_score(result_no_water)
    pose_no_water = _best_pose_path(result_no_water)
    val_no_water = validate_docked_pose(pose_no_water, request.receptor) if pose_no_water is not None else None

    def _no_water_manifest() -> DockingManifest:
        return DockingManifest(
            without_waters=result_no_water,
            with_waters=None,
            score_without=score_no_water,
            score_with=None,
            retained_water_ids=[],
            receptor_without_waters=request.receptor,
            receptor_with_waters=None,
            validation_without=val_no_water,
            validation_with=None,
        )

    if not use_crystal_waters or crystal_waters_pdb is None or not crystal_waters_pdb.exists():
        return _no_water_manifest()

    if request.receptor.suffix.lower() != ".pdb":
        LOGGER.warning("receptor must be a .pdb file for crystal-water docking; skipping water run.")
        return _no_water_manifest()

    ligand_ref = pose_no_water if (pose_no_water is not None and pose_no_water.exists()) else ligand_sdf

    selected_path, retained_ids = select_docking_crystal_waters(
        crystal_waters_pdb,
        request.box,
        ligand_ref,
        request.receptor,
        work_dir / "selected_crystal_waters.pdb",
        ligand_cutoff_angstrom=ligand_cutoff_angstrom,
        receptor_clash_cutoff_angstrom=receptor_clash_cutoff_angstrom,
    )

    if selected_path is None:
        return _no_water_manifest()

    # --- Run 2: receptor + selected crystal waters --------------------------
    receptor_with_waters = work_dir / "receptor_with_crystal_waters.pdb"
    merge_pdb_structures(request.receptor, selected_path, receptor_with_waters)

    try:
        result_with_water = engine.dock(
            request.model_copy(
                update={
                    "receptor": receptor_with_waters,
                    "workdir": work_dir / "with_water",
                }
            )
        )
    # Docking backends may raise backend-specific exceptions depending on the
    # active executable, input structure, and search-space setup. This fallback
    # intentionally catches broad failures because a failed with-water attempt
    # should not prevent returning the no-water docking result.
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "With-waters docking failed (%s: %s); falling back to no-water result.",
            type(exc).__name__,
            exc,
        )
        return _no_water_manifest()

    score_with_water = _best_score(result_with_water)
    pose_with_water = _best_pose_path(result_with_water)
    val_with_water = (
        validate_docked_pose(pose_with_water, request.receptor, selected_path) if pose_with_water is not None else None
    )

    LOGGER.info(
        "Docking scores — without waters: %s | with waters: %s | retained: %s",
        score_no_water,
        score_with_water,
        retained_ids,
    )

    return DockingManifest(
        without_waters=result_no_water,
        with_waters=result_with_water,
        score_without=score_no_water,
        score_with=score_with_water,
        retained_water_ids=retained_ids,
        receptor_without_waters=request.receptor,
        receptor_with_waters=receptor_with_waters,
        validation_without=val_no_water,
        validation_with=val_with_water,
    )
