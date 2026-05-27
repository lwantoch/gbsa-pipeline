"""Crystal-water selection and post-docking validation/orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from gbsa_pipeline.docking._receptor_prep import prepare_receptor_with_crystal_waters

if TYPE_CHECKING:
    from pathlib import Path

    from gbsa_pipeline.docking._models import DockingBox, DockingEngine, DockingRequest, DockingResult

LOGGER = logging.getLogger(__name__)

_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "TIP3", "TIP3P", "SOL"})


# ---------------------------------------------------------------------------
# Crystal-water docking: data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DockingValidation:
    """Post-docking geometry checks for one docked pose.

    All distances are in A.  Each entry in the clash lists is a
    ``(description, distance)`` tuple; bridge entries are
    ``(water_res_id, ligand_description, protein_description, lig_dist, prot_dist)``.
    """

    ligand_protein_clashes: list[tuple[str, float]]
    ligand_water_clashes: list[tuple[str, float]]
    water_protein_clashes: list[tuple[str, float]]
    water_bridges: list[tuple[str, str, str, float, float]]

    @property
    def has_clashes(self) -> bool:
        """Return True when any clash list is non-empty."""
        return bool(self.ligand_protein_clashes or self.ligand_water_clashes or self.water_protein_clashes)


@dataclass(frozen=True)
class DockingManifest:
    """Complete record of a dual docking run with and without active-site crystal waters.

    ``with_waters`` is ``None`` when no crystal waters passed the selection
    criteria (inside box + near ligand + no receptor clash).
    """

    without_waters: DockingResult
    with_waters: DockingResult | None
    score_without: float | None
    score_with: float | None
    retained_water_ids: list[str]
    receptor_without_waters: Path
    receptor_with_waters: Path | None
    validation_without: DockingValidation | None
    validation_with: DockingValidation | None


# ---------------------------------------------------------------------------
# Coordinate extraction helpers
# ---------------------------------------------------------------------------


def _pdb_heavy_atom_coords(
    pdb_path: Path,
    *,
    exclude_residues: frozenset[str] = _WATER_RESNAMES,
) -> np.ndarray:
    """Return (N, 3) array of non-hydrogen, non-water PDB atom coordinates in A."""
    coords: list[list[float]] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        resname = line[17:20].strip().upper()
        if resname in exclude_residues:
            continue
        atom_name = line[12:16].strip()
        if atom_name.startswith("H"):
            continue
        try:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    return np.array(coords) if coords else np.empty((0, 3))


def _sdf_heavy_atom_coords(sdf_path: Path) -> np.ndarray:
    """Return (N, 3) array of heavy-atom coordinates in A from the first SDF conformer."""
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return np.empty((0, 3))
    conf = mol.GetConformer(0)
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def _pose_heavy_atom_coords(pose_path: Path) -> np.ndarray:
    """Return (N, 3) heavy-atom coordinates for a docked pose in A.

    Accepts SDF/MOL (parsed with RDKit) or PDB/PDBQT (parsed column-by-column).
    PDBQT from Vina uses standard PDB coordinate columns, so the same parser
    applies to both.  This allows crystal-water selection to use the actual
    docked position rather than the pre-docking embedded geometry.
    """
    suffix = pose_path.suffix.lower()
    if suffix in (".pdb", ".pdbqt"):
        return _pdb_heavy_atom_coords(pose_path, exclude_residues=frozenset())
    return _sdf_heavy_atom_coords(pose_path)


def _group_water_residues(
    crystal_waters_pdb: Path,
) -> list[tuple[str, list[str], tuple[float, float, float] | None]]:
    """Parse a crystal-waters PDB into (res_id, lines, oxygen_xyz) groups."""
    groups: list[tuple[str, list[str], tuple[float, float, float] | None]] = []
    current_id: str | None = None
    current_lines: list[str] = []
    current_ow: tuple[float, float, float] | None = None

    def _flush() -> None:
        if current_id is not None:
            groups.append((current_id, list(current_lines), current_ow))

    for line in crystal_waters_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        res_id = line[22:26].strip()
        if res_id != current_id:
            _flush()
            current_id = res_id
            current_lines = []
            current_ow = None
        current_lines.append(line)
        atom_name = line[12:16].strip()
        if not atom_name.startswith("H") and current_ow is None:
            with __import__("contextlib").suppress(ValueError):
                current_ow = (float(line[30:38]), float(line[38:46]), float(line[46:54]))

    _flush()
    return groups


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
    """Select crystal waters suitable for inclusion in the docking receptor.

    A water oxygen must satisfy all three criteria to be kept:

    1. Inside the docking box (binding-site spatial filter).
    2. Within ``ligand_cutoff_angstrom`` of any ligand heavy atom — ensures
       the water is in the binding site, not just anywhere in the box.
    3. NOT within ``receptor_clash_cutoff_angstrom`` of any receptor heavy atom
       — discards waters with impossible crystal-packing contacts.

    Returns ``(path_to_selected_pdb, [retained_residue_ids])``.
    The path is ``None`` when no waters survive all three filters.
    """
    if not crystal_waters_pdb.exists():
        return None, []

    groups = _group_water_residues(crystal_waters_pdb)
    if not groups:
        return None, []

    # --- Build spatial indices ---
    lig_coords = _pose_heavy_atom_coords(ligand_sdf)
    rec_coords = _pdb_heavy_atom_coords(receptor_pdb)

    lig_tree = cKDTree(lig_coords) if lig_coords.size > 0 else None
    rec_tree = cKDTree(rec_coords) if rec_coords.size > 0 else None

    # --- Docking box bounds ---
    cx, cy, cz = box.center
    hx, hy, hz = box.size[0] / 2.0, box.size[1] / 2.0, box.size[2] / 2.0

    surviving_lines: list[str] = []
    retained_ids: list[str] = []

    for res_id, lines, ow in groups:
        if ow is None:
            continue

        # 1. Inside box
        if not (cx - hx <= ow[0] <= cx + hx and cy - hy <= ow[1] <= cy + hy and cz - hz <= ow[2] <= cz + hz):
            continue

        # 2. Near ligand
        if lig_tree is not None:
            hits = lig_tree.query_ball_point(ow, r=ligand_cutoff_angstrom)
            if not hits:
                continue

        # 3. No receptor clash
        if rec_tree is not None:
            hits = rec_tree.query_ball_point(ow, r=receptor_clash_cutoff_angstrom)
            if hits:
                LOGGER.debug(
                    "Crystal water %s discarded: clash with receptor (%.2f A cutoff).",
                    res_id,
                    receptor_clash_cutoff_angstrom,
                )
                continue

        surviving_lines.extend(lines)
        retained_ids.append(res_id)

    if not surviving_lines:
        LOGGER.debug("No crystal waters survived selection (box + ligand proximity + clash filter).")
        return None, []

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(surviving_lines) + "\nEND\n", encoding="utf-8")
    LOGGER.info("Selected %d crystal waters for docking: %s", len(retained_ids), retained_ids)
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
    water whose oxygen is within ``bridge_min``-``bridge_max`` A of BOTH a
    ligand heavy atom AND a receptor heavy atom simultaneously.
    """
    lig_coords = _pose_heavy_atom_coords(pose_sdf)
    rec_coords = _pdb_heavy_atom_coords(receptor_pdb)

    lig_protein_clashes: list[tuple[str, float]] = []
    lig_water_clashes: list[tuple[str, float]] = []
    water_protein_clashes: list[tuple[str, float]] = []
    water_bridges: list[tuple[str, str, str, float, float]] = []

    rec_tree = cKDTree(rec_coords) if rec_coords.size > 0 else None

    # Ligand-protein clashes
    if lig_coords.size > 0 and rec_tree is not None:
        for i, lpos in enumerate(lig_coords):
            hits = rec_tree.query_ball_point(lpos, r=clash_cutoff_angstrom)
            for j in hits:
                d = float(np.linalg.norm(lpos - rec_coords[j]))
                lig_protein_clashes.append((f"lig_atom{i}<->rec_atom{j}", d))

    # Crystal water checks
    if retained_waters_pdb is not None and retained_waters_pdb.exists():
        water_groups = _group_water_residues(retained_waters_pdb)
        lig_tree = cKDTree(lig_coords) if lig_coords.size > 0 else None

        for res_id, _lines, ow in water_groups:
            if ow is None:
                continue

            # Water-protein clashes
            if rec_tree is not None:
                hits = rec_tree.query_ball_point(ow, r=clash_cutoff_angstrom)
                for j in hits:
                    d = float(np.linalg.norm(np.array(ow) - rec_coords[j]))
                    water_protein_clashes.append((f"HOH{res_id}<->rec_atom{j}", d))

            # Ligand-water clashes
            if lig_tree is not None:
                hits = lig_tree.query_ball_point(ow, r=clash_cutoff_angstrom)
                for i in hits:
                    d = float(np.linalg.norm(np.array(ow) - lig_coords[i]))
                    lig_water_clashes.append((f"HOH{res_id}<->lig_atom{i}", d))

            # Water bridges: OW within bridge range of BOTH ligand and protein
            if lig_tree is not None and rec_tree is not None:
                near_lig = lig_tree.query_ball_point(ow, r=bridge_max_angstrom)
                near_rec = rec_tree.query_ball_point(ow, r=bridge_max_angstrom)
                for i in near_lig:
                    dl = float(np.linalg.norm(np.array(ow) - lig_coords[i]))
                    if dl < bridge_min_angstrom:
                        continue
                    for j in near_rec:
                        dr = float(np.linalg.norm(np.array(ow) - rec_coords[j]))
                        if dr < bridge_min_angstrom:
                            continue
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
    the receptor.  Set ``use_crystal_waters=False`` to skip the second run
    entirely (useful for debugging MD stability without crystal water influence).

    The receptor in ``request`` must be a ``.pdb`` file so the water-augmented
    version can be assembled.  Both docked poses are validated post-hoc for
    ligand/protein/water clashes and water bridges.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    def _best_score(result: DockingResult) -> float | None:
        return next((p.score for p in result.poses if p.score is not None), None)

    def _best_pose_sdf(result: DockingResult) -> Path | None:
        best = next((p for p in result.poses if p.rank == 1), None)
        return best.pose_path if best is not None else None

    # --- Run 1: no crystal waters -------------------------------------------
    run1_dir = work_dir / "no_water"
    request_no_water = request.model_copy(update={"workdir": run1_dir})
    result_no_water = engine.dock(request_no_water)
    score_no_water = _best_score(result_no_water)
    pose_no_water = _best_pose_sdf(result_no_water)
    val_no_water = validate_docked_pose(pose_no_water, request.receptor) if pose_no_water is not None else None

    # --- Select crystal waters ----------------------------------------------
    def _early_return(val: DockingValidation | None) -> DockingManifest:
        return DockingManifest(
            without_waters=result_no_water,
            with_waters=None,
            score_without=score_no_water,
            score_with=None,
            retained_water_ids=[],
            receptor_without_waters=request.receptor,
            receptor_with_waters=None,
            validation_without=val,
            validation_with=None,
        )

    if not use_crystal_waters or crystal_waters_pdb is None or not crystal_waters_pdb.exists():
        return _early_return(val_no_water)

    if request.receptor.suffix.lower() != ".pdb":
        LOGGER.warning("dock_with_and_without_crystal_waters: receptor must be a .pdb file; skipping water run.")
        return _early_return(val_no_water)

    # Use the docked pose from Run 1 (PDBQT) for crystal water proximity filter.
    ligand_ref = pose_no_water if (pose_no_water is not None and pose_no_water.exists()) else ligand_sdf

    selected_pdb = work_dir / "selected_crystal_waters.pdb"
    selected_path, retained_ids = select_docking_crystal_waters(
        crystal_waters_pdb,
        request.box,
        ligand_ref,
        request.receptor,
        selected_pdb,
        ligand_cutoff_angstrom=ligand_cutoff_angstrom,
        receptor_clash_cutoff_angstrom=receptor_clash_cutoff_angstrom,
    )

    if selected_path is None:
        return _early_return(val_no_water)

    # --- Run 2: receptor + selected crystal waters --------------------------
    receptor_with_waters = work_dir / "receptor_with_crystal_waters.pdb"
    prepare_receptor_with_crystal_waters(request.receptor, selected_path, receptor_with_waters)

    run2_dir = work_dir / "with_water"
    request_with_waters = request.model_copy(update={"receptor": receptor_with_waters, "workdir": run2_dir})
    result_with_water = engine.dock(request_with_waters)
    score_with_water = _best_score(result_with_water)
    pose_with_water = _best_pose_sdf(result_with_water)
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
