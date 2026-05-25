# /home/grheco/repositorios/gbsa-pipeline/src/gbsa_pipeline/docking.py

"""Docking helpers for preparing ligands and running AutoDock Vina.

This module is intentionally small and centered on one workflow: take a ligand
and receptor, prepare the ligand to PDBQT, optionally prepare the receptor to
PDBQT, run Vina, and optionally export or reconstruct the resulting ligand.
The design tries to keep chemistry restoration separate from docking execution
so failures can be debugged stage by stage instead of through one large wrapper.
The current public surface is therefore narrow on purpose and avoids native-
ligand redocking or box-derivation logic that belongs to a different workflow.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdMolTransforms
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit.Geometry.rdGeometry import Point3D

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rdkit.Geometry.rdGeometry import Point3D


LOGGER = logging.getLogger(__name__)

VINA_SCORE_COLUMN_COUNT = 2
VINA_RANK_COLUMN_INDEX = 0
VINA_SCORE_COLUMN_INDEX = 1
VINA_TOP_RANK = "1"


class DockingBox(BaseModel):
    """Docking-box center and size in Angstrom.

    This model exists so Vina box inputs stay explicit and typed instead of
    being passed around as anonymous tuples or dictionaries.
    The box is required because the docking engine cannot run without a search
    region, and making it a model helps catch malformed inputs early.
    We are currently storing only center and size because that is the minimum
    stable interface needed by the Vina command-line backend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    center: tuple[float, float, float]
    size: tuple[float, float, float]


class DockingRequest(BaseModel):
    """Normalized docking request for a receptor, one or more ligands, and a box.

    This model groups the core docking inputs into one validated object so the
    engine layer receives a stable, explicit contract instead of loosely coupled
    parameters.
    The receptor, ligands, and box are required because they define the actual
    docking problem, while seed, workdir, and parameters control runtime details.
    We are currently checking mostly filesystem validity and suffix support here,
    leaving chemistry-specific validation to the preparation and export helpers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    receptor: Path
    ligands: list[Path]
    box: DockingBox
    seed: int | None = None
    workdir: Path | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("receptor")
    @classmethod
    def _check_receptor_exists(cls, path: Path) -> Path:
        """Validate that the receptor exists and has a Vina-compatible suffix.

        The `path` parameter is needed because the docking engine supports both
        prebuilt receptor PDBQT files and plain PDB files that still need
        conversion.
        We are currently checking only file existence, file-ness, and suffix,
        because structural receptor correctness belongs to external preparation
        steps rather than to this thin adapter layer.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise ValueError(f"Receptor file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Receptor path is not a file: {path}")

        if path.suffix.lower() not in {".pdb", ".pdbqt"}:
            raise ValueError(f"Receptor must be .pdb or .pdbqt for vina docking: {path}")

        return path

    @field_validator("ligands")
    @classmethod
    def _check_ligands(cls, ligands: list[Path]) -> list[Path]:
        """Validate that one or more ligand files are present on disk.

        The `ligands` parameter is a list because the current engine API is
        allowed to dock multiple prepared ligands against one receptor request.
        We are currently checking that each path exists and is a file, but not
        whether each file is chemically valid, because that belongs to ligand
        preparation and downstream docking itself.
        """
        if not ligands:
            raise ValueError("At least one ligand file required.")

        checked: list[Path] = []

        for ligand_entry in ligands:
            ligand_path = Path(ligand_entry).resolve()

            if not ligand_path.exists():
                raise ValueError(f"Ligand file missing: {ligand_path}")

            if not ligand_path.is_file():
                raise ValueError(f"Ligand path is not a file: {ligand_path}")

            checked.append(ligand_path)

        return checked


@dataclass(frozen=True)
class DockedPose:
    """Single docked pose plus compact metadata about the docking run.

    This dataclass is intentionally small and stores only what downstream stages
    need immediately: which ligand was docked, where the pose file is, which
    score was parsed, and a compact metadata bundle.
    The metadata field exists because some run details are useful but not
    important enough to deserve top-level strongly typed fields yet.
    We are currently using this object as the bridge between docking execution
    and later export or reconstruction steps.
    """

    ligand: Path
    pose_path: Path
    score: float | None
    rank: int | None
    engine: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DockingResult:
    """Collection of produced poses and compact run metadata.

    This dataclass exists so one docking request returns a single structured
    object even when multiple ligands were processed in one call.
    The `parameters` field is retained because downstream code often needs to
    know with which runtime settings the poses were generated.
    We are currently keeping this result intentionally lightweight and not
    storing large parsed outputs, because the raw log files already exist on disk.
    """

    poses: list[DockedPose]
    engine: str
    parameters: Mapping[str, Any]
    raw_outputs: Mapping[str, Any] = field(default_factory=dict)


class DockingEngine(Protocol):
    """Protocol implemented by docking backends.

    This protocol keeps the rest of the codebase independent from one concrete
    docking backend implementation.
    The `dock()` signature is the minimal stable contract the runner or higher
    workflow code needs in order to call a docking engine generically.
    We are currently using only Vina, but this protocol avoids hard-coding that
    assumption into every consumer of the docking module.
    """

    name: str

    def dock(self, request: DockingRequest) -> DockingResult:
        """Run docking for one validated docking request.

        This method defines the minimal operation a docking backend must expose
        so the rest of the code can treat concrete engines uniformly.
        The `request` parameter is required because it bundles receptor, ligands,
        box, work directory, and runtime parameters into one validated object.
        We are currently requiring only a structured `DockingResult` return value,
        because downstream workflow code should not need to parse backend-specific
        subprocess results directly.
        """


def load_first_sdf_molecule(path: Path, *, remove_hs: bool = False) -> Chem.Mol:
    """Read the first valid molecule from an SDF file.

    This helper exists so SDF loading behavior is centralized and consistent
    instead of being rewritten at multiple call sites with slightly different
    failure behavior.
    The `path` parameter is required because this function is specifically about
    reading one file from disk, while `remove_hs` controls whether RDKit should
    strip hydrogens during import.
    We are currently checking only that the file exists, is a file, and yields
    a non-None first molecule, because this module expects single-ligand SDF
    inputs rather than arbitrary multi-record chemistry archives.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"SDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"SDF path is not a file: {path}")

    supplier = Chem.SDMolSupplier(str(path), removeHs=remove_hs)
    molecule = supplier[0]

    if molecule is None:
        raise ValueError(f"Could not read first molecule from SDF: {path}")

    return molecule


def remove_hydrogens_copy(molecule: Chem.Mol) -> Chem.Mol:
    """Return a copy of a molecule with hydrogens removed.

    This helper exists because pose and chemistry comparisons are often more
    stable on the heavy-atom graph than on a hydrogen-complete representation.
    The `molecule` parameter is required because callers may want to normalize
    molecules coming from templates, raw exports, or rebuilt structures in the
    same way before comparison.
    We are currently using RDKit's hydrogen removal on a copied molecule so the
    original input object stays unchanged, which is important when the same
    molecule is reused later for export or debugging.
    """
    return Chem.RemoveHs(Chem.Mol(molecule))


def molecule_centroid(
    molecule: Chem.Mol,
    *,
    conf_id: int = -1,
    ignore_hs: bool = False,
) -> Point3D:
    """Compute the geometric centroid of one molecular conformer.

    This helper exists because pose-comparison tests need a compact spatial
    summary without introducing a separate alignment step.
    The `molecule` parameter provides the coordinates, while `conf_id` allows
    callers to select a specific conformer when needed.
    We validate the requested conformer before computing the centroid so callers
    get a clear error when they ask for a conformer that is not present.
    """
    conformer_ids = {conformer.GetId() for conformer in molecule.GetConformers()}

    if not conformer_ids:
        raise ValueError("Molecule has no conformer.")

    if conf_id != -1 and conf_id not in conformer_ids:
        raise ValueError(
            f"Requested conformer id {conf_id} is not present. Available conformer ids: {sorted(conformer_ids)}"
        )

    return rdMolTransforms.ComputeCentroid(
        molecule.GetConformer(conf_id),
        ignoreHs=ignore_hs,
    )


def _summarize_stderr(stderr: str, max_lines: int = 4) -> str:
    """Return a short preview of stderr suitable for terminal output.

    This helper exists because full subprocess stderr is still written to log
    files, but a compact preview is more useful for immediate terminal feedback.
    The `stderr` parameter is required because multiple external tools in this
    module can fail, and the same summarization logic should apply to all of them.
    We are currently checking only a few leading non-empty lines because this
    function is meant for human scanability, not for archival completeness.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "no stderr output"

    preview = lines[:max_lines]
    text = " | ".join(preview)

    if len(lines) > max_lines:
        text += " | ..."

    return text


def _write_process_log(
    log_path: Path,
    process: CompletedProcess[str],
    *,
    command: list[str],
    title: str,
) -> None:
    """Write complete subprocess stdout and stderr to a plain-text log file.

    This helper is deliberately separate from `_summarize_stderr()` because the
    module needs both a short terminal summary and a complete on-disk record.
    The `log_path`, `process`, `command`, and `title` parameters are all needed
    to produce a log file that is self-contained and reviewable later.
    We are currently writing command, return code, stdout, and stderr verbatim
    because that is the minimum useful forensic record for external tool failures.
    """
    resolved_log_path = Path(log_path).resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    text = (
        f"{title}\n"
        f"{'=' * len(title)}\n\n"
        f"Command:\n{' '.join(command)}\n\n"
        f"Return code:\n{process.returncode}\n\n"
        f"STDOUT\n"
        f"------\n"
        f"{process.stdout or ''}\n\n"
        f"STDERR\n"
        f"------\n"
        f"{process.stderr or ''}\n"
    )
    resolved_log_path.write_text(text, encoding="utf-8")


def _extract_pdbqt_string_from_meeko_result(result: Any) -> str:
    """Extract the PDBQT string from Meeko's version-dependent return value.

    This helper exists because Meeko's `write_string()` return shape is not
    always uniform across versions or calling contexts.
    The `result` parameter is intentionally typed as `Any` because the function
    is explicitly about normalizing a loosely specified third-party return value.
    We are currently checking only the cases needed by the observed Meeko API:
    a plain string or a tuple whose first element is the PDBQT string.
    """
    if isinstance(result, str):
        return result

    if isinstance(result, tuple):
        if not result:
            raise ValueError("Meeko returned an empty tuple from write_string().")

        pdbqt_string = result[0]

        if not isinstance(pdbqt_string, str):
            raise TypeError(
                f"Expected first element of Meeko write_string() result to be a str, got {type(result[0]).__name__}."
            )

        return pdbqt_string

    raise TypeError(f"Unexpected return type from Meeko write_string(): {type(result).__name__}")


def _parse_vina_best_score_from_log(log_path: Path) -> float | None:
    """Parse the best affinity from the written Vina log file.

    This helper exists because docking output is already persisted to disk and
    downstream parsing should use that durable record rather than in-memory
    subprocess stdout.
    The `log_path` parameter is required because the score now comes from the
    archived process log written by `_write_process_log()`.
    We are currently checking only the first-ranked pose because the current
    adapter exposes a minimal top-pose view instead of a full ranked table.

    Reference:
    https://autodock-vina.readthedocs.io/en/latest/
    """
    resolved_log_path = Path(log_path).resolve()

    if not resolved_log_path.exists():
        return None

    log_text = resolved_log_path.read_text(encoding="utf-8")

    for line in log_text.splitlines():
        columns = line.strip().split()

        if len(columns) < VINA_SCORE_COLUMN_COUNT:
            continue

        if columns[VINA_RANK_COLUMN_INDEX] != VINA_TOP_RANK:
            continue

        try:
            return float(columns[VINA_SCORE_COLUMN_INDEX])
        except ValueError:
            continue

    return None


def _build_compact_pose_metadata(
    *,
    returncode: int,
    log_file: Path,
    output_exists: bool,
    receptor_used: Path,
) -> dict[str, Any]:
    """Build minimal per-pose metadata.

    This helper exists to keep the metadata payload uniform wherever poses are
    created inside the docking engine.
    The parameters are intentionally explicit because they are the smallest set
    of run facts that are still useful for debugging and downstream checks.
    We are currently storing filesystem- and process-level facts only, not
    chemistry-level annotations, because those belong elsewhere in the pipeline.
    """
    return {
        "returncode": returncode,
        "log_file": str(Path(log_file).resolve()),
        "output_exists": output_exists,
        "receptor_used": str(Path(receptor_used).resolve()),
    }


def assign_bond_orders_from_template_mol(
    template_mol: Chem.Mol,
    target_mol: Chem.Mol,
    *,
    add_hydrogens: bool = False,
) -> Chem.Mol:
    """Repair target bond orders from a template while preserving target geometry.

    This is the central chemistry-restoration function in the module, and it is
    intentionally separate from raw export so chemistry and coordinates can be
    reasoned about independently.
    The `template_mol` parameter is required because it is the trusted source of
    bond orders, while `target_mol` is required because it carries the docked
    pose geometry that should remain attached to the repaired molecule.
    We are currently relying on RDKit's `AssignBondOrdersFromTemplate()` to
    repair bond orders directly on the docked heavy-atom graph, then inferring
    stereochemistry from the existing 3D coordinates and optionally re-adding
    hydrogens afterward.

    Reference:
    https://www.rdkit.org/docs/source/rdkit.Chem.AllChem.html
    """
    if template_mol is None:
        raise ValueError("template_mol is None.")

    if target_mol is None:
        raise ValueError("target_mol is None.")

    template_no_h = Chem.RemoveHs(Chem.Mol(template_mol))
    target_no_h = Chem.RemoveHs(Chem.Mol(target_mol))

    try:
        rebuilt = AllChem.AssignBondOrdersFromTemplate(template_no_h, target_no_h)
    except Exception as exc:
        raise RuntimeError(
            "RDKit AssignBondOrdersFromTemplate failed. Template and target molecule likely do not match."
        ) from exc

    rdmolops.AssignStereochemistryFrom3D(rebuilt)

    if add_hydrogens:
        rebuilt = Chem.AddHs(rebuilt, addCoords=True)

    return rebuilt


def export_pdbqt_to_sdf(
    pdbqt_path: Path,
    output_sdf: Path,
    *,
    template_mol: Chem.Mol | None = None,
    add_hydrogens_after_template: bool = True,
) -> Path:
    """Export a PDBQT file to SDF, optionally rebuilding bond orders from a template.

    This function combines two related but distinct steps: raw PDBQT-to-SDF
    export and optional chemistry reconstruction from a trusted template.
    The `pdbqt_path` and `output_sdf` parameters define the file conversion,
    while `template_mol` determines whether bond-order repair should be applied
    and `add_hydrogens_after_template` controls hydrogen re-addition afterward.
    We are currently using the presence of `template_mol` as the single switch
    for template-based repair, because a separate boolean flag would add a
    redundant and potentially contradictory state to this API.
    """
    pdbqt_path = Path(pdbqt_path).resolve()
    output_sdf = Path(output_sdf).resolve()
    output_sdf.parent.mkdir(parents=True, exist_ok=True)

    if not pdbqt_path.exists():
        raise FileNotFoundError(f"PDBQT file not found: {pdbqt_path}")

    if not pdbqt_path.is_file():
        raise ValueError(f"PDBQT path is not a file: {pdbqt_path}")

    pdbqt_molecule = PDBQTMolecule.from_file(str(pdbqt_path), skip_typing=True)
    raw_molecules = RDKitMolCreate.from_pdbqt_mol(pdbqt_molecule)
    valid_raw_molecules = [mol for mol in raw_molecules if mol is not None]

    if not valid_raw_molecules:
        raise RuntimeError(f"Meeko could not reconstruct any molecules from docking output.\nPDBQT: {pdbqt_path}")

    output_molecules: list[Chem.Mol] = []

    for raw_mol in valid_raw_molecules:
        output_mol = raw_mol

        if template_mol is not None:
            output_mol = assign_bond_orders_from_template_mol(
                template_mol=template_mol,
                target_mol=raw_mol,
                add_hydrogens=add_hydrogens_after_template,
            )

            if raw_mol.HasProp("_Name"):
                output_mol.SetProp("_Name", raw_mol.GetProp("_Name"))

        output_molecules.append(output_mol)

    writer = Chem.SDWriter(str(output_sdf))
    try:
        for output_mol in output_molecules:
            writer.write(output_mol)
    finally:
        writer.close()

    if not output_sdf.exists():
        raise RuntimeError(f"Meeko reported success but expected SDF output is missing.\nExpected: {output_sdf}")

    return output_sdf


def convert_receptor_pdb_to_pdbqt(
    receptor_pdb: Path,
    output_path: Path | None = None,
    *,
    mk_prepare_receptor_binary: str = "mk_prepare_receptor.py",
) -> Path:
    """Convert a receptor PDB to rigid receptor PDBQT using Meeko.

    This helper exists because docking often starts from a receptor PDB even
    when Vina requires a receptor PDBQT for execution.
    The `mk_prepare_receptor_binary` parameter is configurable because different
    environments may expose Meeko command-line tools through wrappers or explicit
    executable names.
    We currently use Meeko's `--read_pdb` path for plain PDB input so this helper
    does not require ProDy for the simple rigid-receptor workflow tested here.
    Receptor hydrogen addition, protonation-state decisions, and structural
    cleanup are still expected to happen upstream.
    """
    receptor_pdb = Path(receptor_pdb).resolve()

    if not receptor_pdb.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {receptor_pdb}")

    if not receptor_pdb.is_file():
        raise ValueError(f"Receptor path is not a file: {receptor_pdb}")

    if receptor_pdb.suffix.lower() != ".pdb":
        raise ValueError(f"Expected a .pdb receptor input, got: {receptor_pdb}")

    if shutil.which(mk_prepare_receptor_binary) is None:
        raise RuntimeError(f"Meeko receptor executable not found in PATH: {mk_prepare_receptor_binary}")

    if output_path is None:
        output_path = receptor_pdb.with_suffix(".pdbqt")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_base = output_path.with_suffix("")
    log_path = output_path.with_suffix(".meeko_receptor.log")

    cmd = [
        mk_prepare_receptor_binary,
        "--read_pdb",
        str(receptor_pdb),
        "-o",
        str(output_base),
        "-p",
    ]

    LOGGER.info(
        "Preparing receptor with Meeko: %s -> %s",
        receptor_pdb.name,
        output_path.name,
    )

    process: CompletedProcess[str] = run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    _write_process_log(
        log_path,
        process,
        command=cmd,
        title=f"Meeko receptor preparation log for {receptor_pdb.name}",
    )

    if process.returncode != 0:
        raise RuntimeError(
            "Meeko receptor preparation failed.\n"
            f"Receptor: {receptor_pdb}\n"
            f"Log: {log_path}\n"
            f"stderr summary: {_summarize_stderr(process.stderr)}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"Meeko reported success but receptor PDBQT output is missing.\nExpected: {output_path}\nLog: {log_path}"
        )

    if process.stderr.strip():
        LOGGER.warning(
            "Meeko receptor preparation finished with warnings; full details in %s",
            log_path.name,
        )
    else:
        LOGGER.info("Meeko receptor PDBQT written: %s", output_path.name)

    return output_path


def prepare_ligand_with_meeko(
    ligand: str | Chem.Mol,
    output_path: Path,
    name: str | None = None,
) -> Path:
    """Prepare a SMILES string or RDKit molecule as ligand PDBQT.

    This helper exists so ligand preparation is handled in one place and the
    rest of the module can assume prepared PDBQT ligand inputs when needed.
    The `ligand` parameter accepts either a SMILES string or an RDKit molecule
    because those are the two explicit input types chosen for this minimal API.
    Meeko expects an RDKit molecule with explicit hydrogens and 3D coordinates,
    so this function prepares those prerequisites before calling the current
    documented Meeko Python API.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preparing ligand with Meeko: %s", output_path.name)

    if isinstance(ligand, str):
        mol = Chem.MolFromSmiles(ligand)
        if mol is None:
            raise ValueError("Failed to parse SMILES.")

        mol.SetProp("_Name", name or "LIG")
        mol = Chem.AddHs(mol)

        embed_status = EmbedMolecule(mol)
        if embed_status != 0:
            raise RuntimeError(f"RDKit failed to embed 3D coordinates for ligand: {ligand}")

        optimize_status = UFFOptimizeMolecule(mol)
        if optimize_status not in (0, 1):
            LOGGER.warning(
                "UFF optimization returned non-standard status %s",
                optimize_status,
            )

    elif isinstance(ligand, Chem.Mol):
        mol = Chem.Mol(ligand)

        if mol.GetNumConformers() == 0:
            raise ValueError(
                "Chem.Mol input must contain at least one conformer. "
                "Use a SMILES string input to generate 3D coordinates automatically."
            )

        if name is not None:
            mol.SetProp("_Name", name)
        elif not mol.HasProp("_Name"):
            mol.SetProp("_Name", "LIG")

        mol = Chem.AddHs(mol, addCoords=True)

    else:
        raise TypeError("prepare_ligand_with_meeko supports only SMILES strings and RDKit Chem.Mol objects.")

    preparator = MoleculePreparation()
    mol_setups = preparator(mol)

    if not mol_setups:
        raise RuntimeError("Meeko produced no molecule setups.")

    meeko_result = PDBQTWriterLegacy.write_string(mol_setups[0])
    pdbqt_string = _extract_pdbqt_string_from_meeko_result(meeko_result)

    if not pdbqt_string.strip():
        raise RuntimeError("Generated ligand PDBQT string is empty.")

    output_path.write_text(pdbqt_string, encoding="utf-8")
    LOGGER.info("Ligand PDBQT written: %s", output_path.name)

    return output_path


_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "TIP3", "TIP3P", "SOL"})


# ---------------------------------------------------------------------------
# Crystal-water docking: data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DockingValidation:
    """Post-docking geometry checks for one docked pose.

    All distances are in Å.  Each entry in the clash lists is a
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
# Crystal-water docking: helpers
# ---------------------------------------------------------------------------


def _pdb_heavy_atom_coords(
    pdb_path: Path,
    *,
    exclude_residues: frozenset[str] = _WATER_RESNAMES,
) -> np.ndarray:
    """Return (N, 3) array of non-hydrogen, non-water PDB atom coordinates in Å."""
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
    """Return (N, 3) array of heavy-atom coordinates in Å from the first SDF conformer."""
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return np.empty((0, 3))
    conf = mol.GetConformer(0)
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def _pose_heavy_atom_coords(pose_path: Path) -> np.ndarray:
    """Return (N, 3) heavy-atom coordinates for a docked pose in Å.

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


class VinaEngine:
    """Minimal AutoDock Vina wrapper.

    This class deliberately stays small and only covers what is needed to turn
    a validated request into one or more docked pose files plus compact metadata.
    It does not own ligand preparation, chemistry reconstruction, or box
    generation, because those are separate concerns in this module.
    We are currently keeping the engine narrowly focused on command construction,
    receptor readiness, subprocess execution, score parsing, and result assembly.
    """

    name = "vina"

    def __init__(
        self,
        binary: str = "vina",
        mk_prepare_receptor_binary: str = "mk_prepare_receptor.py",
    ) -> None:
        """Initialize executable names for Vina and Meeko receptor preparation.

        The binary names are configurable because development and CI environments
        often expose the tools under different names or wrappers.
        Both parameters are required to keep receptor preparation and docking
        execution under the same engine object without hard-coding global paths.
        We are currently checking only whether the executables appear in PATH and
        leaving actual runtime validity to the receptor-preparation and docking calls.
        """
        if shutil.which(binary) is None:
            LOGGER.warning("Vina binary not found in PATH: %s", binary)

        if shutil.which(mk_prepare_receptor_binary) is None:
            LOGGER.warning(
                "Meeko receptor preparation binary not found in PATH: %s",
                mk_prepare_receptor_binary,
            )

        self.binary = binary
        self.mk_prepare_receptor_binary = mk_prepare_receptor_binary

    def _build_command(
        self,
        *,
        receptor: Path,
        ligand: Path,
        output: Path,
        box: DockingBox,
        seed: int | None = None,
        num_modes: int | None = None,
        exhaustiveness: int | None = None,
        energy_range: float | None = None,
        extra_flags: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Build the Vina command line for a single ligand.

        This helper exists so command construction is isolated from the rest of
        the execution logic and can be tested directly.
        The receptor, ligand, output, and box parameters are the minimal Vina
        inputs, while the optional parameters expose common runtime controls.
        We are currently checking only the flags needed by the present workflow
        and passing any extra flags through transparently rather than validating them.

        Reference:
        https://autodock-vina.readthedocs.io/en/latest/docking_basic.html
        """
        cmd: list[str] = [
            self.binary,
            "--receptor",
            str(Path(receptor).resolve()),
            "--ligand",
            str(Path(ligand).resolve()),
            "--center_x",
            str(box.center[0]),
            "--center_y",
            str(box.center[1]),
            "--center_z",
            str(box.center[2]),
            "--size_x",
            str(box.size[0]),
            "--size_y",
            str(box.size[1]),
            "--size_z",
            str(box.size[2]),
            "--out",
            str(Path(output).resolve()),
        ]

        if seed is not None:
            cmd += ["--seed", str(seed)]

        if num_modes is not None:
            cmd += ["--num_modes", str(num_modes)]

        if exhaustiveness is not None:
            cmd += ["--exhaustiveness", str(exhaustiveness)]

        if energy_range is not None:
            cmd += ["--energy_range", str(energy_range)]

        if extra_flags:
            for key, value in extra_flags.items():
                cmd.append(str(key))
                if value is not None:
                    cmd.append(str(value))

        return cmd

    def _prepare_receptor_for_docking(
        self,
        receptor: Path,
        workdir: Path,
    ) -> Path:
        """Return the receptor PDBQT to use for docking.

        This helper exists so the engine can accept either a preprepared PDBQT
        receptor or a plain PDB receptor that still needs conversion.
        The `receptor` parameter is the user-provided input, while `workdir` is
        required because on-the-fly receptor preparation needs a destination path.
        We are currently checking only file-type routing here: use PDBQT as-is,
        prepare PDB via Meeko, and reject anything else explicitly.
        """
        receptor = Path(receptor).resolve()

        if receptor.suffix.lower() == ".pdbqt":
            return receptor

        if receptor.suffix.lower() == ".pdb":
            return convert_receptor_pdb_to_pdbqt(
                receptor,
                output_path=workdir / f"{receptor.stem}.pdbqt",
                mk_prepare_receptor_binary=self.mk_prepare_receptor_binary,
            )

        raise ValueError(f"Unsupported receptor file type for docking: {receptor}")

    def dock(self, request: DockingRequest) -> DockingResult:
        """Run docking for all ligands in the request.

        This is the one method that actually executes the backend and turns a
        validated request into structured pose outputs.
        The `request` parameter is required because it carries the complete
        docking contract: receptor, ligands, box, working directory, and runtime
        options in one validated object.
        We are currently checking a minimal but useful workflow: prepare the
        receptor if needed, run Vina ligand by ligand, parse the top score from
        the persisted log file, write full logs, and return structured pose records.
        """
        workdir = request.workdir or Path.cwd()
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        receptor_for_docking = self._prepare_receptor_for_docking(
            request.receptor,
            workdir,
        )

        poses: list[DockedPose] = []
        raw_outputs: dict[str, Any] = {}

        for ligand_entry in request.ligands:
            ligand_path = Path(ligand_entry).resolve()
            out_file = workdir / f"{ligand_path.stem}_vina_out.pdbqt"
            log_file = workdir / f"{ligand_path.stem}_vina.log"

            cmd = self._build_command(
                receptor=receptor_for_docking,
                ligand=ligand_path,
                output=out_file,
                box=request.box,
                seed=request.seed,
                num_modes=request.parameters.get("num_modes"),
                exhaustiveness=request.parameters.get("exhaustiveness"),
                energy_range=request.parameters.get("energy_range"),
                extra_flags=request.parameters.get("extra_flags"),
            )

            process: CompletedProcess[str] = run(  # noqa: S603
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
            )

            _write_process_log(
                log_file,
                process,
                command=cmd,
                title=f"Vina docking log for {ligand_path.name}",
            )

            score = _parse_vina_best_score_from_log(log_file)
            rank = 1 if score is not None else None

            pose_metadata = _build_compact_pose_metadata(
                returncode=process.returncode,
                log_file=log_file,
                output_exists=out_file.exists(),
                receptor_used=receptor_for_docking,
            )

            poses.append(
                DockedPose(
                    ligand=ligand_path,
                    pose_path=out_file,
                    score=score,
                    rank=rank,
                    engine=self.name,
                    metadata=pose_metadata,
                )
            )

            raw_outputs[str(ligand_path)] = pose_metadata

            if process.returncode != 0:
                LOGGER.error(
                    "Vina failed for %s: return code %s; stderr summary: %s",
                    ligand_path.name,
                    process.returncode,
                    _summarize_stderr(process.stderr),
                )
            elif not out_file.exists():
                LOGGER.warning(
                    "Vina returned success but output is missing for %s",
                    ligand_path.name,
                )

        return DockingResult(
            poses=poses,
            engine=self.name,
            parameters=request.parameters,
            raw_outputs=raw_outputs,
        )
