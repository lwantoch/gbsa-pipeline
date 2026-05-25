"""Parametrize protein-ligand complexes."""

from __future__ import annotations

import contextlib
import logging
import pickle
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import BioSimSpace as BSS
import parmed as pmd
from openff.toolkit.topology import Molecule
from openmm.app import ForceField, Modeller, NoCutoff, PDBFile
from openmmforcefields.generators import GAFFTemplateGenerator
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from BioSimSpace._SireWrappers import System

from gbsa_pipeline.parametrization_enum import ChargeMethod, LigandFF, ProteinFF

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Force field configuration
# ---------------------------------------------------------------------------


class ParametrizationConfig(BaseModel):
    """Force field and charge method choices for a parametrization run.

    Defaults to AMBER ff14SB + GAFF2 + AM1-BCC.
    Use the class-method presets for the most common combinations, or
    construct directly to override individual axes.

    Examples:
    --------
    >>> ParametrizationConfig()  # all defaults
    >>> ParametrizationConfig(protein_ff=ProteinFF.FF19SB)  # swap protein FF
    >>> ParametrizationConfig.amber14_gaff2_nagl()  # preset with NAGL charges
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    protein_ff: ProteinFF = ProteinFF.FF14SB
    ligand_ff: LigandFF = LigandFF.GAFF2
    charge_method: ChargeMethod = ChargeMethod.AM1BCC
    extra_ff_files: tuple[Path, ...] = ()

    @field_validator("extra_ff_files", mode="before")
    @classmethod
    def _check_extra_ff_files(cls, paths: Any) -> tuple[Path, ...]:
        result = tuple(Path(p) for p in paths)
        missing = [p for p in result if not p.exists()]
        if missing:
            raise ValueError("Extra force field files not found: " + ", ".join(str(p) for p in missing))
        return result

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def amber14_gaff2(cls) -> ParametrizationConfig:
        """AMBER ff14SB + GAFF2 + AM1-BCC charges (default)."""
        return cls(protein_ff=ProteinFF.FF14SB, charge_method=ChargeMethod.AM1BCC)

    @classmethod
    def amber19_gaff2(cls) -> ParametrizationConfig:
        """AMBER ff19SB + GAFF2 + AM1-BCC charges."""
        return cls(protein_ff=ProteinFF.FF19SB, charge_method=ChargeMethod.AM1BCC)

    @classmethod
    def amber14_gaff2_nagl(cls) -> ParametrizationConfig:
        """AMBER ff14SB + GAFF2 + NAGL graph-neural-network charges."""
        return cls(charge_method=ChargeMethod.NAGL)


# ---------------------------------------------------------------------------
# User-facing input model
# ---------------------------------------------------------------------------


class ParametrizationInput(BaseModel):
    """Validated inputs for a parametrization run.

    Parameters
    ----------
    protein_pdb:
        Path to the protein PDB file. Must exist.
    ligand_sdf:
        Path to the ligand SDF file with embedded 3-D coordinates. Must exist.
    config:
        Force field and charge method selection. Defaults to
        ``ParametrizationConfig()`` (ff14SB + GAFF2 + AM1-BCC).
    net_charge:
        Formal charge of the ligand in elementary charge units.
        ``None`` lets the charge assignment toolkit determine it automatically.
    work_dir:
        Directory where intermediate and output files are written.
        When ``None`` a temporary directory is created automatically.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    protein_pdb: Path
    ligand_sdf: Path
    config: ParametrizationConfig = Field(default_factory=ParametrizationConfig)
    net_charge: int | None = None
    work_dir: Path | None = None

    @field_validator("protein_pdb", "ligand_sdf")
    @classmethod
    def _check_exists(cls, path: Path) -> Path:
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        return path


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParametrisedComplex:
    """Parametrised protein-ligand complex ready for solvation and MD.

    Attributes:
    ----------
    gro_file:
        GROMACS coordinate file (.gro) produced by ParmEd.
    top_file:
        GROMACS topology file (.top) produced by ParmEd.
    config:
        The force field configuration used to produce this complex.
        Stored so that downstream steps can record or reproduce the run.
    forcefield:
        The OpenMM ``ForceField`` with the protein and ligand template
        generator registered. It is passed downstream to solvation, where the
        selected bulk-water XML can be added before generating the final
        solvated system. ``None`` when the complex was loaded from disk.
    parmed_structure:
        The ParmEd ``Structure`` holding the dry protein-ligand force field
        parameters produced during parametrization. It is passed directly to
        :func:`~gbsa_pipeline.solvation_openmm.solvate_openmm` to avoid
        reloading from the GROMACS files. ``None`` when loaded from disk.
    crystal_waters_pdb:
        Optional PDB file containing crystallographic waters extracted from the
        protein input before OpenMM protein-ligand parametrization. The dry
        parametrized complex avoids HOH template failures, while the saved water
        file lets the solvation step restore those waters before adding bulk
        solvent. ``None`` means no crystallographic waters were found or the
        complex was loaded through a path that did not preserve them.
    """

    gro_file: Path
    top_file: Path
    config: ParametrizationConfig
    forcefield: Any = field(default=None, hash=False, compare=False, repr=False)
    parmed_structure: Any = field(default=None, hash=False, compare=False, repr=False)
    crystal_waters_pdb: Path | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parametrize(inp: ParametrizationInput) -> ParametrisedComplex:
    """Parametrize a protein-ligand complex without running tleap.

    Uses OpenMM force field XML files for the protein and
    ``GAFFTemplateGenerator`` for the ligand. Partial charges are assigned
    via the method specified in ``inp.config.charge_method`` (default:
    AM1-BCC via sqm). The system is exported to GROMACS ``.gro`` and
    ``.top`` files via ParmEd.

    Parameters
    ----------
    inp:
        Validated parametrization inputs including file paths, force field
        configuration, and optional work directory.

    Returns:
    -------
    ParametrisedComplex
        Frozen dataclass holding the paths to the output GROMACS files and
        the configuration used.
    """
    return _parametrize_openmm(inp)


# ---------------------------------------------------------------------------
# OpenMM implementation
# ---------------------------------------------------------------------------

# OpenMM XML files (shipped with OpenMM / AmberTools) for each protein FF.
# ff14SB is bundled with OpenMM; ff19SB and ff99SB-ILDN require the XML
# files distributed with AmberTools 24+.
_PROTEIN_FF_XML: dict[ProteinFF, list[str]] = {
    ProteinFF.FF14SB: ["amber14-all.xml"],
    ProteinFF.FF19SB: ["amber/protein.ff19SB.xml"],
    ProteinFF.FF99SB: ["amber/protein.ff99SBildn.xml"],
}

# GAFF version strings accepted by GAFFTemplateGenerator.
# Verify against the version of openmmforcefields installed in your environment.
_GAFF_FF_VERSION: dict[LigandFF, str] = {
    LigandFF.GAFF: "gaff-1.81",
    LigandFF.GAFF2: "gaff-2.11",
}

_WATER_RESIDUE_NAMES = {"HOH", "WAT", "TIP3", "TIP3P", "SOL"}
_PDB_COORDINATE_RECORD_PREFIXES = ("ATOM  ", "HETATM")
_PDB_RESIDUE_NAME_START = 17
_PDB_RESIDUE_NAME_END = 20


def _is_crystal_water_pdb_record(line: str) -> bool:
    """Return whether a PDB coordinate record describes crystallographic water.

    The parametrization stage intentionally builds only the dry protein-ligand
    system because explicit solvent is added later by the solvation stage. This
    helper only inspects fixed-width PDB coordinate records, so it does not try
    to parse mmCIF or arbitrary whitespace-delimited text. It keeps water
    filtering close to PDB text handling instead of hiding it inside force-field
    setup. Non-coordinate records always return ``False`` so headers,
    connectivity records, and metadata do not leak into the extracted water file.
    """
    if not line.startswith(_PDB_COORDINATE_RECORD_PREFIXES):
        return False

    residue_name = line[_PDB_RESIDUE_NAME_START:_PDB_RESIDUE_NAME_END].strip().upper()
    return residue_name in _WATER_RESIDUE_NAMES


def _write_crystal_waters_pdb(protein_pdb: Path, output_pdb: Path) -> Path | None:
    """Write crystallographic water coordinate records to a separate PDB file.

    The generated file is an inspection and preservation artefact; it is not
    part of the dry OpenMM protein-ligand parametrization path. The solvation
    step can later restore these waters before adding bulk solvent, so the
    freshly placed solvent is generated around the retained crystallographic
    waters instead of ignoring them. Only coordinate records with common water
    residue names are written, which avoids copying protein, ligand, metal, or
    header records into the water-only file. ``None`` is returned when no water
    records are present, and an old generated file is removed to avoid stale
    artefacts in persistent integration-test folders.

    Clash filtering (removing waters that overlap with protein/ligand heavy atoms)
    is intentionally left to the OpenMM solvation step
    (``_restore_crystal_waters_before_solvation`` in ``solvation_openmm.py``),
    which uses OpenMM topology to do the check correctly.
    """
    water_lines = [
        line for line in protein_pdb.read_text(encoding="utf-8").splitlines() if _is_crystal_water_pdb_record(line)
    ]

    if not water_lines:
        with contextlib.suppress(FileNotFoundError):
            output_pdb.unlink()
        return None

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(water_lines) + "\nEND\n", encoding="utf-8")
    return output_pdb


def _remove_crystal_waters_from_modeller(pdb: PDBFile) -> Modeller:
    """Return an OpenMM modeller object with crystallographic waters removed.

    OpenMM protein force-field XML files should parameterize the dry protein
    here, not the crystallographic water molecules from the input structure.
    Bulk solvent and retained crystallographic waters are handled by the later
    solvation stage, which keeps parametrization and solvation responsibilities
    separate. This helper uses OpenMM's own ``Modeller.delete(...)`` method
    instead of maintaining a second topology parser in this module. The returned
    modeller preserves the non-water protein coordinates and is then combined
    with the ligand topology.
    """
    modeller = Modeller(pdb.topology, pdb.positions)
    atoms_before = modeller.topology.getNumAtoms()
    water_residues = [
        residue for residue in modeller.topology.residues() if residue.name.strip().upper() in _WATER_RESIDUE_NAMES
    ]

    if water_residues:
        modeller.delete(water_residues)

    atoms_after = modeller.topology.getNumAtoms()

    logger.debug(
        "Removed crystallographic waters from protein topology (%d -> %d atoms).",
        atoms_before,
        atoms_after,
    )
    return modeller


def _parametrize_openmm(inp: ParametrizationInput) -> ParametrisedComplex:
    work_dir = inp.work_dir or Path(tempfile.mkdtemp(prefix="gbsa_param_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Protein -------------------------------------------------------
    logger.debug("Loading protein PDB: %s …", inp.protein_pdb)
    crystal_waters_pdb = _write_crystal_waters_pdb(
        inp.protein_pdb,
        work_dir / "crystal_waters.pdb",
    )
    if crystal_waters_pdb is None:
        logger.debug("No crystallographic waters found in protein PDB.")
    else:
        logger.debug("Crystal waters written → %s.", crystal_waters_pdb)

    pdb = PDBFile(str(inp.protein_pdb))
    protein_modeller = _remove_crystal_waters_from_modeller(pdb)
    logger.debug(
        "Protein PDB loaded without crystal waters (%d atoms).",
        protein_modeller.topology.getNumAtoms(),
    )

    # --- Ligand --------------------------------------------------------
    logger.debug("Loading ligand SDF: %s …", inp.ligand_sdf)
    ligand = Molecule.from_file(str(inp.ligand_sdf))
    if not ligand.conformers:
        raise ValueError(
            f"Ligand SDF '{inp.ligand_sdf}' contains no 3-D conformers. "
            "Provide an SDF file with embedded 3-D coordinates."
        )
    logger.debug(
        "Ligand loaded (%d atoms, %d conformers).",
        ligand.n_atoms,
        len(ligand.conformers),
    )

    logger.debug("Assigning partial charges (method=%s) …", inp.config.charge_method.value)
    kwargs: dict[str, Any] = {
        "partial_charge_method": inp.config.charge_method.value,
        "normalize_partial_charges": True,
        "use_conformers": ligand.conformers,
    }
    if inp.net_charge is not None:
        kwargs["partial_charges"] = None  # reset; net_charge is passed separately
    ligand.assign_partial_charges(**kwargs)
    logger.debug("Partial charges assigned.")

    # --- Force field ---------------------------------------------------
    protein_xmls = _PROTEIN_FF_XML[inp.config.protein_ff]
    extra_xmls = [str(p) for p in inp.config.extra_ff_files]
    logger.debug(
        "Building force field (protein=%s, extra=%d files) …",
        inp.config.protein_ff.value,
        len(extra_xmls),
    )
    forcefield = ForceField(*protein_xmls, *extra_xmls)
    logger.debug("Force field built.")

    logger.debug(
        "Registering GAFF template generator (%s) …",
        _GAFF_FF_VERSION[inp.config.ligand_ff],
    )
    gaff = GAFFTemplateGenerator(
        molecules=[ligand],
        forcefield=_GAFF_FF_VERSION[inp.config.ligand_ff],
        cache=None,
    )
    forcefield.registerTemplateGenerator(gaff.generator)
    logger.debug("GAFF template generator registered.")

    logger.debug("Combining protein+ligand topology …")
    modeller = Modeller(protein_modeller.topology, protein_modeller.positions)
    modeller.add(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())
    logger.debug("Combined dry topology: %d atoms.", modeller.topology.getNumAtoms())

    logger.debug("Creating OpenMM system (may take a moment) …")
    system: System = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,  # VERY IMPORTANT THIS INFORMATION DOES NOT GET LOST: Constraints None is safe, HBOND fails as the generated top file does not fit with the parameters
    )
    logger.debug("OpenMM system created (%d particles).", system.getNumParticles())

    logger.debug("Converting OpenMM system to ParmEd structure …")
    structure = pmd.openmm.load_topology(modeller.topology, system, modeller.positions)
    logger.debug("ParmEd structure ready (%d atoms).", len(structure.atoms))

    gro_file = work_dir / "complex.gro"
    top_file = work_dir / "complex.top"
    logger.debug("Writing GROMACS topology → %s …", top_file)
    structure.save(str(top_file), format="gromacs")
    logger.debug("Writing GROMACS coordinates → %s …", gro_file)
    structure.save(str(gro_file))
    logger.debug("GROMACS files written.")

    complex = ParametrisedComplex(
        gro_file=gro_file,
        top_file=top_file,
        config=inp.config,
        forcefield=forcefield,
        parmed_structure=structure,
        crystal_waters_pdb=crystal_waters_pdb,
    )

    cache_file = work_dir / "complex.pickle"
    with contextlib.suppress(Exception):
        cache_file.write_bytes(pickle.dumps(complex))

    return complex


# ---------------------------------------------------------------------------
# Legacy BSS wrappers — kept for API compatibility
# ---------------------------------------------------------------------------


def load_protein_pdb(pdb_path: PathLike) -> BSS._SireWrappers.Molecule:
    """Load a protein from a PDB file and return the (first) molecule."""
    system = BSS.IO.readMolecules(str(pdb_path))
    mols = system.getMolecules()
    if not mols:
        raise ValueError(f"No molecules found in {pdb_path}")
    return mols[0]


def parameterise_protein_amber(
    protein: BSS._SireWrappers.Molecule,
    ff: str = "ff14SB",
    water_model: str | None = None,
    work_dir: PathLike | None = None,
) -> BSS._SireWrappers.Molecule:
    """Parameterize a protein via BioSimSpace/tleap.

    Returns a BSS Molecule suitable for use with the BSS solvation and MD
    pipeline. For tleap-free parametrization use :func:`parametrize`.
    """
    ff = ff.lower().strip()
    kwargs: dict[str, Any] = {}
    if water_model is not None:
        kwargs["water_model"] = water_model
    if work_dir is not None:
        kwargs["work_dir"] = str(work_dir)

    if ff == "ff14sb":
        out = BSS.Parameters.ff14SB(protein, **kwargs)
    elif ff == "ff19sb":
        out = BSS.Parameters.ff19SB(protein, **kwargs)
    elif ff == "ff99sb":
        out = BSS.Parameters.ff99SB(protein, **kwargs)
    else:
        raise ValueError(f"Unsupported protein FF '{ff}'. Try ff14SB, ff19SB, ff99SB.")

    return _ensure_molecule(out)


def _ensure_molecule(x: Any) -> BSS._SireWrappers.Molecule:
    """Ensure that a molecule is returned as an BSS._SireWrappers.Molecule."""
    if hasattr(x, "getMolecule"):
        return x.getMolecule()
    return x


def parameterise_ligand_gaff2(
    ligand: BSS._SireWrappers.Molecule,
    net_charge: int | None = None,
    charge_method: str = "BCC",
    work_dir: PathLike | None = None,
) -> BSS._SireWrappers.Molecule:
    """Parameterise a ligand via BioSimSpace/antechamber (GAFF2).

    Returns a BSS Molecule suitable for use with the BSS solvation and MD
    pipeline. For the OpenMM-based path use :func:`parametrize`.
    """
    kwargs: dict[str, Any] = {
        "net_charge": net_charge,
        "charge_method": charge_method,
    }
    if work_dir is not None:
        kwargs["work_dir"] = str(work_dir)

    return _ensure_molecule(BSS.Parameters.gaff2(ligand, **kwargs))


def make_protein_ligand_system(
    protein: BSS._SireWrappers.Molecule,
    ligand: BSS._SireWrappers.Molecule,
) -> BSS._SireWrappers.System:
    """Combine a parametrised protein and ligand into a BSS System."""
    system = BSS._SireWrappers.System(protein)
    system.addMolecules(ligand)
    return system


def load_and_parameterise(
    protein_pdb: PathLike,
    ligand: PathLike | BSS._SireWrappers.Molecule,
    protein_ff: str = "ff14SB",
    ligand_net_charge: int | None = None,
    ligand_charge_method: str = "BCC",
    work_dir: PathLike | None = None,
) -> ParametrisedComplex:
    """Load and parametrize a protein-ligand complex.

    .. deprecated::
        Use :func:`parametrize` with :class:`ParametrizationInput` instead.
        ``ligand`` must now be a file path; BSS Molecule inputs are no longer
        accepted. ``ligand_charge_method`` is ignored; AM1-BCC is used.
    """
    warnings.warn(
        "load_and_parameterise() is deprecated. Use parametrize() with ParametrizationInput instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(ligand, (str, Path)):
        raise TypeError(
            "load_and_parameterise() now requires ligand as a file path. "
            "Save the molecule to SDF first, or use parametrize() directly."
        )
    return parametrize(
        ParametrizationInput(
            protein_pdb=Path(protein_pdb),
            ligand_sdf=Path(ligand),
            config=ParametrizationConfig(protein_ff=ProteinFF.from_str(protein_ff)),
            net_charge=ligand_net_charge,
            work_dir=Path(work_dir) if work_dir else None,
        )
    )


def export_gromacs_top_gro(
    system: System,
    prefix: str,
) -> list[Path]:
    """Export GROMACS .gro and .top files from a BSS System."""
    out_gro = Path(f"{prefix}.gro")
    out_top = Path(f"{prefix}.top")

    BSS.IO.saveMolecules(str(out_gro), system, fileformat="gro87")
    BSS.IO.saveMolecules(str(out_top), system, fileformat="grotop")

    return [out_gro, out_top]
