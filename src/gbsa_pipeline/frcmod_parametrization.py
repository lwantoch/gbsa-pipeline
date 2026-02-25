"""Utilities for loading AMBER parameters into the GBSA pipeline.

Provides two independent workflows:

- :class:`AmberInput` / :func:`load_amber_complex` — load a fully
  pre-parametrized AMBER system (prmtop + inpcrd) and export it to GROMACS.
- :class:`AmberFFInput` / :func:`build_amber_ff_xml` — convert AMBER frcmod
  and mol2 residue template files into a unified OpenMM ForceField XML that
  can be passed to :class:`~gbsa_pipeline.parametrization.ParametrizationConfig`
  via ``extra_ff_files``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import parmed as pmd
from pydantic import BaseModel, ConfigDict, FilePath, field_validator

from gbsa_pipeline.parametrization import ParametrisedComplex, ParametrizationConfig
from gbsa_pipeline.parametrization_enum import LigandFF, ProteinFF

# ---------------------------------------------------------------------------
# Internal helpers — AMBER parameter file locations
# ---------------------------------------------------------------------------

# Maps each protein FF to the base AMBER parameter files it needs.
# These files live in $AMBERHOME/dat/leap/parm/ and ship with AmberTools.
_PROTEIN_BASE_PARAMS: dict[ProteinFF, list[str]] = {
    ProteinFF.FF14SB: ["parm19.dat", "frcmod.ff14SB"],
    ProteinFF.FF19SB: ["parm19.dat", "frcmod.ff19SB"],
    ProteinFF.FF99SB: ["parm10.dat", "frcmod.ff99SBildn"],
}

_LIGAND_BASE_PARAMS: dict[LigandFF, str] = {
    LigandFF.GAFF: "gaff.dat",
    LigandFF.GAFF2: "gaff2.dat",
}


def _find_amber_parm_dir() -> Path:
    """Return the AmberTools ``dat/leap/parm/`` directory.

    Checks ``AMBERHOME`` first (set by AmberTools conda packages), then
    falls back to ``sys.prefix`` (the active conda/venv environment root).

    Raises:
        RuntimeError: If the directory cannot be located.
    """
    candidates = []
    if amberhome := os.environ.get("AMBERHOME"):
        candidates.append(Path(amberhome))
    candidates.append(Path(sys.prefix))

    for root in candidates:
        p = root / "dat" / "leap" / "parm"
        if p.is_dir():
            return p

    raise RuntimeError(
        "Cannot locate the AMBER parameter directory (dat/leap/parm). "
        "Set AMBERHOME or ensure AmberTools is installed in the active environment."
    )


# ---------------------------------------------------------------------------
# AmberFFInput + build_amber_ff_xml
# ---------------------------------------------------------------------------


class AmberFFInput(BaseModel):
    """Inputs for converting AMBER frcmod + mol2 files to an OpenMM XML.

    The generated XML contains all atom types, bonded parameters, LJ
    parameters, and residue templates needed by OpenMM's ``ForceField``.
    Pass the output path to
    :class:`~gbsa_pipeline.parametrization.ParametrizationConfig` via
    ``extra_ff_files`` alongside the standard protein/water XML files.

    Parameters
    ----------
    frcmod_files:
        One or more AMBER frcmod files (e.g. from MCPB.py) that supply
        custom bond, angle, dihedral, and LJ parameters.
    residue_mol2s:
        Mol2 files for non-standard residues (e.g. metal-coordinating CYS/HIS
        modified by MCPB.py, bare metal ions).  Each file must contain exactly
        one ``@<TRIPOS>MOLECULE`` block.
    protein_ff:
        Protein force field.  Determines which base AMBER parameter files are
        loaded (``parm19.dat + frcmod.ff14SB`` for FF14SB, etc.).
    ligand_ff:
        Ligand force field.  Determines ``gaff.dat`` vs ``gaff2.dat``.
    output_xml:
        Path where the unified XML is written.  A file inside a temporary
        directory is used when ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    frcmod_files: tuple[Path, ...] = ()
    residue_mol2s: tuple[Path, ...] = ()
    protein_ff: ProteinFF = ProteinFF.FF14SB
    ligand_ff: LigandFF = LigandFF.GAFF2
    output_xml: Path | None = None

    @field_validator("frcmod_files", "residue_mol2s", mode="before")
    @classmethod
    def _check_files_exist(cls, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise ValueError("Files not found: " + ", ".join(str(p) for p in missing))
        return tuple(Path(p) for p in paths)


def build_amber_ff_xml(inp: AmberFFInput) -> Path:
    """Build a unified OpenMM ForceField XML from AMBER frcmod and mol2 files.

    Loads the base AMBER parameter files for the requested force fields, adds
    the custom frcmod parameters, merges mol2 residue templates directly into
    the :class:`parmed.openmm.OpenMMParameterSet`, and writes a single XML
    file that OpenMM's ``ForceField`` can read.

    Parameters
    ----------
    inp:
        Validated inputs.

    Returns:
        Path to the written XML file.

    Raises:
        RuntimeError: If the AMBER parameter directory cannot be found.
        ValueError: If any atom type referenced in a residue template is not
            defined in the combined parameter set.
    """
    parm_dir = _find_amber_parm_dir()

    # Base AMBER parameter files for the chosen force fields.
    base_files = [str(parm_dir / f) for f in _PROTEIN_BASE_PARAMS[inp.protein_ff]]
    base_files.append(str(parm_dir / _LIGAND_BASE_PARAMS[inp.ligand_ff]))

    # Load base params + user frcmod files into a single AmberParameterSet.
    amber_parm = pmd.amber.AmberParameterSet(
        *base_files,
        *(str(f) for f in inp.frcmod_files),
    )
    ff = pmd.openmm.OpenMMParameterSet.from_parameterset(amber_parm)

    # Merge mol2 residue templates directly into the parameter set.
    for mol2_path in inp.residue_mol2s:
        mol2 = pmd.load_file(str(mol2_path))
        if isinstance(mol2, pmd.modeller.ResidueTemplateContainer):
            ff.residues.update(mol2.to_library())
        else:
            ff.residues[mol2.name] = mol2

    # Validate: all types used in templates must be defined.
    defined = set(ff.atom_types.keys())
    used: set[str] = set()
    for templ in ff.residues.values():
        for atom in templ.atoms:
            if at := getattr(atom, "type", None):
                used.add(str(at))
    missing = used - defined
    if missing:
        raise ValueError(
            f"Residue templates reference atom types not defined in the "
            f"parameter set: {sorted(missing)}. "
            f"Add a frcmod file that defines these types."
        )

    # Write unified XML.
    output_xml = inp.output_xml or Path(tempfile.mkdtemp(prefix="gbsa_ff_")) / "combined.xml"
    ff.write(str(output_xml), write_unused=True)
    return output_xml


# ---------------------------------------------------------------------------
# AmberInput + load_amber_complex
# ---------------------------------------------------------------------------


class AmberInput(BaseModel):
    """Validated inputs for loading a pre-parametrized AMBER system.

    Parameters
    ----------
    prmtop:
        AMBER parameter/topology file (``.prmtop`` / ``.parm7``), produced
        by e.g. tleap, MCPB.py, or antechamber.
    inpcrd:
        AMBER coordinate file (``.inpcrd`` / ``.rst7``).
    output_dir:
        Directory for GROMACS output files.  A temporary directory is
        created when ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    prmtop: FilePath
    inpcrd: FilePath
    output_dir: Path | None = None


def load_amber_complex(inp: AmberInput) -> ParametrisedComplex:
    """Load an AMBER prmtop + inpcrd and export to GROMACS ``.gro`` / ``.top``.

    Uses ParmEd to read the AMBER files directly and write the GROMACS files.

    Parameters
    ----------
    inp:
        Validated inputs.

    Returns:
        :class:`~gbsa_pipeline.parametrization.ParametrisedComplex` with
        paths to the GROMACS coordinate and topology files.
    """
    work_dir = inp.output_dir or Path(tempfile.mkdtemp(prefix="gbsa_amber_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    structure = pmd.load_file(str(inp.prmtop), xyz=str(inp.inpcrd))

    gro_file = work_dir / "complex.gro"
    top_file = work_dir / "complex.top"
    structure.save(str(top_file), format="gromacs")
    structure.save(str(gro_file))

    return ParametrisedComplex(
        gro_file=gro_file,
        top_file=top_file,
        config=ParametrizationConfig(),
    )
