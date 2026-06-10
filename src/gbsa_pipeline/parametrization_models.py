"""Shared data models and constants for the parametrization pipeline."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
from pydantic import BaseModel, ConfigDict, Field, FilePath

from gbsa_pipeline._constants import WATER_RESIDUE_NAMES
from gbsa_pipeline.parametrization_enum import ChargeMethod, LigandFF, ProteinFF

if TYPE_CHECKING:
    import parmed as pmd
    from openmm.app import ForceField

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
    extra_ff_files: tuple[FilePath, ...] = ()
    mcpb_tleap_in: FilePath | None = None
    leaprc_extra_sources: tuple[str, ...] = ()

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
    cofactor_sdfs:
        Paths to cofactor SDF files (e.g. metal-chelating ligands, cofactors).
        Each must contain embedded 3-D coordinates. GAFF2 parameters and AM1-BCC
        charges are assigned automatically. Defaults to no cofactors.
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

    protein_pdb: FilePath
    ligand_sdf: FilePath
    cofactor_sdfs: tuple[FilePath, ...] = ()
    config: ParametrizationConfig = Field(default_factory=ParametrizationConfig)
    net_charge: int | None = None
    work_dir: Path | None = None


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
    forcefield: ForceField | None = field(default=None, hash=False, compare=False, repr=False)
    parmed_structure: pmd.Structure | None = field(default=None, hash=False, compare=False, repr=False)
    crystal_waters_pdb: Path | None = None


# ---------------------------------------------------------------------------
# Shared protein PDB utilities
# ---------------------------------------------------------------------------


def _write_crystal_waters_pdb(protein_pdb: Path, output_pdb: Path) -> Path | None:
    """Write crystallographic water residues to a separate PDB file.

    The generated file is an inspection and preservation artefact; it is not
    part of the dry OpenMM protein-ligand parametrization path. The solvation
    step can later restore these waters before adding bulk solvent, so the
    freshly placed solvent is generated around the retained crystallographic
    waters instead of ignoring them. ``None`` is returned when no water
    residues are present, and an old generated file is removed to avoid stale
    artefacts in persistent integration-test folders.

    Clash filtering (removing waters that overlap with protein/ligand heavy atoms)
    is intentionally left to the OpenMM solvation step
    (``_restore_crystal_waters_before_solvation`` in ``solvation_openmm.py``),
    which uses OpenMM topology to do the check correctly.
    """
    st = gemmi.read_pdb(str(protein_pdb))
    water_found = False
    for model in st:
        for chain in model:
            to_remove = [i for i, res in enumerate(chain) if res.name.upper() not in WATER_RESIDUE_NAMES]
            for i in reversed(to_remove):
                del chain[i]
            if len(chain) > 0:
                water_found = True

    if not water_found:
        with contextlib.suppress(FileNotFoundError):
            output_pdb.unlink()
        return None

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(output_pdb))
    return output_pdb
