"""Parametrize protein-ligand complexes."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import BioSimSpace as BSS

if TYPE_CHECKING:
    from BioSimSpace._SireWrappers import System

from gbsa_pipeline.parametrization_models import (
    ParametrisedComplex,
    ParametrizationConfig,
    ParametrizationInput,
)
from gbsa_pipeline.parametrize_openmm import _parametrize_openmm
from gbsa_pipeline.tleap import _parametrize_tleap

PathLike = Union[str, Path]

__all__ = [
    "ParametrisedComplex",
    "ParametrizationConfig",
    "ParametrizationInput",
    "export_gromacs_top_gro",
    "load_and_parameterise",
    "load_protein_pdb",
    "make_protein_ligand_system",
    "parameterise_ligand_gaff2",
    "parameterise_protein_amber",
    "parametrize",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parametrize(inp: ParametrizationInput) -> ParametrisedComplex:
    """Parametrize a protein-ligand complex.

    Routes to the tleap path when the extra_ff_files include mol2 or frcmod
    files (non-standard residues like CSD, metal-coordinating cysteines from
    MCPB.py, etc.). The tleap path calls antechamber for the ligand and uses
    tleap directly for the combined protein+ligand system so that HETATM
    non-standard residues are handled natively without OpenMM template matching.

    For XML-only extra_ff (e.g. phosaa14SB.xml for PTR), the OpenMM path is
    used unchanged.

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
    has_frcmod_mol2 = any(p.suffix.lower() in {".frcmod", ".mol2"} for p in inp.config.extra_ff_files)
    if has_frcmod_mol2 or inp.config.leaprc_extra_sources or inp.config.mcpb_tleap_in is not None:
        return _parametrize_tleap(inp)
    return _parametrize_openmm(inp)


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
    from gbsa_pipeline.parametrization_enum import ProteinFF  # noqa: PLC0415

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
