"""Parametrize complex."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import BioSimSpace as BSS
import sire as sr


PathLike = Union[str, Path]

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
    *,
    water_model: str | None = None,
    work_dir: PathLike | None = None,
) -> BSS._SireWrappers.Molecule:
    """Parameterize a protein using AMBER ff14SB."""
    ff = ff.lower().strip()
    kwargs = {}
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
    *,
    net_charge: int | None = None,
    charge_method: str = "BCC",
    work_dir: PathLike | None = None,
) -> BSS._SireWrappers.Molecule:
    """Parameterise a ligand using GAFF2."""
    kwargs = {
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
    """Make a protein ligand using GAFF2."""
    system = BSS._SireWrappers.System(protein)
    system.addMolecules(ligand)
    return system


@dataclass(frozen=True)
class ParametrisedComplex:
    """Parametrised complex object."""

    protein: BSS._SireWrappers.Molecule
    ligand: BSS._SireWrappers.Molecule
    system: BSS._SireWrappers.System


def load_and_parameterise(
    protein_pdb: PathLike,
    ligand_sire: BSS._SireWrappers.Molecule,
    *,
    protein_ff: str = "ff14SB",
    ligand_net_charge: int | None = None,
    ligand_charge_method: str = "BCC",
    work_dir: PathLike | None = None,
) -> ParametrisedComplex:
    """Convenience function: load protein + ligand, parameterise both, return combined System."""
    protein = load_protein_pdb(protein_pdb)
    ligand = ligand_sire

    p_protein = parameterise_protein_amber(
        protein, ff=protein_ff, work_dir=work_dir
    )
    p_ligand = parameterise_ligand_gaff2(
        ligand,
        net_charge=ligand_net_charge,
        charge_method=ligand_charge_method,
        work_dir=work_dir,
    )

    system = make_protein_ligand_system(p_protein, p_ligand)
    return ParametrisedComplex(
        protein=p_protein,
        ligand=p_ligand,
        system=system,
    )
