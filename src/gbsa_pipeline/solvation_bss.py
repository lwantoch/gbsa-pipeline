"""BioSimSpace solvation: BSS.Solvent.tip3p (wraps gmx solvate + gmx genion).

BSS writes a fully self-consistent GROMACS topology that BSS/Sire can load back
for MD.  The previous gmx-direct approach (gmx solvate + manual topology
injection) produced topologies that BSS/Sire rejected with "There are no
molecule groups called 'all'".

Crystallographic waters carried on the parametrized complex
(``crystal_waters_pdb``) are normalized to AMBER TIP3P naming, parametrized as
TIP3P (``BSS.Parameters.ff14SB(..., water_model="tip3p")``), and merged into the
system BEFORE bulk solvation, so BSS.Solvent fills bulk water + ions around
them.  The whole path stays in AMBER/GROMACS — no OpenMM.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gbsa_pipeline.solvation_box import SolvatedComplex, SolvationParams, WaterModel, run_solvation

if TYPE_CHECKING:
    from gbsa_pipeline.parametrization import ParametrisedComplex

logger = logging.getLogger(__name__)


def solvate_bss(
    parametrized: ParametrisedComplex,
    params: SolvationParams,
    output_gro: Path,
    output_top: Path,
) -> SolvatedComplex:
    """Solvate a parametrized complex via BSS.Solvent.

    Loads the dry GROMACS complex into BSS, calls the water/ion placement
    via BSS.Solvent (which internally runs gmx solvate and gmx genion), and
    saves the result as GROMACS GRO/TOP files that BSS/Sire can load for MD.

    Parameters
    ----------
    parametrized:
        Dry protein-ligand complex from :func:`~gbsa_pipeline.parametrization.parametrize`.
    params:
        Solvation box parameters (water model, padding, ion concentration, ...).
    output_gro:
        Path for the solvated GROMACS coordinate file.
    output_top:
        Path for the solvated GROMACS topology file.

    Returns:
    -------
    SolvatedComplex
        Dataclass holding paths to the written GROMACS files.
    """
    import BioSimSpace as BSS  # noqa: PLC0415

    work_dir = output_gro.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    system = BSS.IO.readMolecules([str(parametrized.gro_file), str(parametrized.top_file)])
    logger.debug(
        "Dry complex loaded: %d molecules, %d atoms.",
        system.nMolecules(),
        system.nAtoms(),
    )

    # Re-insert crystallographic waters BEFORE bulk solvation (AMBER/GROMACS
    # path, no OpenMM). BSS.Solvent then fills bulk water + ions around them.
    if parametrized.crystal_waters_pdb is not None and parametrized.crystal_waters_pdb.exists():
        waters = _load_crystal_waters_tip3p(parametrized.crystal_waters_pdb, work_dir)
        if waters is not None:
            n_before = system.nMolecules()
            system = system + waters
            logger.info(
                "Re-inserted crystal waters before solvation: +%d molecules (%d atoms).",
                system.nMolecules() - n_before,
                waters.nAtoms(),
            )

    bss_work = work_dir / "_bss_solvate"
    bss_work.mkdir(exist_ok=True)
    solvated = run_solvation(system, params, work_dir=bss_work)
    logger.debug(
        "Solvated system: %d molecules, %d atoms.",
        solvated.nMolecules(),
        solvated.nAtoms(),
    )

    # BSS.IO.saveMolecules appends .gro / .top to the given prefix.
    prefix = str(output_gro.with_suffix(""))
    BSS.IO.saveMolecules(prefix, solvated, ["gro87", "grotop"])

    return SolvatedComplex(gro_file=output_gro, top_file=output_top)


_WATER_RESNAMES = {"HOH", "WAT", "SOL", "TIP3", "T3P"}
_AMBER_WATER_ATOM = {
    "OW": " O  ", "HW1": " H1 ", "HW2": " H2 ",
    "O": " O  ", "H1": " H1 ", "H2": " H2 ",
}


def _guess_amberhome() -> str:
    """Best-effort AMBERHOME from the location of ``tleap`` (conda/pixi env prefix)."""
    tleap = shutil.which("tleap")
    if tleap:
        return str(Path(tleap).resolve().parent.parent)
    return os.environ.get("CONDA_PREFIX", "")


def _normalize_crystal_waters(src: Path, dst: Path) -> int:
    """Rewrite crystallographic waters to AMBER TIP3P convention.

    DEKOIS ``*_WAT.pdb`` files store waters with GROMACS atom names
    (OW/HW1/HW2) and sometimes pack two waters into a single residue, both of
    which make tleap (and therefore ``BSS.Parameters``) fail with
    "atom ... does not have a type".  This rewrites each water as a single
    ``WAT`` residue with AMBER atom names (O/H1/H2), splicing into the original
    line so the coordinate columns are preserved byte-for-byte.  Returns the
    number of waters written.
    """
    resid = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if line[:6] not in ("ATOM  ", "HETATM"):
                continue
            if line[17:20].strip() not in _WATER_RESNAMES:
                continue
            atom_name = _AMBER_WATER_ATOM.get(line[12:16].strip())
            if atom_name is None:
                continue
            if atom_name == " O  ":  # an oxygen starts a new water
                resid += 1
            fout.write(line[:12] + atom_name + line[16:17] + "WAT" + line[20:22] + f"{resid:>4}" + line[26:])
        fout.write("TER\nEND\n")
    return resid


def _load_crystal_waters_tip3p(pdb: Path, work_dir: Path) -> Any | None:
    """Parametrize crystallographic waters as TIP3P and return a BSS molecule.

    Normalizes the input waters then runs the openbiosim crystal-water tutorial
    path (``BSS.Parameters.ff14SB(..., water_model="tip3p", ensure_compatible=False)``),
    which uses tleap internally — no OpenMM.  Returns ``None`` when no waters are
    found.  ``BSS.Parameters`` needs ``AMBERHOME``; if unset it is derived from
    the location of ``tleap`` on PATH.
    """
    import BioSimSpace as BSS  # noqa: PLC0415

    if "AMBERHOME" not in os.environ:
        amberhome = _guess_amberhome()
        if amberhome:
            os.environ["AMBERHOME"] = amberhome

    norm = work_dir / "crystal_waters_tip3p.pdb"
    n_waters = _normalize_crystal_waters(pdb, norm)
    if n_waters == 0:
        logger.warning("No crystallographic waters found in %s; skipping re-insertion.", pdb)
        return None

    loaded = BSS.IO.readMolecules([str(norm)])
    waters = loaded[0] if loaded.nMolecules() == 1 else loaded
    parametrized = BSS.Parameters.ff14SB(
        waters, water_model="tip3p", ensure_compatible=False, work_dir=str(work_dir)
    ).getMolecule()
    logger.info("Parametrized %d crystal waters as TIP3P.", n_waters)
    return parametrized


# ---------------------------------------------------------------------------
# Legacy BSS wrapper
# ---------------------------------------------------------------------------


def solvate_parametrized_complex(
    parametrized: ParametrisedComplex,
    *,
    shell_nm: float = 1.0,
    water_model: WaterModel | str = WaterModel.TIP3P,
    work_dir: Path | None = None,
) -> Any:
    """Deprecated — use solvate_bss instead."""
    import BioSimSpace as BSS  # noqa: PLC0415

    logger.warning("solvate_parametrized_complex is deprecated; use solvate_bss instead.")
    dry_system = BSS.IO.readMolecules([str(parametrized.gro_file), str(parametrized.top_file)])
    solvent_fn = getattr(BSS.Solvent, WaterModel(water_model).value)
    kwargs: dict[str, Any] = {
        "ion_conc": 0,
        "is_neutral": False,
        "shell": shell_nm * BSS.Units.Length.nanometer,
    }
    if work_dir is not None:
        kwargs["work_dir"] = str(work_dir)
    return solvent_fn(dry_system, **kwargs)
