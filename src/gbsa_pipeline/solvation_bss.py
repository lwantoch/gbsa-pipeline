"""BioSimSpace solvation: BSS.Solvent.tip3p (wraps gmx solvate + gmx genion).

BSS writes a fully self-consistent GROMACS topology that BSS/Sire can load back
for MD.  The previous gmx-direct approach (gmx solvate + manual topology
injection) produced topologies that BSS/Sire rejected with "There are no
molecule groups called 'all'".

Crystal water positions from the parametrized complex are not explicitly
pre-inserted.  BSS.Solvent places bulk TIP3P water around the complex, which
naturally fills crystal-water binding sites.
"""

from __future__ import annotations

import logging
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
        Solvation box parameters (water model, padding, ion concentration, …).
    output_gro:
        Path for the solvated GROMACS coordinate file.
    output_top:
        Path for the solvated GROMACS topology file.

    Returns
    -------
    SolvatedComplex
        Dataclass holding paths to the written GROMACS files.
    """
    import BioSimSpace as BSS  # noqa: PLC0415

    work_dir = output_gro.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    if parametrized.crystal_waters_pdb is not None:
        logger.debug(
            "Crystal waters at %s are not pre-inserted; BSS.Solvent places fresh TIP3P water.",
            parametrized.crystal_waters_pdb,
        )

    system = BSS.IO.readMolecules([str(parametrized.gro_file), str(parametrized.top_file)])
    logger.debug(
        "Dry complex loaded: %d molecules, %d atoms.",
        system.nMolecules(),
        system.nAtoms(),
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
