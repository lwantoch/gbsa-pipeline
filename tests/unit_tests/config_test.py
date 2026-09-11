"""Tests for RunConfig and CLI argument parsing."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from gbsa_pipeline.cli import main as cli_main
from gbsa_pipeline.config import (
    MembraneSystemConfig,
    MinimizationConfig,
    RunConfig,
    SolvationConfig,
    SystemConfig,
)
from gbsa_pipeline.mdp import (
    Barostat,
    Constraints,
    GromacsParams,
    Thermostat,
    VelocityGeneration,
)
from gbsa_pipeline.parametrization_enum import ChargeMethod, LigandFF, ProteinFF
from gbsa_pipeline.solvation_box import BoxShape, WaterModel


def _write_toml(tmp_path: Path, content: str) -> Path:
    """Write a temporary TOML configuration file for RunConfig tests.

    The helper keeps TOML setup local to each test while avoiding repeated file
    creation boilerplate. The content is dedented so test configurations can be
    written as readable multi-line strings. The file is always written into the
    pytest-provided temporary directory. An explicit encoding is used because the
    test suite runs with PYTHONWARNDEFAULTENCODING enabled.
    """
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# RunConfig.from_toml
# ---------------------------------------------------------------------------


def test_from_toml_minimal(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    protein.write_text("", encoding="utf-8")
    toml = _write_toml(
        tmp_path,
        f"""
        [system]
        protein = "{protein}"
        """,
    )

    cfg = RunConfig.from_toml(toml)

    assert cfg.system is not None
    assert cfg.system.protein == protein
    assert cfg.system.ligand is None
    assert cfg.forcefield.protein_ff == ProteinFF.FF14SB
    assert cfg.solvation.water_model == WaterModel.TIP3P
    assert cfg.minimization.nsteps == 10_000
    assert cfg.equilibration.simulation_time_ps == 50.0


def test_from_toml_full(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.sdf"
    protein.write_text("", encoding="utf-8")
    ligand.write_text("", encoding="utf-8")

    toml = _write_toml(
        tmp_path,
        f"""
        [system]
        protein = "{protein}"
        ligand  = "{ligand}"
        net_charge = -1

        [forcefield]
        protein_ff    = "ff19SB"
        ligand_ff     = "gaff"
        charge_method = "nagl"

        [solvation]
        water_model = "tip4p"
        shape       = "cubic"
        box_size    = 10.0

        [minimization]
        nsteps = 5000
        emtol  = 5.0

        [equilibration]
        simulation_time_ps = 200.0

        [md]
        nsteps = 1000
        dt     = 0.002
        tcoupl = "v-rescale"
        pcoupl = "Parrinello-Rahman"
        """,
    )

    cfg = RunConfig.from_toml(toml)

    assert cfg.system is not None
    assert cfg.system.ligand == ligand
    assert cfg.system.net_charge == -1
    assert cfg.forcefield.protein_ff == ProteinFF.FF19SB
    assert cfg.forcefield.ligand_ff == LigandFF.GAFF
    assert cfg.forcefield.charge_method == ChargeMethod.NAGL
    assert cfg.solvation.water_model == WaterModel.TIP4P
    assert cfg.solvation.shape == BoxShape.CUBIC
    assert cfg.solvation.box_size == 10.0
    assert cfg.minimization.nsteps == 5000
    assert cfg.minimization.emtol == 5.0
    assert cfg.equilibration.simulation_time_ps == 200.0
    assert cfg.md.nsteps == 1000
    assert cfg.md.dt == 0.002
    assert cfg.md.tcoupl == Thermostat.VRESCALE
    assert cfg.md.pcoupl == Barostat.PARRINELLO_RAHMAN


def test_from_toml_rejects_unknown_section(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    protein.write_text("", encoding="utf-8")
    toml = _write_toml(
        tmp_path,
        f"""
        [system]
        protein = "{protein}"

        [unknown_section]
        foo = "bar"
        """,
    )

    with pytest.raises(ValidationError, match="unknown_section"):
        RunConfig.from_toml(toml)


def test_from_toml_system_protein_required(tmp_path: Path) -> None:
    toml = _write_toml(tmp_path, "[system]\n")

    with pytest.raises(ValidationError, match="protein"):
        RunConfig.from_toml(toml)


# ---------------------------------------------------------------------------
# [membrane_system] -- pre-built protein-in-bilayer systems (e.g. MemProtMD)
# ---------------------------------------------------------------------------


def test_from_toml_membrane_system(tmp_path: Path) -> None:
    """[membrane_system] loads instead of [system], with [system] left unset."""
    structure = tmp_path / "atomistic-system.pdb"
    topology = tmp_path / "topol.top"
    structure.write_text("", encoding="utf-8")
    topology.write_text("", encoding="utf-8")
    toml = _write_toml(
        tmp_path,
        f"""
        [membrane_system]
        structure = "{structure}"
        topology  = "{topology}"
        """,
    )

    cfg = RunConfig.from_toml(toml)

    assert cfg.system is None
    assert cfg.membrane_system is not None
    assert cfg.membrane_system.structure == structure
    assert cfg.membrane_system.topology == topology


def test_from_toml_requires_system_or_membrane_system(tmp_path: Path) -> None:
    """Neither [system] nor [membrane_system] set is rejected, not silently defaulted."""
    toml = _write_toml(tmp_path, "")

    with pytest.raises(ValidationError, match="Exactly one of"):
        RunConfig.from_toml(toml)


def test_run_config_rejects_both_system_and_membrane_system(tmp_path: Path) -> None:
    """[system] and [membrane_system] together are rejected as ambiguous."""
    protein = tmp_path / "protein.pdb"
    structure = tmp_path / "atomistic-system.pdb"
    topology = tmp_path / "topol.top"
    for f in (protein, structure, topology):
        f.write_text("", encoding="utf-8")

    with pytest.raises(ValidationError, match="Exactly one of"):
        RunConfig(
            system=SystemConfig(protein=protein),
            membrane_system=MembraneSystemConfig(structure=structure, topology=topology),
        )


def test_membrane_system_config_requires_existing_files(tmp_path: Path) -> None:
    """structure/topology must exist on disk -- MembraneSystemConfig uses FilePath."""
    topology = tmp_path / "topol.top"
    topology.write_text("", encoding="utf-8")

    with pytest.raises(ValidationError):
        MembraneSystemConfig(structure=tmp_path / "missing.pdb", topology=topology)


def test_membrane_system_config_detects_gromacs_topology(tmp_path: Path) -> None:
    """A .top topology is detected as GROMACS, not AMBER."""
    structure = tmp_path / "atomistic-system.pdb"
    topology = tmp_path / "topol.top"
    structure.write_text("", encoding="utf-8")
    topology.write_text("", encoding="utf-8")

    cfg = MembraneSystemConfig(structure=structure, topology=topology)

    assert cfg.is_amber_format is False


@pytest.mark.parametrize("suffix", [".prmtop", ".parm7"])
def test_membrane_system_config_detects_amber_topology(tmp_path: Path, suffix: str) -> None:
    """.prmtop/.parm7 topologies are detected as AMBER -- no canonicalization needed.

    Confirmed end-to-end (load, minimize) against a real Lipid21-parametrized
    membrane protein built with packmol-memgen; see the pipeline._stage_load_
    membrane_system docstring.
    """
    structure = tmp_path / "system.inpcrd"
    topology = tmp_path / f"system{suffix}"
    structure.write_text("", encoding="utf-8")
    topology.write_text("", encoding="utf-8")

    cfg = MembraneSystemConfig(structure=structure, topology=topology)

    assert cfg.is_amber_format is True


def test_to_parametrization_input_raises_for_membrane_system(tmp_path: Path) -> None:
    """A [membrane_system] config has nothing to parametrize -- fails with a clear message."""
    structure = tmp_path / "atomistic-system.pdb"
    topology = tmp_path / "topol.top"
    structure.write_text("", encoding="utf-8")
    topology.write_text("", encoding="utf-8")

    cfg = RunConfig(membrane_system=MembraneSystemConfig(structure=structure, topology=topology))

    with pytest.raises(ValueError, match="membrane_system"):
        cfg.to_parametrization_input(tmp_path / "work")


# ---------------------------------------------------------------------------
# SystemConfig
# ---------------------------------------------------------------------------


def test_system_config_defaults() -> None:
    cfg = SystemConfig(protein=Path("/some/protein.pdb"))

    assert cfg.ligand is None
    assert cfg.net_charge is None


def test_minimization_config_define_defaults_to_none() -> None:
    """``define`` defaults to None -- the ordinary protein-ligand path is unaffected."""
    cfg = MinimizationConfig()

    assert cfg.define is None


def test_minimization_config_accepts_flex_spc_define() -> None:
    """``define`` carries a GROMACS preprocessor flag through, e.g. "-DFLEX_SPC".

    Needed for systems whose starting coordinates aren't precise enough for
    rigid SETTLE-constrained water (e.g. MemProtMD's CG2AT-backmapped
    output) -- minimizing with flexible water for the first pass avoids the
    resulting NaN potential energy at step 0. See MinimizationConfig's
    docstring for the full rationale.
    """
    cfg = MinimizationConfig(define="-DFLEX_SPC")

    assert cfg.define == "-DFLEX_SPC"


def test_system_config_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        SystemConfig(protein=Path("/p.pdb"), bad_field="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SolvationConfig
# ---------------------------------------------------------------------------


def test_solvation_config_defaults() -> None:
    cfg = SolvationConfig()

    assert cfg.water_model == WaterModel.TIP3P
    assert cfg.shape == BoxShape.TRUNCATED_OCTAHEDRON
    assert cfg.box_size == 8.0
    assert cfg.padding is None
    assert cfg.ion_concentration == 0.15
    assert cfg.neutralize is True


# ---------------------------------------------------------------------------
# GromacsParams fields
# ---------------------------------------------------------------------------


def test_gromacs_params_thermostat_fields() -> None:
    params = GromacsParams(tcoupl=Thermostat.VRESCALE, ref_t=310.0, tau_t=0.5)

    mapping = params.to_mapping()

    assert mapping["tcoupl"] == "v-rescale"
    assert mapping["ref-t"] == 310.0
    assert mapping["tau-t"] == 0.5


def test_gromacs_params_barostat_fields() -> None:
    params = GromacsParams(
        pcoupl=Barostat.PARRINELLO_RAHMAN,
        ref_p=1.0,
        tau_p=2.0,
        compressibility=4.5e-5,
    )

    mapping = params.to_mapping()

    assert mapping["pcoupl"] == "Parrinello-Rahman"
    assert mapping["ref-p"] == 1.0
    assert mapping["tau-p"] == 2.0
    assert mapping["compressibility"] == pytest.approx(4.5e-5)


def test_gromacs_params_velocity_fields() -> None:
    params = GromacsParams(
        gen_vel=VelocityGeneration.YES,
        gen_temp=310.0,
        gen_seed=42,
    )

    mapping = params.to_mapping()

    assert mapping["gen-vel"] == "yes"
    assert mapping["gen-temp"] == 310.0
    assert mapping["gen-seed"] == 42


def test_gromacs_params_constraints_field() -> None:
    params = GromacsParams(constraints=Constraints.HYDROGENS_BONDS)

    mapping = params.to_mapping()

    assert mapping["constraints"] == "h-bonds"


def test_gromacs_params_roundtrip_fields() -> None:
    original = GromacsParams(
        tcoupl=Thermostat.NOSE_HOOVER,
        ref_t=300.0,
        tau_t=0.1,
        nhchainlength=5,
        pcoupl=Barostat.CRESCALE,
        tau_p=1.0,
        ref_p=1.0,
        compressibility=4.5e-5,
    )

    mapping = original.to_mapping()
    restored = GromacsParams.from_mapping(mapping)

    assert restored == original


# ---------------------------------------------------------------------------
# to_parametrization_input
# ---------------------------------------------------------------------------


def test_to_parametrization_input_raises_when_no_ligand(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    protein.write_text("", encoding="utf-8")

    cfg = RunConfig(system=SystemConfig(protein=protein))

    with pytest.raises(ValueError, match=r"system\.ligand"):
        cfg.to_parametrization_input(tmp_path / "work")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def test_cli_parses_config_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    protein = tmp_path / "protein.pdb"
    protein.write_text("", encoding="utf-8")
    cfg_path.write_text(f'[system]\nprotein = "{protein}"\n', encoding="utf-8")

    with mock.patch("gbsa_pipeline.cli.run_pipeline") as mock_run:
        cli_main([str(cfg_path)])

    mock_run.assert_called_once()
    _, output_dir = mock_run.call_args.args
    assert output_dir == tmp_path / "gbsa_output"


def test_cli_custom_output_dir(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    protein = tmp_path / "protein.pdb"
    protein.write_text("", encoding="utf-8")
    cfg_path.write_text(f'[system]\nprotein = "{protein}"\n', encoding="utf-8")
    output_dir = tmp_path / "custom_out"

    with mock.patch("gbsa_pipeline.cli.run_pipeline") as mock_run:
        cli_main([str(cfg_path), "-o", str(output_dir)])

    mock_run.assert_called_once()
    _, parsed_output_dir = mock_run.call_args.args
    assert parsed_output_dir == output_dir
