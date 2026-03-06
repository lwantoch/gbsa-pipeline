"""Tests for RunConfig and CLI argument parsing."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from gbsa_pipeline.change_defaults import GromacsParams
from gbsa_pipeline.change_defaults_enum import Barostat, Constraints, Thermostat, VelocityGeneration
from gbsa_pipeline.cli import main as cli_main
from gbsa_pipeline.config import (
    RunConfig,
    SolvationConfig,
    SystemConfig,
)
from gbsa_pipeline.parametrization_enum import ChargeMethod, LigandFF, ProteinFF
from gbsa_pipeline.solvation_box import BoxShape, WaterModel

# ---------------------------------------------------------------------------
# RunConfig.from_toml
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(content))
    return path


def test_from_toml_minimal(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    protein.write_text("")
    toml = _write_toml(
        tmp_path,
        f"""
        [system]
        protein = "{protein}"
        """,
    )

    cfg = RunConfig.from_toml(toml)

    assert cfg.system.protein == protein
    assert cfg.system.ligand is None
    # Defaults populated
    assert cfg.forcefield.protein_ff == ProteinFF.FF14SB
    assert cfg.solvation.water_model == WaterModel.TIP3P
    assert cfg.minimization.nsteps == 10_000
    assert cfg.equilibration.simulation_time_ps == 500.0


def test_from_toml_full(tmp_path: Path) -> None:
    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.sdf"
    protein.write_text("")
    ligand.write_text("")

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
        box_shape   = "cubic"
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

    assert cfg.system.ligand == ligand
    assert cfg.system.net_charge == -1
    assert cfg.forcefield.protein_ff == ProteinFF.FF19SB
    assert cfg.forcefield.ligand_ff == LigandFF.GAFF
    assert cfg.forcefield.charge_method == ChargeMethod.NAGL
    assert cfg.solvation.water_model == WaterModel.TIP4P
    assert cfg.solvation.box_shape == BoxShape.CUBIC
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
    protein.write_text("")
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
# SystemConfig
# ---------------------------------------------------------------------------


def test_system_config_defaults() -> None:
    cfg = SystemConfig(protein=Path("/some/protein.pdb"))
    assert cfg.ligand is None
    assert cfg.extra_ff_files == ()
    assert cfg.net_charge is None


def test_system_config_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        SystemConfig(protein=Path("/p.pdb"), bad_field="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SolvationConfig
# ---------------------------------------------------------------------------


def test_solvation_config_defaults() -> None:
    cfg = SolvationConfig()
    assert cfg.water_model == WaterModel.TIP3P
    assert cfg.box_shape == BoxShape.TRUNCATED_OCTAHEDRON
    assert cfg.box_size == 8.0
    assert cfg.padding is None
    assert cfg.ion_concentration == 0.15
    assert cfg.is_neutral is True


# ---------------------------------------------------------------------------
# GromacsParams new fields
# ---------------------------------------------------------------------------


def test_gromacs_params_new_thermostat_fields() -> None:
    p = GromacsParams(tcoupl=Thermostat.VRESCALE, ref_t=310.0, tau_t=0.5)
    mapping = p.to_mapping()

    assert mapping["tcoupl"] == "v-rescale"
    assert mapping["ref-t"] == 310.0
    assert mapping["tau-t"] == 0.5


def test_gromacs_params_new_barostat_fields() -> None:
    p = GromacsParams(pcoupl=Barostat.PARRINELLO_RAHMAN, ref_p=1.0, tau_p=2.0, compressibility=4.5e-5)
    mapping = p.to_mapping()

    assert mapping["pcoupl"] == "Parrinello-Rahman"
    assert mapping["ref-p"] == 1.0
    assert mapping["tau-p"] == 2.0
    assert mapping["compressibility"] == pytest.approx(4.5e-5)


def test_gromacs_params_new_velocity_fields() -> None:
    p = GromacsParams(gen_vel=VelocityGeneration.YES, gen_temp=310.0, gen_seed=42)
    mapping = p.to_mapping()

    assert mapping["gen-vel"] == "yes"
    assert mapping["gen-temp"] == 310.0
    assert mapping["gen-seed"] == 42


def test_gromacs_params_new_constraints_field() -> None:
    p = GromacsParams(constraints=Constraints.HYDROGENS_BONDS)
    mapping = p.to_mapping()

    assert mapping["constraints"] == "h-bonds"


def test_gromacs_params_roundtrip_new_fields() -> None:
    """from_mapping → to_mapping round-trip for newly added fields."""
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
    protein.write_text("")

    cfg = RunConfig(system=SystemConfig(protein=protein))

    with pytest.raises(ValueError, match=r"system\.ligand"):
        cfg.to_parametrization_input(tmp_path / "work")


# ---------------------------------------------------------------------------
# CLI argument parsing (no side effects — just parser)
# ---------------------------------------------------------------------------


def test_cli_parses_config_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    protein = tmp_path / "protein.pdb"
    protein.write_text("")
    cfg_path.write_text(f'[system]\nprotein = "{protein}"\n')

    with mock.patch("gbsa_pipeline.cli.run_pipeline") as mock_run:
        cli_main([str(cfg_path)])
        mock_run.assert_called_once()
        _, output_dir = mock_run.call_args.args
        assert output_dir == tmp_path / "gbsa_output"


def test_cli_custom_output_dir(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    protein = tmp_path / "protein.pdb"
    protein.write_text("")
    cfg_path.write_text(f'[system]\nprotein = "{protein}"\n')
    out = tmp_path / "custom_out"

    with mock.patch("gbsa_pipeline.cli.run_pipeline") as mock_run:
        cli_main([str(cfg_path), "-o", str(out)])
        _, output_dir = mock_run.call_args.args
        assert output_dir == out
