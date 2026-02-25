"""Tests for frcmod_parametrization: AmberFFInput/build_amber_ff_xml and AmberInput/load_amber_complex.

Unit tests cover input validation (no heavy imports needed).
Functional tests exercise build_amber_ff_xml against the MCPB.py testdata.
Integration tests use the pre-built ZN1_Blimp1 AMBER files for the full
prmtop → GROMACS round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import parmed as pmd
import pytest
from openff.toolkit import Molecule
from openmm import LangevinIntegrator, unit
from openmm.app import ForceField, NoCutoff, PDBFile, Simulation
from openmm.app.modeller import Modeller
from openmmforcefields.generators import GAFFTemplateGenerator
from parmed.modeller import ResidueTemplateContainer

from gbsa_pipeline.frcmod_parametrization import (
    AmberFFInput,
    AmberInput,
    build_amber_ff_xml,
    load_amber_complex,
)
from gbsa_pipeline.parametrization import (
    ParametrizationConfig,
    ParametrizationInput,
    parametrize,
)
from gbsa_pipeline.parametrization_enum import ChargeMethod, LigandFF, ProteinFF

# Expected values read from the prmtop via ParmEd
_EXPECTED_ATOMS = 12814
_EXPECTED_RESIDUES = 826
_ZN_TYPE = "M1"
_ZN_CHARGE = 0.4314
_ZN_EPSILON_KCAL = 0.01492  # kcal/mol → stored by ParmEd
_ZN_RMIN_ANG = 1.395  # Å (Rmin/2)

# ---------------------------------------------------------------------------
# MCPB testdata paths
# ---------------------------------------------------------------------------

_TD = Path("tests/testdata").resolve()
_FRCMOD = _TD / "Model4_mcpbpy.frcmod"
_MOL2S = (
    _TD / "CM1.mol2",
    _TD / "CM2.mol2",
    _TD / "HD1.mol2",
    _TD / "HD2.mol2",
    _TD / "ZN1.mol2",
)

# ---------------------------------------------------------------------------
# Pre-built AMBER test data (Zn-finger Blimp1, MCPB.py, no water box)
# ---------------------------------------------------------------------------

DRY_PRMTOP = _TD / "Model4_dry.prmtop"
DRY_INPCRD = _TD / "Model4_dry.inpcrd"
# ---------------------------------------------------------------------------
# AmberInput validation (unit tests)
# ---------------------------------------------------------------------------


def test_amber_input_missing_prmtop_raises(tmp_path: Path) -> None:
    """AmberInput raises ValueError when prmtop does not exist."""
    inpcrd = tmp_path / "x.inpcrd"
    inpcrd.touch()
    with pytest.raises(ValueError, match="path_not_file"):
        AmberInput(prmtop=Path("nonexistent.prmtop"), inpcrd=inpcrd)


def test_amber_input_missing_inpcrd_raises(tmp_path: Path) -> None:
    """AmberInput raises ValueError when inpcrd does not exist."""
    prmtop = tmp_path / "x.prmtop"
    prmtop.touch()
    with pytest.raises(ValueError, match="path_not_file"):
        AmberInput(prmtop=prmtop, inpcrd=Path("nonexistent.inpcrd"))


def test_amber_input_output_dir_defaults_to_none(tmp_path: Path) -> None:
    """AmberInput.output_dir defaults to None."""
    prmtop = tmp_path / "x.prmtop"
    inpcrd = tmp_path / "x.inpcrd"
    prmtop.touch()
    inpcrd.touch()
    assert AmberInput(prmtop=prmtop, inpcrd=inpcrd).output_dir is None


# ---------------------------------------------------------------------------
# AmberFFInput validation (unit tests)
# ---------------------------------------------------------------------------


def test_amber_ff_input_missing_frcmod_raises(tmp_path: Path) -> None:
    """AmberFFInput raises ValueError when a frcmod file does not exist."""
    with pytest.raises(ValueError, match="Files not found"):
        AmberFFInput(frcmod_files=(tmp_path / "nonexistent.frcmod",))


def test_amber_ff_input_missing_mol2_raises(tmp_path: Path) -> None:
    """AmberFFInput raises ValueError when a mol2 file does not exist."""
    with pytest.raises(ValueError, match="Files not found"):
        AmberFFInput(residue_mol2s=(tmp_path / "nonexistent.mol2",))


def test_amber_ff_input_defaults() -> None:
    """AmberFFInput defaults: empty tuples, FF14SB, GAFF2, no output_xml."""
    inp = AmberFFInput()
    assert inp.frcmod_files == ()
    assert inp.residue_mol2s == ()
    assert inp.protein_ff == ProteinFF.FF14SB
    assert inp.ligand_ff == LigandFF.GAFF2
    assert inp.output_xml is None


# ---------------------------------------------------------------------------
# build_amber_ff_xml — functional tests (use MCPB testdata, no antechamber)
# ---------------------------------------------------------------------------


def test_build_amber_ff_xml_creates_file(tmp_path: Path) -> None:
    """build_amber_ff_xml writes a non-empty XML file."""
    xml = build_amber_ff_xml(
        AmberFFInput(
            frcmod_files=(_FRCMOD,),
            residue_mol2s=_MOL2S,
            protein_ff=ProteinFF.FF14SB,
            ligand_ff=LigandFF.GAFF,
            output_xml=tmp_path / "combined.xml",
        )
    )
    assert xml.exists()
    assert xml.stat().st_size > 0


def test_build_amber_ff_xml_no_missing_atomtypes(tmp_path: Path) -> None:
    """All atom types referenced by residue templates are defined in the XML."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    xml = build_amber_ff_xml(
        AmberFFInput(
            frcmod_files=(_FRCMOD,),
            residue_mol2s=_MOL2S,
            protein_ff=ProteinFF.FF14SB,
            ligand_ff=LigandFF.GAFF,
            output_xml=tmp_path / "combined.xml",
        )
    )
    tree = ET.parse(xml)  # noqa: S314
    defined = {t.get("name") for t in tree.findall(".//AtomTypes/Type")} - {None}
    used: set[str] = {t for a in tree.findall(".//Residues/Residue/Atom") if (t := a.get("type")) is not None}
    missing_types = used - defined
    assert not missing_types, f"Missing atom types: {sorted(missing_types)}"


def test_build_amber_ff_xml_loadable_by_openmm(tmp_path: Path) -> None:
    """The generated XML can be loaded by OpenMM's ForceField without errors."""
    from openmm.app import ForceField  # noqa: PLC0415

    xml = build_amber_ff_xml(
        AmberFFInput(
            frcmod_files=(_FRCMOD,),
            residue_mol2s=_MOL2S,
            protein_ff=ProteinFF.FF14SB,
            ligand_ff=LigandFF.GAFF,
            output_xml=tmp_path / "combined.xml",
        )
    )
    # Should not raise; amber14-all.xml provides standard residue templates.
    ForceField("amber14-all.xml", str(xml))


def test_build_amber_ff_xml_output_xml_none_creates_temp() -> None:
    """When output_xml is None, a file in a temp directory is returned."""
    xml = build_amber_ff_xml(
        AmberFFInput(
            frcmod_files=(_FRCMOD,),
            residue_mol2s=_MOL2S,
            protein_ff=ProteinFF.FF14SB,
            ligand_ff=LigandFF.GAFF,
        )
    )
    assert xml.exists()
    assert xml.suffix == ".xml"


# ---------------------------------------------------------------------------
# MCPB system tests: build_amber_ff_xml → ForceField → createSystem
# (mirrors tt_test.py — no GAFF, protein-only PDB)
# ---------------------------------------------------------------------------


def test_build_amber_ff_xml_create_system(tmp_path: Path) -> None:
    """build_amber_ff_xml XML + amber14-all.xml can createSystem on MCPB PDB."""
    from openmm.app import ForceField, NoCutoff, PDBFile  # noqa: PLC0415

    xml = build_amber_ff_xml(
        AmberFFInput(
            frcmod_files=(_FRCMOD,),
            residue_mol2s=_MOL2S,
            protein_ff=ProteinFF.FF14SB,
            ligand_ff=LigandFF.GAFF,
            output_xml=tmp_path / "combined.xml",
        )
    )
    pdb = PDBFile(str(_TD / "Model4_mcpbpy.pdb"))
    ff = ForceField("amber14-all.xml", str(xml))
    system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None)
    assert system.getNumParticles() == pdb.topology.getNumAtoms()
    assert system.getNumForces() > 0


@pytest.mark.integration
def test_build_amber_ff_xml_energy_minimization(tmp_path: Path) -> None:
    """System built from the generated XML can be energy-minimized."""
    from openmm import LangevinIntegrator, Platform, unit  # noqa: PLC0415
    from openmm.app import ForceField, NoCutoff, PDBFile, Simulation  # noqa: PLC0415

    xml = build_amber_ff_xml(
        AmberFFInput(
            frcmod_files=(_FRCMOD,),
            residue_mol2s=_MOL2S,
            protein_ff=ProteinFF.FF14SB,
            ligand_ff=LigandFF.GAFF,
            output_xml=tmp_path / "combined.xml",
        )
    )
    pdb = PDBFile(str(_TD / "Model4_mcpbpy.pdb"))
    ff = ForceField("amber14-all.xml", str(xml))
    system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None)

    integrator = LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = Simulation(pdb.topology, system, integrator, Platform.getPlatformByName("CPU"))
    simulation.context.setPositions(pdb.positions)

    initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy(maxIterations=100)
    final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    assert final_energy < initial_energy


@pytest.mark.integration
def test_parametrize_mcpb_with_ligand(tmp_path: Path) -> None:
    """Test that frcmod → XML → parametrization with a ligand produces valid GROMACS files.

    Checks all force field parameters are preserved in the conversion.
    """
    xml = build_amber_ff_xml(
        AmberFFInput(
            frcmod_files=(_FRCMOD,),
            residue_mol2s=_MOL2S,
            protein_ff=ProteinFF.FF14SB,
            ligand_ff=LigandFF.GAFF,
            output_xml=tmp_path / "mcpb.xml",
        )
    )

    result = parametrize(
        ParametrizationInput(
            protein_pdb=_TD / "Model4_mcpbpy.pdb",
            ligand_sdf=_TD / "complex3.sdf",
            config=ParametrizationConfig(
                protein_ff=ProteinFF.FF14SB,
                ligand_ff=LigandFF.GAFF,
                extra_ff_files=(xml,),
            ),
            work_dir=tmp_path,
        )
    )

    assert result.gro_file.exists()
    assert result.gro_file.stat().st_size > 0
    assert result.top_file.exists()
    assert result.top_file.stat().st_size > 0

    struct = pmd.load_file(str(result.top_file), xyz=str(result.gro_file))

    # Check that all custom MCPB residues exist
    residue_names = {r.name for r in struct.residues}
    assert {"CM1", "CM2", "HD1", "HD2", "ZN1"} <= residue_names

    # Zinc: charge, epsilon, and Rmin/2 should match the frcmod parameters
    zn_atoms = [a for a in struct.atoms if a.atomic_number == 30]
    assert len(zn_atoms) == 1, "expected exactly one Zn atom"
    zn = zn_atoms[0]
    assert abs(zn.charge - 0.431432) < 1e-4, f"Zn charge {zn.charge} != 0.431432"
    assert abs(zn.epsilon - 0.014917) < 1e-4, f"Zn epsilon {zn.epsilon} != 0.014917 kcal/mol"
    assert abs(zn.rmin - 1.3950) < 1e-3, f"Zn Rmin/2 {zn.rmin} != 1.395 Å"

    # Check that all force field parameters are correctly applied:
    for residue in struct.residues:
        for atom in residue.atoms:
            # Assert atom masses (check for typical atom names and expected masses)
            assert atom.mass is not None, f"Mass not assigned for atom {atom.name} in residue {residue.name}"

    for bond in struct.bonds:
        assert bond.type is not None, f"Bond type not assigned for bond between {bond.atom1.name} and {bond.atom2.name}"

    for angle in struct.angles:
        assert angle.type is not None, (
            f"Angle type not assigned for angle between {angle.atom1.name}, {angle.atom2.name}, {angle.atom3.name}"
        )

    for dihedral in struct.dihedrals:
        assert dihedral.type is not None, (
            f"Torsion type not assigned for dihedral between {dihedral.atom1.name}, {dihedral.atom2.name}, {dihedral.atom3.name}, {dihedral.atom4.name}"
        )

    for improper in struct.impropers:
        assert improper.type is not None, (
            f"Improper type not assigned for improper between {improper.atom1.name}, {improper.atom2.name}, {improper.atom3.name}, {improper.atom4.name}"
        )

    # NONBON (Non-bonded parameters such as charge and epsilon for each atom)
    for atom in struct.atoms:
        assert atom.charge is not None, f"Charge not assigned for atom {atom.name}"
        assert atom.epsilon is not None, f"Epsilon not assigned for atom {atom.name}"


###
#   BASE TESTS USING PURE OPENMM JUST TO VALIDATE IT WORKS
###


@pytest.mark.integration
def test_load_frcmod_openmm(tmp_path: Path) -> None:
    """Base test used to identify how to load frcmod in openmm.

    This uses pure OpenMM and parmed and not the gbsa library, it was used to find out how to properly make the system work. It is kept for reference.
    """
    tf = Path("tests/testdata").resolve()
    residues = (
        tf / "CM1.mol2",
        tf / "CM2.mol2",
        tf / "HD1.mol2",
        tf / "HD2.mol2",
        tf / "ZN1.mol2",
    )

    pdb = PDBFile(str(tf / "Model4_mcpbpy.pdb"))
    protein_xmls = "amber14-all.xml"
    parm = Path("/home/davide/Lavoro/libraries/gbsa-pipeline/.pixi/envs/default/dat/leap/parm")

    base_parm = pmd.amber.AmberParameterSet(
        str(parm / "parm19.dat"),
        str(parm / "frcmod.ff14SB"),
        str(parm / "gaff.dat"),
        str(tf / "Model4_mcpbpy.frcmod"),
    )
    ff = pmd.openmm.OpenMMParameterSet.from_parameterset(base_parm)

    for mol2_file in residues:
        mol2 = pmd.load_file(str(mol2_file))
        if isinstance(mol2, ResidueTemplateContainer):
            ff.residues.update(mol2.to_library())
        else:
            ff.residues[mol2.name] = mol2

    xml_path = tmp_path / "combined.xml"
    ff.write(str(xml_path), write_unused=True)
    assert xml_path.exists()

    forcefield = ForceField(protein_xmls, str(xml_path))

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
    )

    assert system.getNumParticles() == pdb.topology.getNumAtoms()
    assert system.getNumForces() > 0

    temperature = unit.Quantity(300, unit.kelvin)
    friction = unit.Quantity(1.0, 1 / unit.picosecond)
    timestep = unit.Quantity(0.002, unit.picoseconds)

    integrator = LangevinIntegrator(temperature, friction, timestep)

    simulation = Simulation(pdb.topology, system, integrator)

    simulation.context.setPositions(pdb.positions)

    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy()
    simulation.minimizeEnergy(maxIterations=100)

    state = simulation.context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy()

    assert final_energy < initial_energy
    assert not unit.is_quantity(final_energy) or (final_energy._value == final_energy._value)

    # Test saving to gromacs output

    structure = pmd.openmm.load_topology(pdb.topology, system, pdb.positions)

    gro_file = tmp_path / "complex.gro"
    top_file = tmp_path / "complex.top"
    structure.save(str(top_file), format="gromacs")
    structure.save(str(gro_file))

    struct = pmd.load_file(str(top_file), xyz=str(gro_file))
    assert len(struct.atoms) == 12814
    assert len(struct.residues) == 826

    residue_names = {r.name for r in struct.residues}
    assert {"CM1", "CM2", "HD1", "HD2", "ZN1"} <= residue_names

    # Zinc: charge, epsilon, and Rmin/2 should match the frcmod parameters
    zn_atoms = [a for a in struct.atoms if a.atomic_number == 30]
    assert len(zn_atoms) == 1, "expected exactly one Zn atom"
    zn = zn_atoms[0]
    assert abs(zn.charge - 0.431432) < 1e-4, f"Zn charge {zn.charge} != 0.431432"
    assert abs(zn.epsilon - 0.014917) < 1e-4, f"Zn epsilon {zn.epsilon} != 0.014917 kcal/mol"
    assert abs(zn.rmin - 1.3950) < 1e-3, f"Zn Rmin/2 {zn.rmin} != 1.395 Å"

    # Check bond parameters are present
    for bond in struct.bonds:
        assert bond.type is not None, f"Bond type missing for bond between {bond.atom1.name} and {bond.atom2.name}"

    # Check angles, torsions, and improper parameters
    for angle in struct.angles:
        assert angle.type is not None, (
            f"Angle type missing for angle between {angle.atom1.name}, {angle.atom2.name}, {angle.atom3.name}"
        )

    for dihedral in struct.dihedrals:
        assert dihedral.type is not None, (
            f"Torsion type missing for dihedral between {dihedral.atom1.name}, {dihedral.atom2.name}, {dihedral.atom3.name}, {dihedral.atom4.name}"
        )

    for improper in struct.impropers:
        assert improper.type is not None, (
            f"Improper type missing for improper between {improper.atom1.name}, {improper.atom2.name}, {improper.atom3.name}, {improper.atom4.name}"
        )


@pytest.mark.integration
def test_parametrize_mcpb_with_ligand_openmm(tmp_path: Path) -> None:
    """Base test used to identify how to load frcmod in openmm.

    This uses pure OpenMM and parmed and not the gbsa library, it was used to find out how to properly make the system work. It is kept for reference.
    """
    tf = Path("tests/testdata").resolve()
    residues = (
        tf / "CM1.mol2",
        tf / "CM2.mol2",
        tf / "HD1.mol2",
        tf / "HD2.mol2",
        tf / "ZN1.mol2",
    )

    pdb = PDBFile(str(tf / "Model4_mcpbpy.pdb"))
    protein_xmls = "amber14-all.xml"
    parm = Path("/home/davide/Lavoro/libraries/gbsa-pipeline/.pixi/envs/default/dat/leap/parm")

    base_parm = pmd.amber.AmberParameterSet(
        str(parm / "parm19.dat"),
        str(parm / "frcmod.ff14SB"),
        str(parm / "gaff.dat"),
        str(tf / "Model4_mcpbpy.frcmod"),
    )
    ff = pmd.openmm.OpenMMParameterSet.from_parameterset(base_parm)

    for mol2_file in residues:
        mol2 = pmd.load_file(str(mol2_file))
        if isinstance(mol2, ResidueTemplateContainer):
            ff.residues.update(mol2.to_library())
        else:
            ff.residues[mol2.name] = mol2

    xml_path = tmp_path / "combined.xml"
    ff.write(str(xml_path), write_unused=True)
    assert xml_path.exists()

    ligand = Molecule.from_file(str(tf / "complex3.sdf"))
    if not ligand.conformers:
        raise ValueError(
            f"Ligand SDF '{tf / 'complex3.sdf'}' contains no 3-D conformers. "
            "Provide an SDF file with embedded 3-D coordinates."
        )

    kwargs: dict[str, Any] = {
        "partial_charge_method": ChargeMethod.AM1BCC.value,
        "normalize_partial_charges": False,
        "use_conformers": ligand.conformers,
    }
    ligand.assign_partial_charges(**kwargs)
    forcefield = ForceField(protein_xmls, str(xml_path))

    gaff = GAFFTemplateGenerator(molecules=[ligand], forcefield="gaff-1.81", cache=None)
    forcefield.registerTemplateGenerator(gaff.generator)

    # --- Combine protein + ligand --------------------------------------
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.add(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
    )

    assert system.getNumParticles() == modeller.topology.getNumAtoms()
    assert system.getNumForces() > 0

    temperature = unit.Quantity(300, unit.kelvin)
    friction = unit.Quantity(1.0, 1 / unit.picosecond)
    timestep = unit.Quantity(0.002, unit.picoseconds)

    integrator = LangevinIntegrator(temperature, friction, timestep)

    simulation = Simulation(modeller.topology, system, integrator)

    simulation.context.setPositions(modeller.positions)

    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy()
    simulation.minimizeEnergy(maxIterations=100)

    state = simulation.context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy()

    assert final_energy < initial_energy
    assert not unit.is_quantity(final_energy) or (final_energy._value == final_energy._value)


# ---------------------------------------------------------------------------
# Load ZN1_Blimp1 pre-built AMBER files
# ---------------------------------------------------------------------------


def test_load_amber_complex_creates_gromacs_files(tmp_path: Path) -> None:
    """load_amber_complex writes non-empty .gro and .top files."""
    result = load_amber_complex(AmberInput(prmtop=DRY_PRMTOP, inpcrd=DRY_INPCRD, output_dir=tmp_path))
    assert result.gro_file.exists()
    assert result.gro_file.stat().st_size > 0
    assert result.top_file.exists()
    assert result.top_file.stat().st_size > 0


def test_load_amber_complex_atom_and_residue_count(tmp_path: Path) -> None:
    """Exported topology has the expected number of atoms and residues."""
    result = load_amber_complex(AmberInput(prmtop=DRY_PRMTOP, inpcrd=DRY_INPCRD, output_dir=tmp_path))
    struct = pmd.load_file(str(result.top_file), xyz=str(result.gro_file))
    assert len(struct.atoms) == _EXPECTED_ATOMS
    assert len(struct.residues) == _EXPECTED_RESIDUES


def test_load_amber_complex_zn_type(tmp_path: Path) -> None:
    """Exported topology contains exactly one Zn atom with type M1."""
    result = load_amber_complex(AmberInput(prmtop=DRY_PRMTOP, inpcrd=DRY_INPCRD, output_dir=tmp_path))
    struct = pmd.load_file(str(result.top_file), xyz=str(result.gro_file))
    zn_atoms = [a for a in struct.atoms if a.atomic_number == 30]
    assert len(zn_atoms) == 1
    assert zn_atoms[0].type == _ZN_TYPE


def test_load_amber_complex_zn_charge(tmp_path: Path) -> None:
    """Zn RESP charge survives the prmtop → GROMACS round-trip."""
    result = load_amber_complex(AmberInput(prmtop=DRY_PRMTOP, inpcrd=DRY_INPCRD, output_dir=tmp_path))
    struct = pmd.load_file(str(result.top_file), xyz=str(result.gro_file))
    zn = next(a for a in struct.atoms if a.atomic_number == 30)
    assert abs(zn.charge - _ZN_CHARGE) < 1e-4


def test_load_amber_complex_zn_lj_params(tmp_path: Path) -> None:
    """Zn LJ parameters survive the prmtop → GROMACS round-trip."""
    result = load_amber_complex(AmberInput(prmtop=DRY_PRMTOP, inpcrd=DRY_INPCRD, output_dir=tmp_path))
    struct = pmd.load_file(str(result.top_file), xyz=str(result.gro_file))
    zn = next(a for a in struct.atoms if a.atomic_number == 30)
    assert abs(zn.epsilon - _ZN_EPSILON_KCAL) < 1e-4
    assert abs(zn.rmin - _ZN_RMIN_ANG) < 1e-3
