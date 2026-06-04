"""Parametrize protein-ligand complexes."""

from __future__ import annotations

import contextlib
import logging
import pickle
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import BioSimSpace as BSS
import parmed as pmd
from openff.toolkit.topology import Molecule
from openmm.app import ForceField, Modeller, NoCutoff
from openmmforcefields.generators import GAFFTemplateGenerator
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from BioSimSpace._SireWrappers import System

from gbsa_pipeline._openmm_utils import _delete_residues_by_name, _load_pdb_as_modeller
from gbsa_pipeline.parametrization_enum import ChargeMethod, LigandFF, ProteinFF

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

PDB_ELEMENT_START = 76
PDB_ELEMENT_END = 78
PDB_ATOM_NAME_WIDTH = 4
THREE_CHAR_ATOM_NAME_LENGTH = 3
MIN_GREEK_ATOM_NAME_LENGTH = 2
MIN_PDBQT_ATOM_FIELDS = 9
MIN_PDBQT_BOND_FIELDS = 4

# ---------------------------------------------------------------------------
# Mol2 cap-stripping helpers (shared between tleap and legacy XML paths)
# ---------------------------------------------------------------------------

# GAFF nitrogen and carbon atom type sets, used to detect ACE/NME cap atoms.
_GAFF_N_TYPES = {
    "n",
    "n1",
    "n2",
    "n3",
    "n4",
    "na",
    "nb",
    "nc",
    "nd",
    "nh",
    "no",
    "ns",
    "nt",
    "nu",
    "nv",
}
_GAFF_C_TYPES = {
    "c",
    "ca",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "cc",
    "cd",
    "ce",
    "cf",
    "cg",
    "ch",
    "ci",
    "ck",
    "cm",
    "cn",
    "co",
    "cp",
    "cq",
    "cr",
    "cu",
    "cv",
    "cw",
    "cx",
    "cy",
    "cz",
}

# AMBER ff14SB backbone atom types for residue templates.
_AMBER_BACKBONE_TYPE = {
    "backbone_N": "N",
    "backbone_H": "H",
    "backbone_CA": "CX",
    "backbone_HA": "H1",
    "backbone_CB": "2C",
    "backbone_HB": "H1",
    "backbone_C": "C",
    "backbone_O": "O",
}

_WATER_RESIDUE_NAMES_SET = {"HOH", "WAT", "TIP3", "TIP3P", "SOL"}


def _pdb_sidechain_names_by_depth(protein_pdb: Path, resname: str) -> dict[tuple[str, int], list[str]]:
    """Return a mapping (element, depth_from_CA) → [atom_names] for a residue in the PDB."""
    pdb_atoms: list[tuple[str, str]] = []
    for line in protein_pdb.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rname = line[17:20].strip()
        if rname != resname:
            continue
        aname = line[12:16].strip()
        element = line[PDB_ELEMENT_START:PDB_ELEMENT_END].strip() if len(line) >= PDB_ELEMENT_END else aname[:1]
        pdb_atoms.append((aname, element))

    backbone_names = {
        "N",
        "H",
        "CA",
        "HA",
        "C",
        "O",
        "HN",
        "1H",
        "2H",
        "3H",
        "H1",
        "H2",
        "H3",
        "OXT",
        "HB",
        "HB2",
        "HB3",
    }
    greek = {"A": 0, "B": 1, "G": 2, "D": 3, "E": 4, "Z": 5, "H": 6}
    depth: dict[str, int] = {"CA": 0}
    for aname, _ in pdb_atoms:
        if len(aname) >= MIN_GREEK_ATOM_NAME_LENGTH and aname[1:2].upper() in greek:
            depth[aname] = greek[aname[1:2].upper()]
        else:
            depth[aname] = 0

    result: dict[tuple[str, int], list[str]] = {}
    for aname, elem in pdb_atoms:
        if aname in backbone_names:
            continue
        key = (elem.upper(), depth.get(aname, 99))
        result.setdefault(key, []).append(aname)
    return result


def _gaff_type_to_element(gaff_type: str) -> str:
    """Map a GAFF atom type string to its element symbol."""
    t = gaff_type.lower()
    if t.startswith("cl"):
        return "Cl"
    if t.startswith("br"):
        return "Br"
    if t.startswith("s"):
        return "S"
    if t.startswith("o"):
        return "O"
    if t.startswith("n"):
        return "N"
    if t.startswith("p"):
        return "P"
    if t.startswith("h"):
        return "H"
    if t.startswith("c"):
        return "C"
    if t.startswith("f"):
        return "F"
    return t[0].upper()


def _strip_mol2_dipeptide_caps_parmed(
    mol2_path: Path,
    output_mol2: Path,
    protein_pdb: Path | None = None,
) -> Path:
    """ParmEd-based cap stripping: load mol2, rename backbone atoms, strip ACE/NME, save."""
    import parmed as pmd

    structure = pmd.load_file(str(mol2_path))

    res_names = {r.name.upper() for r in structure.residues}
    if "ACE" not in res_names:
        raise ValueError(f"No ACE residue found by ParmEd in {mol2_path}")

    cap_idx: set[int] = {a.idx for a in structure.atoms if a.residue.name.upper() in {"ACE", "NME"}}

    adj_pmd: dict[int, list[pmd.Atom]] = {a.idx: [] for a in structure.atoms}
    for bond in structure.bonds:
        adj_pmd[bond.atom1.idx].append(bond.atom2)
        adj_pmd[bond.atom2.idx].append(bond.atom1)

    backbone_n = None
    for atom in structure.atoms:
        if atom.idx in cap_idx or atom.type.lower() not in _GAFF_N_TYPES:
            continue
        if any(nb.idx in cap_idx for nb in adj_pmd[atom.idx]):
            backbone_n = atom
            break
    if backbone_n is None:
        raise ValueError(f"Could not identify backbone N in {mol2_path}")

    backbone_ca = next(
        (nb for nb in adj_pmd[backbone_n.idx] if nb.idx not in cap_idx and nb.type.lower() == "c3"),
        None,
    )
    if backbone_ca is None:
        raise ValueError(f"Could not identify backbone CA in {mol2_path}")

    backbone_c = backbone_o = None
    for nb in adj_pmd[backbone_ca.idx]:
        if nb is backbone_n or nb.idx in cap_idx:
            continue
        if nb.type.lower() in _GAFF_C_TYPES and nb.type.lower() != "c3":
            o_nbs = [x for x in adj_pmd[nb.idx] if x.type.lower() == "o" and x is not backbone_ca]
            if o_nbs:
                backbone_c = nb
                backbone_o = o_nbs[0]
                break
    if backbone_c is None or backbone_o is None:
        raise ValueError(f"Could not identify backbone C/O in {mol2_path}")

    backbone_ha = next(
        (nb for nb in adj_pmd[backbone_ca.idx] if nb.type.lower() == "h1" and nb.idx not in cap_idx),
        None,
    )
    backbone_h = next(
        (nb for nb in adj_pmd[backbone_n.idx] if nb.type.lower() in {"hn", "h"} and nb.idx not in cap_idx),
        None,
    )
    backbone_cb = next(
        (
            nb for nb in adj_pmd[backbone_ca.idx]
            if nb is not backbone_n and nb is not backbone_c
            and nb.idx not in cap_idx and nb.type.lower() == "c3"
            and nb is not backbone_ha
        ),
        None,
    )
    backbone_hb_atoms: list[pmd.Atom] = []
    if backbone_cb is not None:
        backbone_hb_atoms = [
            nb for nb in adj_pmd[backbone_cb.idx]
            if nb.type.lower() in {"h1", "hc", "hx"} and nb.idx not in cap_idx
        ]

    backbone_n.name = "N";   backbone_n.type   = _AMBER_BACKBONE_TYPE["backbone_N"]
    backbone_ca.name = "CA"; backbone_ca.type  = _AMBER_BACKBONE_TYPE["backbone_CA"]
    backbone_c.name = "C";   backbone_c.type   = _AMBER_BACKBONE_TYPE["backbone_C"]
    backbone_o.name = "O";   backbone_o.type   = _AMBER_BACKBONE_TYPE["backbone_O"]
    if backbone_ha:
        backbone_ha.name = "HA"; backbone_ha.type = _AMBER_BACKBONE_TYPE["backbone_HA"]
    if backbone_h:
        backbone_h.name = "H";  backbone_h.type  = _AMBER_BACKBONE_TYPE["backbone_H"]
    if backbone_cb:
        backbone_cb.name = "CB"; backbone_cb.type = _AMBER_BACKBONE_TYPE["backbone_CB"]
    for i, hb in enumerate(backbone_hb_atoms, start=2):
        hb.name = f"HB{i}"; hb.type = _AMBER_BACKBONE_TYPE["backbone_HB"]

    resname = structure.residues[0].name.strip() if structure.residues else "UNK"
    pdb_names_by_depth: dict[tuple[str, int], list[str]] = {}
    if protein_pdb is not None:
        with contextlib.suppress(Exception):
            pdb_names_by_depth = _pdb_sidechain_names_by_depth(protein_pdb, resname)

    if pdb_names_by_depth:
        named_idx = {a.idx for a in [backbone_n, backbone_ca, backbone_c, backbone_o, backbone_ha, backbone_h, backbone_cb, *backbone_hb_atoms] if a is not None}
        mol2_depth: dict[int, int] = {backbone_ca.idx: 0}
        bfs_q = [backbone_ca]
        while bfs_q:
            node = bfs_q.pop(0)
            for nb in adj_pmd[node.idx]:
                if nb.idx not in mol2_depth and nb.idx not in cap_idx:
                    mol2_depth[nb.idx] = mol2_depth[node.idx] + 1
                    bfs_q.append(nb)

        sc_by_elem_depth: dict[tuple[str, int], list[pmd.Atom]] = {}
        for atom in structure.atoms:
            if atom.idx in cap_idx or atom.idx in named_idx:
                continue
            elem = _gaff_type_to_element(atom.type)
            sc_by_elem_depth.setdefault((elem, mol2_depth.get(atom.idx, 99)), []).append(atom)

        pdb_names_used: set[str] = set()
        for (elem, depth), sc_atoms in sorted(sc_by_elem_depth.items()):
            available = [n for n in pdb_names_by_depth.get((elem, depth), []) if n not in pdb_names_used]
            for atom, pdb_name in zip(sc_atoms, available):
                atom.name = pdb_name
                pdb_names_used.add(pdb_name)

    structure.strip(":ACE,NME")
    output_mol2.parent.mkdir(parents=True, exist_ok=True)
    structure.save(str(output_mol2), format="mol2", overwrite=True)
    return output_mol2


def _strip_mol2_dipeptide_caps(
    mol2_path: Path,
    output_mol2: Path,
    protein_pdb: Path | None = None,
) -> Path:
    """Strip ACE/NME caps from a capped-dipeptide mol2, trying ParmEd first.

    Falls back to raw-text parsing when ParmEd cannot read the GAFF mol2 variant.
    """
    try:
        return _strip_mol2_dipeptide_caps_parmed(mol2_path, output_mol2, protein_pdb)
    except Exception:
        return _strip_mol2_dipeptide_caps_text(mol2_path, output_mol2, protein_pdb)


def _strip_mol2_dipeptide_caps_text(
    mol2_path: Path,
    output_mol2: Path,
    protein_pdb: Path | None = None,
) -> Path:
    """Strip ACE/NME caps from a capped-dipeptide mol2 and rename backbone atoms.

    RESP charges are often derived on ACE-RES-NME capped dipeptides. This
    function removes the cap atoms and renames backbone atoms (N, CA, CB, C, O)
    to AMBER ff14SB convention so the mol2 can serve as an embedded residue
    template for tleap. Mol2 files that are already residue templates (e.g.
    CS1-4 from MCPB.py) will raise ValueError (no ACE cap found), and the
    caller should fall back to using the original mol2 unchanged.
    """
    text = mol2_path.read_text()
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("@<TRIPOS>"):
            current = line[9:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    atoms: dict[int, dict] = {}
    for line in sections.get("ATOM", []):
        parts = line.split()
        if len(parts) < MIN_PDBQT_ATOM_FIELDS:
            continue
        aid = int(parts[0])
        atoms[aid] = {
            "id": aid,
            "name": parts[1],
            "x": parts[2],
            "y": parts[3],
            "z": parts[4],
            "type": parts[5],
            "subst_id": parts[6],
            "subst_name": parts[7],
            "charge": parts[8],
        }

    bonds: list[tuple[int, int, str]] = []
    adj: dict[int, list[int]] = {a: [] for a in atoms}
    for line in sections.get("BOND", []):
        parts = line.split()
        if len(parts) < MIN_PDBQT_BOND_FIELDS:
            continue
        a1, a2, bt = int(parts[1]), int(parts[2]), parts[3]
        bonds.append((a1, a2, bt))
        adj[a1].append(a2)
        adj[a2].append(a1)

    # Identify backbone N bonded to ACE cap carbonyl C.
    backbone_n_id = None
    ace_cap_c_id = None
    for aid, atom in atoms.items():
        if atom["type"].lower() not in _GAFF_N_TYPES:
            continue
        for nb in adj[aid]:
            nb_atom = atoms[nb]
            if nb_atom["type"].lower() not in _GAFF_C_TYPES:
                continue
            nb_neighbors = adj[nb]
            has_o = any(atoms[x]["type"].lower() == "o" for x in nb_neighbors if x != aid)
            has_methyl = any(atoms[x]["type"].lower() == "c3" for x in nb_neighbors if x != aid)
            if has_o and has_methyl:
                backbone_n_id = aid
                ace_cap_c_id = nb
                break
        if backbone_n_id is not None:
            break

    if backbone_n_id is None or ace_cap_c_id is None:
        raise ValueError(f"Could not identify backbone N in {mol2_path}. Expected a capped dipeptide (ACE-RES-NME).")

    # BFS to collect ACE cap atoms.
    ace_atoms: set[int] = set()
    queue = [ace_cap_c_id]
    while queue:
        node = queue.pop()
        if node in ace_atoms or node == backbone_n_id:
            continue
        ace_atoms.add(node)
        for nb in adj[node]:
            if nb not in ace_atoms and nb != backbone_n_id:
                queue.append(nb)

    backbone_ca_id = next(
        (nb for nb in adj[backbone_n_id] if nb not in ace_atoms and atoms[nb]["type"].lower() == "c3"),
        None,
    )
    if backbone_ca_id is None:
        raise ValueError(f"Could not identify backbone CA in {mol2_path}.")

    backbone_c_id = None
    backbone_o_id = None
    for nb in adj[backbone_ca_id]:
        if nb == backbone_n_id or nb in ace_atoms:
            continue
        if atoms[nb]["type"].lower() in _GAFF_C_TYPES and atoms[nb]["type"].lower() != "c3":
            o_neighbors = [x for x in adj[nb] if atoms[x]["type"].lower() == "o" and x != backbone_ca_id]
            if o_neighbors:
                backbone_c_id = nb
                backbone_o_id = o_neighbors[0]
                break
    if backbone_c_id is None or backbone_o_id is None:
        raise ValueError(f"Could not identify backbone C in {mol2_path}.")

    nme_cap_n_id = next(
        (
            nb
            for nb in adj[backbone_c_id]
            if nb not in (backbone_ca_id, backbone_o_id) and atoms[nb]["type"].lower() in _GAFF_N_TYPES
        ),
        None,
    )
    nme_atoms: set[int] = set()
    if nme_cap_n_id is not None:
        queue = [nme_cap_n_id]
        while queue:
            node = queue.pop()
            if node in nme_atoms or node == backbone_c_id:
                continue
            nme_atoms.add(node)
            for nb in adj[node]:
                if nb not in nme_atoms and nb != backbone_c_id:
                    queue.append(nb)

    cap_atoms = ace_atoms | nme_atoms
    core_ids = [aid for aid in sorted(atoms) if aid not in cap_atoms]

    backbone_ha_id = next(
        (nb for nb in adj[backbone_ca_id] if atoms[nb]["type"].lower() == "h1" and nb not in cap_atoms),
        None,
    )
    backbone_h_id = next(
        (nb for nb in adj[backbone_n_id] if atoms[nb]["type"].lower() in {"hn", "h"} and nb not in cap_atoms),
        None,
    )
    backbone_cb_id = next(
        (
            nb
            for nb in adj[backbone_ca_id]
            if nb not in (backbone_n_id, backbone_c_id)
            and nb not in cap_atoms
            and atoms[nb]["type"].lower() == "c3"
            and nb != backbone_ha_id
        ),
        None,
    )
    backbone_hb_ids: list[int] = []
    if backbone_cb_id is not None:
        backbone_hb_ids = [
            nb for nb in adj[backbone_cb_id] if atoms[nb]["type"].lower() in {"h1", "hc", "hx"} and nb not in cap_atoms
        ]

    rename: dict[int, tuple[str, str]] = {}
    rename[backbone_n_id] = ("N", _AMBER_BACKBONE_TYPE["backbone_N"])
    if backbone_h_id:
        rename[backbone_h_id] = ("H", _AMBER_BACKBONE_TYPE["backbone_H"])
    rename[backbone_ca_id] = ("CA", _AMBER_BACKBONE_TYPE["backbone_CA"])
    if backbone_ha_id:
        rename[backbone_ha_id] = ("HA", _AMBER_BACKBONE_TYPE["backbone_HA"])
    rename[backbone_c_id] = ("C", _AMBER_BACKBONE_TYPE["backbone_C"])
    rename[backbone_o_id] = ("O", _AMBER_BACKBONE_TYPE["backbone_O"])
    if backbone_cb_id:
        rename[backbone_cb_id] = ("CB", _AMBER_BACKBONE_TYPE["backbone_CB"])
    for i, hb_id in enumerate(backbone_hb_ids, start=2):
        rename[hb_id] = (f"HB{i}", _AMBER_BACKBONE_TYPE["backbone_HB"])

    mol_name = sections.get("MOLECULE", ["UNK"])[0].strip() if sections.get("MOLECULE") else "UNK"
    resname = mol_name.strip()
    pdb_names_by_depth: dict[tuple[str, int], list[str]] = {}
    if protein_pdb is not None:
        with contextlib.suppress(Exception):
            pdb_names_by_depth = _pdb_sidechain_names_by_depth(protein_pdb, resname)

    if pdb_names_by_depth:
        mol2_depth: dict[int, int] = {backbone_ca_id: 0}
        bfs_queue = [backbone_ca_id]
        while bfs_queue:
            node = bfs_queue.pop(0)
            for nb in adj[node]:
                if nb not in mol2_depth and nb not in cap_atoms:
                    mol2_depth[nb] = mol2_depth[node] + 1
                    bfs_queue.append(nb)

        sc_atoms_by_elem_depth: dict[tuple[str, int], list[int]] = {}
        already_named = set(rename)
        for aid in core_ids:
            if aid in already_named:
                continue
            atom = atoms[aid]
            elem = _gaff_type_to_element(atom["type"])
            depth_from_ca = mol2_depth.get(aid, 99)
            sc_atoms_by_elem_depth.setdefault((elem, depth_from_ca), []).append(aid)

        pdb_names_used: set[str] = set()
        for (elem, depth), mol2_aids in sorted(sc_atoms_by_elem_depth.items()):
            pdb_candidates = pdb_names_by_depth.get((elem, depth), [])
            available = [n for n in pdb_candidates if n not in pdb_names_used]
            for mol2_aid, pdb_name in zip(mol2_aids, available):
                rename[mol2_aid] = (pdb_name, atoms[mol2_aid]["type"])
                pdb_names_used.add(pdb_name)

    n_atoms = len(core_ids)
    core_bond_set = [(a1, a2, bt) for a1, a2, bt in bonds if a1 not in cap_atoms and a2 not in cap_atoms]
    n_bonds = len(core_bond_set)
    new_id: dict[int, int] = {old: i + 1 for i, old in enumerate(core_ids)}

    lines = [
        "@<TRIPOS>MOLECULE",
        mol_name,
        f"   {n_atoms}    {n_bonds}     1     0     0",
        "SMALL",
        "RESP Charge",
        "",
        "",
        "@<TRIPOS>ATOM",
    ]
    for old_id in core_ids:
        atom = atoms[old_id]
        nid = new_id[old_id]
        name, atype = rename.get(old_id, (atom["name"], atom["type"]))
        lines.append(
            f"      {nid} {name:<10s} {atom['x']} {atom['y']} {atom['z']} "
            f"{atype:<8s} {atom['subst_id']}  {atom['subst_name']:<8s} {atom['charge']}"
        )

    lines.append("@<TRIPOS>BOND")
    for bid, (a1, a2, bt) in enumerate(core_bond_set, start=1):
        lines.append(f"     {bid}    {new_id[a1]}    {new_id[a2]} {bt}")

    subst_lines = sections.get("SUBSTRUCTURE", [])
    if subst_lines:
        lines.append("@<TRIPOS>SUBSTRUCTURE")
        lines.extend(subst_lines)

    output_mol2.write_text("\n".join(lines) + "\n")
    return output_mol2


# ---------------------------------------------------------------------------
# tleap/antechamber helpers (used when extra_ff has mol2/frcmod files)
# ---------------------------------------------------------------------------


def _write_dry_protein_pdb(protein_pdb: Path, output_pdb: Path) -> Path:
    """Write a PDB for tleap: strip water and H, adapt heavy-atom names to AMBER.

    Uses gemmi to preserve original PDB residue numbers — critical for MCPB.py
    workflows where tleap bond commands reference specific residue IDs.
    Renames heavy atoms that differ between standard PDB/GROMACS convention and
    AMBER ff14SB (OC1→O, OC2→OXT, ILE CD→CD1). H atoms are stripped so tleap
    re-adds them with correct ff14SB names.

    TER placement: gemmi writes TER after the last polymer (ATOM) residue in each
    chain, before any non-polymer (HETATM) residues such as metals.  This is the
    exact behaviour tleap requires to recognise the C-terminal residue and apply
    the CLYS/CXXX template; without TER, saveAmberParm fails on the OXT atom.
    """
    import gemmi  # noqa: PLC0415

    amber_renames: dict[tuple[str, str], str] = {
        ("OC1", ""): "O",
        ("OC2", ""): "OXT",
        ("CD", "ILE"): "CD1",
    }

    st = gemmi.read_pdb_string(protein_pdb.read_text(encoding="utf-8", errors="replace"))

    for model in st:
        for chain in model:
            res_indices_to_remove: list[int] = []
            for ri, residue in enumerate(chain):
                if residue.name.upper() in _WATER_RESIDUE_NAMES_SET:
                    res_indices_to_remove.append(ri)
                    continue
                resname = residue.name.upper()
                atom_indices_to_remove: list[int] = []
                for ai, atom in enumerate(residue):
                    if atom.is_hydrogen():
                        atom_indices_to_remove.append(ai)
                    else:
                        new_name = amber_renames.get((atom.name, resname)) or amber_renames.get((atom.name, ""))
                        if new_name:
                            atom.name = new_name
                for ai in reversed(atom_indices_to_remove):
                    del residue[ai]
            for ri in reversed(res_indices_to_remove):
                del chain[ri]

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(output_pdb))
    return output_pdb


def _compute_net_charge_from_sdf(sdf_path: Path) -> int:
    """Return the total formal charge of the first molecule in an SDF file."""
    from rdkit import Chem  # noqa: PLC0415

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"Could not read any molecule from {sdf_path}")
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def _resolve_executable(name: str) -> str:
    """Return an absolute executable path resolved from the active environment."""
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")
    return executable


def _run_antechamber(sdf_path: Path, work_dir: Path, net_charge: int | None) -> tuple[Path, Path]:
    """Run antechamber + parmchk2 on a ligand SDF; return (mol2, frcmod) paths."""
    if net_charge is None:
        net_charge = _compute_net_charge_from_sdf(sdf_path)

    mol2_out = work_dir / "antechamber.mol2"
    frcmod_out = work_dir / "antechamber.frcmod"

    ante_result = subprocess.run(  # noqa: S603
        [
            _resolve_executable("antechamber"),
            "-i",
            str(sdf_path.resolve()),
            "-fi",
            "sdf",
            "-o",
            str(mol2_out),
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-s",
            "2",
            "-nc",
            str(net_charge),
            "-at",
            "gaff2",
        ],
        capture_output=True,
        cwd=str(work_dir),
        text=True,
        check=False,
    )
    if ante_result.returncode != 0 or not mol2_out.exists():
        raise RuntimeError(f"antechamber failed for {sdf_path.name}:\n{ante_result.stderr}\n{ante_result.stdout}")

    parmchk_result = subprocess.run(  # noqa: S603
        [
            _resolve_executable("parmchk2"),
            "-s",
            "2",
            "-i",
            str(mol2_out),
            "-f",
            "mol2",
            "-o",
            str(frcmod_out),
        ],
        capture_output=True,
        cwd=str(work_dir),
        text=True,
        check=False,
    )
    if parmchk_result.returncode != 0 or not frcmod_out.exists():
        raise RuntimeError(f"parmchk2 failed for {sdf_path.name}:\n{parmchk_result.stderr}")

    return mol2_out, frcmod_out


_PROTEIN_FF_LEAPRC: dict[ProteinFF, str] = {
    ProteinFF.FF14SB: "leaprc.protein.ff14SB",
    ProteinFF.FF19SB: "leaprc.protein.ff19SB",
    ProteinFF.FF99SB: "leaprc.protein.ff99SBildn",
}

_LIGAND_FF_LEAPRC: dict[LigandFF, str] = {
    LigandFF.GAFF: "leaprc.gaff",
    LigandFF.GAFF2: "leaprc.gaff2",
}


def _write_tleap_script(
    protein_pdb: Path,
    protein_mol2s: list[Path],
    protein_frcmods: list[Path],
    ligand_mol2: Path,
    ligand_frcmod: Path,
    cofactor_mol2s: list[Path],
    cofactor_frcmods: list[Path],
    output_prefix: Path,
    add_atom_types_block: str | None = None,
    bond_commands: list[str] | None = None,
    extra_frcmod_names: list[str] | None = None,
    leaprc_extra_sources: tuple[str, ...] | list[str] | None = None,
    protein_ff: ProteinFF = ProteinFF.FF14SB,
    ligand_ff: LigandFF = LigandFF.GAFF2,
) -> Path:
    """Write a tleap script that combines protein + ligand + cofactors.

    ``add_atom_types_block``, ``bond_commands``, and ``extra_frcmod_names`` are
    populated from an MCPB.py-generated tleap.in when a metal site is present.
    They are otherwise empty and the script degenerates to the standard path.
    The addAtomTypes block must appear before mol2 templates are loaded so that
    custom atom-type symbols (M1, Y1-Yn for any metal / any geometry) are
    recognised when the mol2 files are parsed.  Bond commands are appended after
    combine so that residue numbers still refer to the original PDB numbering.
    ``leaprc_extra_sources`` adds additional leaprc sources (e.g.
    leaprc.phosaa14SB for PTR/SEP/TPO) after the main protein and ligand FFs.
    """
    bond_commands = bond_commands or []
    extra_frcmod_names = extra_frcmod_names or []
    leaprc_extra_sources = leaprc_extra_sources or []

    lines = [
        f"source {_PROTEIN_FF_LEAPRC[protein_ff]}",
        f"source {_LIGAND_FF_LEAPRC[ligand_ff]}",
    ]

    # Additional leaprc files (e.g. phosphorylated amino acids, custom residues).
    for leaprc in leaprc_extra_sources:
        lines.append(f"source {leaprc}")

    # MCPB.py custom atom types (must come before loadMol2 calls).
    if add_atom_types_block:
        lines.append(add_atom_types_block)

    # Standard AMBER frcmods (e.g. frcmod.ions1lm_126_tip3p) from MCPB.py script.
    for fname in extra_frcmod_names:
        lines.append(f"loadAmberParams {fname}")

    # Non-standard protein residue parameters (frcmod then mol2 templates).
    for frcmod in protein_frcmods:
        lines.append(f"loadAmberParams {frcmod.resolve()}")
    for mol2 in protein_mol2s:
        resname = mol2.stem.replace("_stripped", "").upper()
        lines.append(f"{resname} = loadMol2 {mol2.resolve()}")

    # Ligand parameters and coordinates (antechamber mol2 carries docked pose).
    lines.append(f"loadAmberParams {ligand_frcmod.resolve()}")
    lines.append(f"mol_lig = loadMol2 {ligand_mol2.resolve()}")

    # Cofactor parameters.
    for i, (cof_mol2, cof_frcmod) in enumerate(zip(cofactor_mol2s, cofactor_frcmods)):
        lines.append(f"loadAmberParams {cof_frcmod.resolve()}")
        lines.append(f"mol_cof{i} = loadMol2 {cof_mol2.resolve()}")

    # Load protein directly from PDB (handles HETATM non-standard residues and
    # MCPB.py-renamed residues like CS1-4 / ZN1).
    lines.append(f"mol_prot = loadPdb {protein_pdb.resolve()}")

    # Metal-coordination and backbone-reconnection bond commands from MCPB.py.
    # Must be applied to mol_prot BEFORE combine: tleap preserves PDB residue
    # numbers on a loadPdb unit but renumbers sequentially after combine, so
    # bond commands like "bond mol.400.ZN" fail post-combine when the protein
    # has fewer than 400 sequential residues.
    for bond_cmd in bond_commands:
        lines.append(bond_cmd.replace("mol.", "mol_prot."))

    # Combine protein + ligand + cofactors.
    parts = "mol_prot mol_lig"
    for i in range(len(cofactor_mol2s)):
        parts += f" mol_cof{i}"
    lines.append(f"mol = combine {{{parts}}}")

    lines.extend(
        [
            f"saveAmberParm mol {output_prefix}.prmtop {output_prefix}.inpcrd",
            "quit",
        ]
    )

    script_path = output_prefix.parent / "tleap.in"
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def _run_tleap(script_path: Path, work_dir: Path) -> None:
    """Run tleap and raise RuntimeError on failure."""
    result = subprocess.run(  # noqa: S603
        [_resolve_executable("tleap"), "-f", str(script_path)],
        capture_output=True,
        cwd=str(work_dir),
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"tleap failed (exit {result.returncode}):\n{output}")
    if "FATAL" in output.upper():
        raise RuntimeError(f"tleap reported FATAL error:\n{output}")
    logger.debug("tleap completed successfully.")
    if "Warning" in output or "error" in output.lower():
        logger.debug("tleap warnings/non-fatal messages:\n%s", output)


def _parse_mcpb_tleap_in(tleap_in: Path) -> dict:
    """Parse an MCPB.py-generated tleap.in and extract elements for our combined script.

    MCPB.py step 4 generates a complete tleap script for any metal / any geometry.
    This parser extracts the portable parts (addAtomTypes block, bond commands,
    protein PDB path, and standard ion frcmod names) without hardcoding residue
    names, atom types, coordination numbers, or metal identity.  Works for
    3-8-coordinate sites with any transition metal.
    """
    tleap_dir = tleap_in.parent
    add_atom_types_lines: list[str] = []
    bond_commands: list[str] = []
    protein_pdb: Path | None = None
    extra_frcmod_names: list[str] = []

    in_add_atom_types = False

    for line in tleap_in.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        low = stripped.lower()

        if low.startswith("addatomtypes"):
            in_add_atom_types = True
            add_atom_types_lines.append(stripped)
            continue

        if in_add_atom_types:
            add_atom_types_lines.append(stripped)
            if "}" in stripped and "{" not in stripped:
                in_add_atom_types = False
            continue

        # mol = loadPdb X  (the MCPB.py-renamed protein PDB, no ligand)
        if re.match(r"mol\s*=\s*loadpdb\s+", low):
            pdb_name = re.split(r"loadpdb\s+", stripped, flags=re.IGNORECASE, maxsplit=1)[-1].strip()
            candidate = (tleap_dir / pdb_name).resolve()
            if candidate.exists():
                protein_pdb = candidate
            continue

        # bond commands — generic for any coordination geometry
        if low.startswith("bond "):
            bond_commands.append(stripped)
            continue

        # Standard AMBER frcmod files (no leading "/" = resolved from $AMBERHOME)
        if low.startswith("loadamberparams "):
            frcmod_name = stripped.split(None, 1)[1].strip()
            if not frcmod_name.startswith("/") and not (tleap_dir / frcmod_name).exists():
                extra_frcmod_names.append(frcmod_name)
            continue

    return {
        "add_atom_types_block": "\n".join(add_atom_types_lines) if add_atom_types_lines else None,
        "bond_commands": bond_commands,
        "protein_pdb": protein_pdb,
        "extra_frcmod_names": extra_frcmod_names,
    }


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
    mcpb_tleap_in: Path | None = None
    leaprc_extra_sources: tuple[str, ...] = ()

    @field_validator("extra_ff_files", mode="before")
    @classmethod
    def _check_extra_ff_files(cls, paths: Any) -> tuple[Path, ...]:
        result = tuple(Path(p) for p in paths)
        missing = [p for p in result if not p.exists()]
        if missing:
            raise ValueError("Extra force field files not found: " + ", ".join(str(p) for p in missing))
        return result

    @field_validator("mcpb_tleap_in", mode="before")
    @classmethod
    def _check_mcpb_tleap_in(cls, v: Any) -> Path | None:
        if v is None:
            return None
        p = Path(v)
        if not p.exists():
            raise ValueError(f"MCPB.py tleap.in not found: {p}")
        return p

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

    protein_pdb: Path
    ligand_sdf: Path
    cofactor_sdfs: tuple[Path, ...] = ()
    config: ParametrizationConfig = Field(default_factory=ParametrizationConfig)
    net_charge: int | None = None
    work_dir: Path | None = None

    @field_validator("protein_pdb", "ligand_sdf")
    @classmethod
    def _check_exists(cls, path: Path) -> Path:
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        return path

    @field_validator("cofactor_sdfs", mode="before")
    @classmethod
    def _check_cofactor_sdfs(cls, paths: Any) -> tuple[Path, ...]:
        result = tuple(Path(p) for p in paths)
        missing = [p for p in result if not p.exists()]
        if missing:
            raise ValueError("Cofactor SDF files not found: " + ", ".join(str(p) for p in missing))
        return result


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
# tleap / antechamber implementation
# ---------------------------------------------------------------------------


def _remap_bond_residue_numbers(bond_commands: list[str], hetatm_resnum_map: dict[int, int]) -> list[str]:
    """Remap PDB HETATM residue numbers in bond commands to tleap sequential positions.

    Modern tleap (teLeap) assigns HETATM residues that follow a gap in the PDB
    residue numbering a sequential position immediately after the last ATOM
    residue, not the actual PDB residue number.  E.g. a ZN at PDB residue 400
    in a protein whose last ATOM residue is 191 gets tleap index 192.

    hetatm_resnum_map: {pdb_resnum: tleap_seq_num} for all HETATM residues.
    """
    if not hetatm_resnum_map:
        return bond_commands

    def _repl(m: re.Match) -> str:
        n = int(m.group(1))
        return f".{hetatm_resnum_map.get(n, n)}."

    return [re.sub(r"\.(\d+)\.", _repl, cmd) for cmd in bond_commands]


def _parametrize_tleap(inp: ParametrizationInput) -> ParametrisedComplex:
    """Parametrize using tleap for the protein and antechamber for the ligand.

    Proteins with non-standard residues supplied as mol2+frcmod files (e.g.
    MCPB.py metal sites, RESP-charge non-standard amino acids) cannot be
    reliably handled by OpenMM template matching when those residues appear as
    HETATM records in the PDB. tleap reads the original PDB directly and
    resolves residue templates by name, so HETATM non-standard residues are
    parametrized correctly as long as a matching mol2 template is loaded first.
    """
    work_dir = inp.work_dir or Path(tempfile.mkdtemp(prefix="gbsa_param_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    all_frcmod_files = [p for p in inp.config.extra_ff_files if p.suffix.lower() == ".frcmod"]
    all_mol2_files = [p for p in inp.config.extra_ff_files if p.suffix.lower() == ".mol2"]
    # XML files (e.g. phosaa14SB.xml) are OpenMM-only; tleap uses leaprc instead.

    # When a cofactor SDF has a same-stem mol2+frcmod in extra_ff_files, those
    # pre-computed parameters are used directly (antechamber is skipped for that
    # cofactor). Files not matching any cofactor stem are treated as protein residue
    # templates and passed to _write_tleap_script as protein_mol2s/protein_frcmods.
    _cof_stems = {cof_sdf.stem for cof_sdf in inp.cofactor_sdfs}
    precomp_mol2 = {p.stem: p for p in all_mol2_files if p.stem in _cof_stems}
    precomp_frcmod = {p.stem: p for p in all_frcmod_files if p.stem in _cof_stems}
    frcmod_files = [p for p in all_frcmod_files if p.stem not in _cof_stems]
    mol2_files = [p for p in all_mol2_files if p.stem not in _cof_stems]

    # MCPB.py path: use the MCPB.py-renamed PDB (with metal residues and
    # metal ion already in place) instead of the original protein PDB.
    # The tleap.in provides the addAtomTypes block (generic for any metal /
    # any geometry) and all bond commands (coordination + backbone reconnection).
    if inp.config.mcpb_tleap_in is not None:
        mcpb_info = _parse_mcpb_tleap_in(inp.config.mcpb_tleap_in)
        source_pdb = mcpb_info["protein_pdb"]
        if source_pdb is None:
            raise RuntimeError(f"Could not locate loadpdb line in MCPB.py tleap.in: {inp.config.mcpb_tleap_in}")
        add_atom_types_block: str | None = mcpb_info["add_atom_types_block"]
        bond_commands: list[str] = mcpb_info["bond_commands"]
        extra_frcmod_names: list[str] = mcpb_info["extra_frcmod_names"]
    else:
        source_pdb = inp.protein_pdb
        add_atom_types_block = None
        bond_commands = []
        extra_frcmod_names = []

    # Crystal waters: save for solvate_bss to restore, strip from protein PDB.
    crystal_waters_pdb = _write_crystal_waters_pdb(source_pdb, work_dir / "crystal_waters.pdb")
    dry_pdb = _write_dry_protein_pdb(source_pdb, work_dir / "protein_dry.pdb")

    # Modern tleap (teLeap) assigns HETATM residues that appear after a gap in
    # the PDB residue numbering a sequential position immediately after the last
    # ATOM residue (e.g. ZN at PDB 400 → tleap index 192 when last ATOM is 191).
    # Remap any such residue numbers in the MCPB.py bond commands.
    if bond_commands:
        _atom_resnums: set[int] = set()
        _hetatm_resnums: set[int] = set()
        for _line in dry_pdb.read_text().splitlines():
            _rec = _line[:6].strip().upper()
            try:
                _rn = int(_line[22:26])
            except (ValueError, IndexError):
                continue
            if _rec == "ATOM":
                _atom_resnums.add(_rn)
            elif _rec == "HETATM":
                _hetatm_resnums.add(_rn)
        _hetatm_map: dict[int, int] = {}
        if _atom_resnums and _hetatm_resnums:
            _max_atom = max(_atom_resnums)
            for _i, _pdb_rn in enumerate(sorted(_hetatm_resnums)):
                _hetatm_map[_pdb_rn] = _max_atom + 1 + _i
        bond_commands = _remap_bond_residue_numbers(bond_commands, _hetatm_map)

    # Strip ACE/NME caps from mol2 files prepared as capped dipeptides.
    # MCPB.py mol2s (CS1-4, ZN1) are already stripped residue templates;
    # _strip_mol2_dipeptide_caps raises ValueError for them, so we fall back
    # to the original mol2 unchanged — which is correct for tleap.
    stripped_mol2s: list[Path] = []
    for mol2 in mol2_files:
        stripped = work_dir / f"{mol2.stem}_stripped.mol2"
        try:
            _strip_mol2_dipeptide_caps(mol2, stripped, protein_pdb=source_pdb)
            stripped_mol2s.append(stripped)
        # Mol2 templates come from heterogeneous chemistry tools; this fallback
        # intentionally keeps already-stripped templates unchanged on any parser failure.
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Cap stripping skipped for {mol2.name}: {exc}", stacklevel=2)
            stripped_mol2s.append(mol2)

    # Parametrize ligand with antechamber + parmchk2 (GAFF2 + AM1-BCC).
    lig_work = work_dir / "ligand"
    lig_work.mkdir(exist_ok=True)
    lig_mol2, lig_frcmod = _run_antechamber(inp.ligand_sdf, lig_work, inp.net_charge)

    # Parametrize cofactors (use pre-computed mol2+frcmod from extra_ff_files when
    # both files are present for a given cofactor stem; fall back to antechamber).
    cof_mol2s: list[Path] = []
    cof_frcmods: list[Path] = []
    for i, cof_sdf in enumerate(inp.cofactor_sdfs):
        pm2 = precomp_mol2.get(cof_sdf.stem)
        pfc = precomp_frcmod.get(cof_sdf.stem)
        if pm2 is not None and pfc is not None:
            cof_mol2s.append(pm2)
            cof_frcmods.append(pfc)
        else:
            cof_work = work_dir / f"cofactor_{i}"
            cof_work.mkdir(exist_ok=True)
            c_mol2, c_frcmod = _run_antechamber(cof_sdf, cof_work, net_charge=None)
            cof_mol2s.append(c_mol2)
            cof_frcmods.append(c_frcmod)

    # Write and run tleap to get combined AMBER topology + coordinates.
    script = _write_tleap_script(
        protein_pdb=dry_pdb,
        protein_mol2s=stripped_mol2s,
        protein_frcmods=frcmod_files,
        ligand_mol2=lig_mol2,
        ligand_frcmod=lig_frcmod,
        cofactor_mol2s=cof_mol2s,
        cofactor_frcmods=cof_frcmods,
        output_prefix=work_dir / "complex",
        add_atom_types_block=add_atom_types_block,
        bond_commands=bond_commands,
        extra_frcmod_names=extra_frcmod_names,
        leaprc_extra_sources=list(inp.config.leaprc_extra_sources),
        protein_ff=inp.config.protein_ff,
        ligand_ff=inp.config.ligand_ff,
    )
    _run_tleap(script, work_dir=work_dir)

    # Convert AMBER prmtop/inpcrd → GROMACS GRO/TOP with ParmEd.
    prmtop = work_dir / "complex.prmtop"
    inpcrd = work_dir / "complex.inpcrd"
    if not prmtop.exists() or not inpcrd.exists():
        raise RuntimeError(
            f"tleap did not produce expected output files in {work_dir}. Check tleap.in and the tleap output."
        )
    struct = pmd.load_file(str(prmtop), str(inpcrd))
    gro_file = work_dir / "complex.gro"
    top_file = work_dir / "complex.top"
    gro_file.unlink(missing_ok=True)
    top_file.unlink(missing_ok=True)
    struct.save(str(top_file), format="gromacs")
    struct.save(str(gro_file))

    return ParametrisedComplex(
        gro_file=gro_file,
        top_file=top_file,
        config=inp.config,
        forcefield=None,
        parmed_structure=struct,
        crystal_waters_pdb=crystal_waters_pdb,
    )


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


def _assign_nagl_charges_direct(mol: Molecule) -> None:
    """Assign AM1-BCC charges via the openff-nagl GNNModel API, bypassing the toolkit wrapper.

    The OpenFF toolkit's NAGL toolkit wrapper may not register in all environments
    (e.g. SLURM compute nodes). This function loads the model directly and writes
    charges back to the molecule, normalised to the molecular formal charge sum.
    """
    import numpy as np  # noqa: PLC0415
    import openff.nagl as _nagl  # noqa: PLC0415
    import openff.nagl_models as _nm  # noqa: PLC0415
    from openff.units import unit as _unit  # noqa: PLC0415

    model_path = str(_nm.get_model("openff-gnn-am1bcc-1.0.0.pt"))
    model = _nagl.GNNModel.load(model_path, eval_mode=True)
    charges_dict = model.compute_properties(mol)
    charges = charges_dict["am1bcc_charges"].astype(float)
    formal_sum = float(sum(a.formal_charge.m for a in mol.atoms))
    charges -= (float(np.sum(charges)) - formal_sum) / len(charges)
    mol.partial_charges = _unit.Quantity(charges, _unit.elementary_charge)


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

    protein_modeller = _load_pdb_as_modeller(inp.protein_pdb)
    n_removed = _delete_residues_by_name(protein_modeller, _WATER_RESIDUE_NAMES)
    if n_removed:
        logger.debug("Removed %d crystallographic water residue(s) from protein topology.", n_removed)
    logger.debug("Protein PDB loaded without crystal waters (%d atoms).", protein_modeller.topology.getNumAtoms())

    # --- Ligand --------------------------------------------------------
    logger.debug("Loading ligand SDF: %s …", inp.ligand_sdf)
    ligand = Molecule.from_file(str(inp.ligand_sdf), allow_undefined_stereo=True)
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

    # --- Cofactors -----------------------------------------------------
    cofactors: list[Molecule] = []
    for cof_path in inp.cofactor_sdfs:
        logger.debug("Loading cofactor SDF: %s …", cof_path)
        cof = Molecule.from_file(str(cof_path), allow_undefined_stereo=True)
        if not cof.conformers:
            raise ValueError(
                f"Cofactor SDF '{cof_path}' contains no 3-D conformers. "
                "Provide an SDF file with embedded 3-D coordinates."
            )
        logger.debug(
            "Cofactor loaded (%d atoms, %d conformers).",
            cof.n_atoms,
            len(cof.conformers),
        )
        logger.debug("Assigning partial charges to cofactor (method=%s) …", inp.config.charge_method.value)
        try:
            cof.assign_partial_charges(
                partial_charge_method=inp.config.charge_method.value,
                normalize_partial_charges=True,
                use_conformers=cof.conformers,
            )
        # OpenFF charge assignment can fail with toolkit/backend-specific exceptions;
        # this fallback deliberately catches the broad failure and tries NAGL directly.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cofactor charge assignment failed with %s (%s); falling back to NAGL direct API.",
                inp.config.charge_method.value,
                exc,
            )
            _assign_nagl_charges_direct(cof)
            logger.info("Cofactor charges assigned via NAGL fallback.")
        cofactors.append(cof)
    if cofactors:
        logger.debug("Loaded and parametrized %d cofactor(s).", len(cofactors))

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
        molecules=[ligand, *cofactors],
        forcefield=_GAFF_FF_VERSION[inp.config.ligand_ff],
        cache=None,
    )
    forcefield.registerTemplateGenerator(gaff.generator)
    logger.debug("GAFF template generator registered.")

    logger.debug("Combining protein+ligand topology …")
    modeller = Modeller(protein_modeller.topology, protein_modeller.positions)
    modeller.add(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())
    for cof in cofactors:
        modeller.add(cof.to_topology().to_openmm(), cof.conformers[0].to_openmm())
    logger.debug("Combined dry topology: %d atoms.", modeller.topology.getNumAtoms())

    logger.debug("Creating OpenMM system (may take a moment) …")
    system: System = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        # VERY IMPORTANT THIS INFORMATION DOES NOT GET LOST: Constraints None is safe, HBOND fails as the generated top file does not fit with the parameters
    )
    logger.debug("OpenMM system created (%d particles).", system.getNumParticles())

    logger.debug("Converting OpenMM system to ParmEd structure …")
    structure = pmd.openmm.load_topology(modeller.topology, system, modeller.positions)
    logger.debug("ParmEd structure ready (%d atoms).", len(structure.atoms))

    gro_file = work_dir / "complex.gro"
    top_file = work_dir / "complex.top"
    gro_file.unlink(missing_ok=True)
    top_file.unlink(missing_ok=True)
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
