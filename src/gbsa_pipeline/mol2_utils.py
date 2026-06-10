"""Mol2 cap-stripping helpers for capped-dipeptide RESP parametrization."""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import TYPE_CHECKING

import gemmi

if TYPE_CHECKING:
    from pathlib import Path
import parmed as pmd

logger = logging.getLogger(__name__)

THREE_CHAR_ATOM_NAME_LENGTH = 3
MIN_GREEK_ATOM_NAME_LENGTH = 2
MIN_PDBQT_ATOM_FIELDS = 9
MIN_PDBQT_BOND_FIELDS = 4

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


def _pdb_sidechain_names_by_depth(protein_pdb: Path, resname: str) -> dict[tuple[str, int], list[str]]:
    """Return a mapping (element, depth_from_CA) → [atom_names] for a residue in the PDB."""
    st = gemmi.read_pdb(str(protein_pdb))
    pdb_atoms: list[tuple[str, str]] = []
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.name.upper() != resname:
                    continue
                for atom in residue:
                    pdb_atoms.append((atom.name, atom.element.name))

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
    structure = pmd.load_file(str(mol2_path), structure=True)

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
            nb
            for nb in adj_pmd[backbone_ca.idx]
            if nb is not backbone_n
            and nb is not backbone_c
            and nb.idx not in cap_idx
            and nb.type.lower() == "c3"
            and nb is not backbone_ha
        ),
        None,
    )
    backbone_hb_atoms: list[pmd.Atom] = []
    if backbone_cb is not None:
        backbone_hb_atoms = [
            nb for nb in adj_pmd[backbone_cb.idx] if nb.type.lower() in {"h1", "hc", "hx"} and nb.idx not in cap_idx
        ]

    backbone_n.name = "N"
    backbone_n.type = _AMBER_BACKBONE_TYPE["backbone_N"]

    backbone_ca.name = "CA"
    backbone_ca.type = _AMBER_BACKBONE_TYPE["backbone_CA"]

    backbone_c.name = "C"
    backbone_c.type = _AMBER_BACKBONE_TYPE["backbone_C"]

    backbone_o.name = "O"
    backbone_o.type = _AMBER_BACKBONE_TYPE["backbone_O"]

    if backbone_ha:
        backbone_ha.name = "HA"
        backbone_ha.type = _AMBER_BACKBONE_TYPE["backbone_HA"]

    if backbone_h:
        backbone_h.name = "H"
        backbone_h.type = _AMBER_BACKBONE_TYPE["backbone_H"]

    if backbone_cb:
        backbone_cb.name = "CB"
        backbone_cb.type = _AMBER_BACKBONE_TYPE["backbone_CB"]

    for i, hb in enumerate(backbone_hb_atoms, start=2):
        hb.name = f"HB{i}"
        hb.type = _AMBER_BACKBONE_TYPE["backbone_HB"]

    resname = structure.residues[0].name.strip() if structure.residues else "UNK"
    pdb_names_by_depth: dict[tuple[str, int], list[str]] = {}
    if protein_pdb is not None:
        with contextlib.suppress(Exception):
            pdb_names_by_depth = _pdb_sidechain_names_by_depth(protein_pdb, resname)

    if pdb_names_by_depth:
        named_idx = {
            a.idx
            for a in [
                backbone_n,
                backbone_ca,
                backbone_c,
                backbone_o,
                backbone_ha,
                backbone_h,
                backbone_cb,
                *backbone_hb_atoms,
            ]
            if a is not None
        }
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

    structure.strip(":ACE,NME")  # noqa: B005
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
    except (OSError, ValueError, RuntimeError, AttributeError) as exc:
        logger.warning(
            "ParmEd cap stripping failed for %s; falling back to text-based stripping: %s",
            mol2_path,
            exc,
        )
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


def _strip_mol2_or_original(
    mol2: Path,
    work_dir: Path,
    protein_pdb: Path | None = None,
) -> Path:
    """Strip ACE/NME caps from mol2, falling back to the original path on any failure.

    Returns ``work_dir/<stem>_stripped.mol2`` on success, or the original ``mol2``
    path unchanged when stripping fails (e.g. the file is already a bare residue
    template produced by MCPB.py). A :class:`UserWarning` is emitted in the
    fallback case so callers are informed without raising.
    """
    stripped = work_dir / f"{mol2.stem}_stripped.mol2"
    try:
        _strip_mol2_dipeptide_caps(mol2, stripped, protein_pdb=protein_pdb)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Cap stripping skipped for {mol2.name}: {exc}", stacklevel=2)
        return mol2
    else:
        return stripped
