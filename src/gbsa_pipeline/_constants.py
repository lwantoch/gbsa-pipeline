"""Package-wide constants shared across modules."""

from __future__ import annotations

WATER_RESIDUE_NAMES: frozenset[str] = frozenset({"HOH", "WAT", "TIP3", "TIP3P", "SOL"})

# Bulk monovalent counter-ion names across common naming conventions (GROMACS,
# CHARMM-GUI, AMBER). Deliberately excludes divalent metals (Mg2+, Ca2+,
# Zn2+, ...): those are commonly structural/catalytic, not bulk solvent ions,
# and gmx_MMPBSA's own topology cleaning (GMXMMPBSA.make_top.cleantop) does
# not strip them either.
ION_RESIDUE_NAMES: frozenset[str] = frozenset({"NA", "CL", "SOD", "Na+", "CLA", "Cl-", "POT", "K+"})
