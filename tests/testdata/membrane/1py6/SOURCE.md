# Source

Downloaded from [MemProtMD](https://memprotmd.bioch.ox.ac.uk/) on 2026-09-11:

```
https://memprotmd.bioch.ox.ac.uk/data/memprotmd/simulations/1py6_default_dppc/files/run/at.zip
```

PDB entry [1PY6](https://www.rcsb.org/structure/1PY6) (bacteriorhodopsin),
MemProtMD's atomistic (CG2AT-converted) system: the protein embedded in a
DPPC bilayer, solvated, with ions — the end state of MemProtMD's coarse-grained
self-assembly simulation, converted back to atomistic detail. GROMOS53a6
protein force field, SPC water, DPPC parametrized via
`itp/lipid-gmx53a6.itp` (from LipidBook).

Trimmed from the original ~13 MB archive to the files this pipeline's
`gbsa_pipeline.membrane` module and tests actually use:

- `atomistic-system.pdb` — structure (protein + 209 DPPC + water + ions)
- `atomistic-system.ndx` — index groups
- `topol.top` — topology (`#include`s `gromos53a6.ff/...`, which ships with
  any standard GROMACS install, and the two `itp/` files below)
- `posre.itp` — position restraints
- `itp/gromos-lipids.itp`, `itp/lipid-gmx53a6.itp` — DPPC parameters

Dropped: `charmm36.ff/` (an alternative force field not referenced by this
`topol.top`), `itp/charmm36-lipids.itp` (same reason), `mdp_files/` (this
pipeline generates its own MDP via `GromacsParams`), `readme-at.txt`.

If you use MemProtMD data, cite:
Newport TD, Sansom MSP, Stansfeld PJ. *The MemProtMD database: a resource for
membrane-embedded protein structures and their lipid interactions.*
Nucleic Acids Research, 2019. https://doi.org/10.1093/nar/gky1047
