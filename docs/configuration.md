---
title: Configuration Reference
---

# Configuration Reference

All pipeline settings are declared in a single TOML file. Every section is
optional except exactly one of `[system]` or `[membrane_system]`.

GROMACS MDP parameter names use **hyphens** in `.mdp` files but **underscores**
in the TOML config (e.g., `ref-t` → `ref_t`).

---

## `[system]`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `protein` | path | **yes** | Path to the protein PDB file |
| `ligand` | path | no | Path to the ligand SDF file (3-D conformer required) |
| `extra_ff_files` | list[path] | no | Extra OpenMM ForceField XML files (e.g., metal parameters) |
| `net_charge` | int | no | Formal charge of the ligand (auto-detected when omitted) |

---

## `[membrane_system]`

Alternative to `[system]` for a protein already embedded in a lipid bilayer
and solvated — e.g. the output of
[`gbsa_pipeline.membrane.fetch_memprotmd_system`][gbsa_pipeline.membrane.fetch_memprotmd_system].
`[forcefield]` is ignored (nothing is parametrized) and stages 1-2
(parametrize, solvate) are skipped entirely — the pipeline loads
`structure`/`topology` directly and starts at SD minimization. See
[Membrane protein example](#membrane-protein-example) below.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `structure` | path | **yes** | Structure file (`.pdb` or `.gro`) of the complete protein-in-bilayer-in-water system |
| `topology` | path | **yes** | GROMACS `.top` topology for `structure`. Its `#include`d force-field/lipid `.itp` files must stay alongside it on disk |

---

## `[forcefield]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `protein_ff` | str | `"ff14SB"` | Protein force field. Options: `"ff14SB"`, `"ff19SB"`, `"ff99SB"` |
| `ligand_ff` | str | `"gaff2"` | Ligand force field. Options: `"gaff"`, `"gaff2"` |
| `charge_method` | str | `"am1bcc"` | Partial charge method. Options: `"am1bcc"`, `"nagl"`, `"espaloma-am1bcc"` |

---

## `[solvation]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `water_model` | str | `"tip3p"` | Water model. Options: `"tip3p"`, `"tip4p"`, `"tip5p"`, `"spc"`, `"spce"` |
| `box_shape` | str | `"truncated_octahedron"` | Box shape. Options: `"truncated_octahedron"`, `"cubic"` |
| `padding` | float | `null` | Distance (nm) from solute to box edge. Takes precedence over `box_size` |
| `box_size` | float | `8.0` | Absolute box edge length (nm). Used when `padding` is not set |
| `ion_concentration` | float | `0.15` | Salt concentration (mol/L) |
| `neutralize` | bool | `true` | Add counter-ions to neutralize the system |

Either `padding` or `box_size` must be provided.

---

## `[minimization]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nsteps` | int | `10000` | Maximum number of minimization steps |
| `emtol` | float | `10.0` | Convergence criterion: max force (kJ mol⁻¹ nm⁻¹) |
| `define` | str | `null` | GROMACS preprocessor define for the SD minimization stage, e.g. `"-DFLEX_SPC"`. For systems whose starting coordinates aren't precise enough for rigid SETTLE-constrained water (e.g. externally-built membrane systems) — minimizing with flexible water for this first pass avoids the resulting NaN potential energy at step 0. Leave unset for the ordinary protein-ligand path |

---

## `[equilibration]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `simulation_time_ps` | float | `500.0` | NVT heating duration (ps), ramping from 0 K to 300 K |

---

## `[md]` — Production MD (GROMACS MDP parameters)

The `[md]` section accepts any field of
[`GromacsParams`][gbsa_pipeline.change_defaults.GromacsParams].
Field names use underscores; they map to hyphenated GROMACS MDP keys.

### Integrator

| TOML field | MDP key | Type | Default | Description |
|------------|---------|------|---------|-------------|
| `integrator` | `integrator` | str | `"md"` | Integration algorithm: `"md"` (leapfrog), `"md-vv"`, `"sd"`, `"bd"` |
| `dt` | `dt` | float | `0.001` | Time step (ps) |
| `nsteps` | `nsteps` | int | `500` | Number of MD steps |
| `tinit` | `tinit` | float | `0.0` | Start time (ps) |

### Output control

| TOML field | MDP key | Type | Default | Description |
|------------|---------|------|---------|-------------|
| `nstlog` | `nstlog` | int | `500` | Steps between log entries |
| `nstenergy` | `nstenergy` | int | `500` | Steps between energy writes |
| `nstxout_compressed` | `nstxout-compressed` | int | `500` | Steps between compressed trajectory frames |

### Thermostat

| TOML field | MDP key | Type | Default | Description |
|------------|---------|------|---------|-------------|
| `tcoupl` | `tcoupl` | str | `"no"` | Thermostat: `"no"`, `"berendsen"`, `"nose-hoover"`, `"v-rescale"`, `"andersen"` |
| `tc_grps` | `tc-grps` | str | `"System"` | Temperature coupling group(s) |
| `tau_t` | `tau-t` | float | `0.1` | Temperature coupling time constant (ps) |
| `ref_t` | `ref-t` | float | `300.0` | Reference temperature (K) |
| `nhchainlength` | `nhchainlength` | int | `10` | Nosé-Hoover chain length |

### Barostat

| TOML field | MDP key | Type | Default | Description |
|------------|---------|------|---------|-------------|
| `pcoupl` | `pcoupl` | str | `"no"` | Barostat: `"no"`, `"Berendsen"`, `"Parrinello-Rahman"`, `"C-rescale"`, `"MTTK"` |
| `pcoupltype` | `pcoupltype` | str | `"isotropic"` | Coupling geometry: `"isotropic"`, `"semiisotropic"`, `"anisotropic"`, `"surface-tension"` |
| `tau_p` | `tau-p` | float | `2.0` | Pressure coupling time constant (ps) |
| `ref_p` | `ref-p` | float or [float, float] | `1.0` | Reference pressure (bar). A 2-value list is required for `semiisotropic`/`anisotropic` (membrane-plane, bilayer-normal) |
| `compressibility` | `compressibility` | float or [float, float] | `4.5e-5` | Isothermal compressibility (bar⁻¹); same 2-value rule as `ref_p` |

`ref_p`/`compressibility` are also read (with the same stability overrides
applied to `dt`/LINCS) by the two NPT equilibration stages, not only
production — so a `semiisotropic` `[md]` barostat applies consistently
through equilibration too.

### Electrostatics & VdW

| TOML field | MDP key | Type | Default | Description |
|------------|---------|------|---------|-------------|
| `coulombtype` | `coulombtype` | str | `"PME"` | Electrostatics method |
| `rcoulomb` | `rcoulomb` | float | `1.2` | Coulomb cutoff (nm) |
| `vdw_type` | `vdw-type` | str | `"Cut-off"` | VdW interaction type |
| `rvdw` | `rvdw` | float | `1.2` | VdW cutoff (nm) |
| `rvdw_switch` | `rvdw-switch` | float | `1.0` | VdW switch start (nm) |

### Constraints

| TOML field | MDP key | Type | Default | Description |
|------------|---------|------|---------|-------------|
| `constraints` | `constraints` | str | `"none"` | Constraint type: `"none"`, `"h-bonds"`, `"all-bonds"`, `"h-angles"`, `"all-angles"` |
| `constraint_algorithm` | `constraint-algorithm` | str | `"LINCS"` | Solver: `"LINCS"` or `"SHAKE"` |
| `lincs_order` | `lincs-order` | int | `4` | LINCS expansion order |

### Velocity generation

| TOML field | MDP key | Type | Default | Description |
|------------|---------|------|---------|-------------|
| `gen_vel` | `gen-vel` | str | `"no"` | Generate initial velocities: `"yes"` or `"no"` |
| `gen_temp` | `gen-temp` | float | `300.0` | Temperature for velocity generation (K) |
| `gen_seed` | `gen-seed` | int | `-1` | Random seed (`-1` = use system clock) |

---

## Membrane protein example

`examples/membrane_1py6.toml` configures a run for bacteriorhodopsin (PDB
[1py6](https://www.rcsb.org/structure/1PY6)) in a DPPC bilayer, starting from
a pre-built [MemProtMD](https://memprotmd.bioch.ox.ac.uk/) system committed
at `tests/testdata/membrane/1py6/`:

```toml
[membrane_system]
structure = "tests/testdata/membrane/1py6/atomistic-system.pdb"
topology  = "tests/testdata/membrane/1py6/topol.top"

[md]
nsteps          = 500000
dt              = 0.002
pcoupl          = "C-rescale"
pcoupltype      = "semiisotropic"
ref_p           = [1.0, 1.0]
compressibility = [4.5e-5, 4.5e-5]
constraints     = "h-bonds"
```

```bash
gbsa-pipeline examples/membrane_1py6.toml -o results/1py6
```

**Known limitation.** As committed, this config loads successfully (stage
1-2) but does not complete a full run. MemProtMD's default output is
GROMOS53a6-parametrized (G96 bonds/angles), which BioSimSpace/Sire cannot
fully round-trip — see `scripts/convert_gromos_to_harmonic.py`, a standalone
data-prep script that converts them to an equivalent harmonic form; point
this config's `structure`/`topology` at its output. Even after that,
BioSimSpace/Sire's topology writer separately drops the lipid's
`[nonbond_params]`/`[pairtypes]` override tables on re-serialization, which
is enough to blow up minimization to a NaN potential energy for this
specific force field — a confirmed BioSimSpace/Sire library bug (full
investigation in
[`gbsa_pipeline.membrane`][gbsa_pipeline.membrane]'s module docstring), not
something fixable from this pipeline's code. `[membrane_system]` itself
works end-to-end for systems whose force field doesn't rely on such
overrides — e.g. an AMBER/CHARMM-parametrized membrane system built with
CHARMM-GUI + tleap.

Three things differ from the standard protein-ligand path:

1. **`[membrane_system]` replaces `[system]`.** The structure/topology are
   already a complete, solvated system — MemProtMD ran its own coarse-grained
   self-assembly simulation and converted the result back to atomistic
   detail — so parametrize and solvate (stages 1-2) are skipped; the pipeline
   loads the files directly and starts at SD minimization. To fetch a
   different PDB entry instead of the committed fixture, use
   [`gbsa_pipeline.membrane.fetch_memprotmd_system`][gbsa_pipeline.membrane.fetch_memprotmd_system].
2. **`pcoupltype = "semiisotropic"`.** A bilayer needs independent barostat
   scaling in-plane vs. along the membrane normal; isotropic coupling (the
   default) would squash it. `ref_p`/`compressibility` take the 2-value form
   for this — see the `[md]` Barostat table above.
3. **GBSA (`gmx_MMPBSA`) uses PB, not GB, with the membrane enabled.**
   GB has no membrane term (see
   [`mmbsa.MMPBSAConfig.__post_init__`][gbsa_pipeline.mmbsa.MMPBSAConfig]),
   and `mthick`/`mctrdz` should come from the equilibrated system you actually
   simulated, not a guess — that's what
   [`gbsa_pipeline.membrane.estimate_membrane_geometry`][gbsa_pipeline.membrane.estimate_membrane_geometry]
   is for. The full snippet is in the comments at the bottom of
   `examples/membrane_1py6.toml`.
