---
title: CLI Usage
---

# CLI Usage

`gbsa-pipeline` provides a single command that runs the full GBSA MD simulation
pipeline from a TOML configuration file.

## Installation

The package is installed via Pixi:

```bash
pixi install -e dev
```

After installation the `gbsa-pipeline` executable is available inside the Pixi
environment.

## Running a simulation

```bash
gbsa-pipeline config.toml
```

Optionally specify an output directory:

```bash
gbsa-pipeline config.toml -o results/my_run
```

Enable verbose (debug) logging:

```bash
gbsa-pipeline config.toml -v
```

## Minimal configuration

Create a file called `config.toml` alongside your input files:

```toml
[system]
protein = "protein.pdb"
ligand  = "ligand.sdf"

[solvation]
water_model = "tip3p"
padding = 10.0

[md]
nsteps = 500000
dt     = 0.002
```

All sections except `[system]` are optional and fall back to sensible defaults.
See [Configuration Reference](configuration.md) for all available fields,
including the [membrane protein example](configuration.md#membrane-protein-example)
for a protein already embedded in a lipid bilayer.

## Output structure

Running the pipeline creates a numbered directory for each stage under the
output directory (default: `gbsa_output/` next to the config file):

| Directory | Contents |
|-----------|----------|
| `01_parametrize/` | GROMACS `.gro` and `.top` after force-field assignment |
| `02_solvated.gro` / `.top` | Solvated system with water box and counter-ions |
| `03_minimized.gro` / `.top` | Energy-minimized coordinates |
| `04_equilibrated.gro` / `.top` | NVT-heated (0 → 300 K) coordinates |
| `05_production.gro` / `.top` | Final production MD coordinates |
| `run_config.json` | Snapshot of the resolved configuration (for reproducibility) |

## Full NPT example

```toml
[system]
protein   = "apo_protein.pdb"
ligand    = "inhibitor.sdf"
net_charge = -1

[forcefield]
protein_ff    = "ff14SB"
ligand_ff     = "gaff2"
charge_method = "am1bcc"

[solvation]
water_model      = "tip3p"
box_shape        = "truncated_octahedron"
padding          = 12.0
ion_concentration = 0.15
neutralize       = true

[minimization]
nsteps = 5000
emtol  = 10.0

[equilibration]
simulation_time_ps = 500.0

[md]
integrator    = "md"
nsteps        = 2500000   # 5 ns at dt=0.002
dt            = 0.002
tcoupl        = "v-rescale"
ref_t         = 300.0
tau_t         = 0.1
pcoupl        = "Parrinello-Rahman"
ref_p         = 1.0
tau_p         = 2.0
compressibility = 4.5e-5
constraints   = "h-bonds"
```
