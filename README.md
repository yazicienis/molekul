# MOLEKUL

A notebook-based teaching platform for ab initio quantum chemistry in pure Python.

[![CI](https://github.com/yazicienis/molekul/actions/workflows/ci.yml/badge.svg)](https://github.com/yazicienis/molekul/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19763107.svg)](https://doi.org/10.5281/zenodo.19763107)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

MOLEKUL is a notebook-based teaching platform for ab initio quantum chemistry.
It pairs a small pure-Python library with a 14-notebook curriculum that walks
from atoms, constants, and Gaussian integrals to SCF, correlation, DFT, excited
states, gradients, visualization, population analysis, and model periodic systems.

MOLEKUL is **not** a production alternative to PySCF, Psi4, ORCA, or similar
research codes. It favors readable NumPy implementations and small validated
examples that students can step through in a debugger.

**Validated against PySCF** where an independent molecular reference is appropriate.
The generated validation tables live in `benchmarks/validation_table.md` and
`benchmarks/periodic_validation.md`.

## Features

| Feature | Status |
|---------|--------|
| RHF SCF (DIIS + SAD guess + level shift) | ✅ validated |
| UHF (unrestricted, paired DIIS, ⟨S²⟩) | ✅ validated |
| MP2 correlation energy | ✅ validated |
| CCSD spin-orbital | ✅ validated |
| CCSD(T) driver | ✅ for two-electron regression; triples correction not independently validated |
| KS-DFT (LDA/PBE) | STO-3G LDA ok; wider-basis/PBE and cc-pVDZ grid behavior experimental |
| CIS excited states | ✅ validated |
| EOM-CCSD excited states (EE) | ✅ validated |
| TD-DFT TDA | STO-3G LDA ok; wider-basis/PBE behavior experimental |
| STO-3G, 6-31G\*, cc-pVDZ | STO-3G H-Ne; 6-31G\*/cc-pVDZ H, He, C, N, O, F |
| Mulliken & Löwdin population analysis | ✅ |
| Electric dipole moment | ✅ |
| Geometry optimizer (numerical gradients) | ✅ |
| Harmonic frequencies (numerical Hessian) | ✅ |
| Cube file export | ✅ |

## Installation

```bash
git clone https://github.com/yazicienis/molekul.git
cd molekul
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy. Core electronic-structure routines are implemented with NumPy; geometry optimization uses SciPy (`scipy.optimize.minimize`). No compiled extensions are required by the MOLEKUL core implementation.


## Notebook course map

Install the notebook extra and start with the index notebook:

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/00_index.ipynb
```

The notebooks are the primary teaching interface. They mirror `notebooks/README.md`:

| # | Notebook | Topic |
|---|----------|-------|
| 00 | `00_index.ipynb` | environment check and roadmap |
| 01 | `01_atoms_molecules_constants.ipynb` | atoms, molecules, units, nuclear repulsion |
| 02 | `02_one_electron_integrals.ipynb` | overlap, kinetic, nuclear attraction, Boys function |
| 03 | `03_basis_sets.ipynb` | STO-3G, 6-31G\*, cc-pVDZ basis sets |
| 04 | `04_hartree_fock_scf.ipynb` | RHF, DIIS, density matrices, energy components |
| 05 | `05_geometry_optimization.ipynb` | potential-energy scans and SciPy-backed optimization |
| 06 | `06_visualization_population.ipynb` | cube files, Mulliken/Lowdin charges, dipoles |
| 07 | `07_mp2_harmonic_frequencies.ipynb` | MP2 and finite-difference harmonic frequencies |
| 08 | `08_coupled_cluster.ipynb` | CCSD and CCSD(T) on tiny systems |
| 09 | `09_density_functional_theory.ipynb` | LDA/PBE Kohn-Sham DFT and grid limitations |
| 10 | `10_excited_states.ipynb` | UHF, CIS, TD-DFT/TDA, EOM-CCSD |
| 11 | `11_gradients_gpu.ipynb` | semi-numerical gradients and optional CuPy backend |
| 12 | `12_periodic_systems_phonons.ipynb` | 1D bands, DOS, and phonons |
| 13 | `13_full_workflow.ipynb` | capstone workflow |

## Quick start

```python
from molekul.molecule import Molecule
from molekul.atoms import Atom
from molekul.basis_sto3g import get_sto3g
from molekul.rhf import rhf_scf
from molekul.mp2 import mp2_energy
import numpy as np

ANG2BOHR = 1.8897259886

mol = Molecule([
    Atom('O', np.array([ 0.000,  0.000,  0.117]) * ANG2BOHR),
    Atom('H', np.array([ 0.000,  0.757, -0.469]) * ANG2BOHR),
    Atom('H', np.array([ 0.000, -0.757, -0.469]) * ANG2BOHR),
])

basis = get_sto3g()
rhf_result = rhf_scf(mol, basis)
print(f"RHF energy: {rhf_result.energy_total:.8f} Eh")   # -74.96294667 Eh

mp2_result = mp2_energy(mol, basis, rhf_result)
print(f"MP2 energy: {mp2_result.energy_total:.8f} Eh")   # -74.99844951 Eh
```

## Running tests

```bash
pytest tests/
```

## Validation

```bash
python benchmarks/validation_table.py
```

Results are regenerated in `benchmarks/validation_table.md` and `benchmarks/periodic_validation.md`.

## Project structure

```
src/molekul/       Core library
tests/             Automated regression and validation tests
scripts/           Benchmark and validation scripts
outputs/logs/      JSON benchmark logs
examples/          Example XYZ geometries
docs/              Documentation
profiling/         Performance profiling results
```

## Known limitations

- Dense N⁴ ERI storage: practical limit ~N_AO ≤ 50
- Element coverage: H–F only
- No integral screening, ECPs, or relativistic corrections
- Geometry optimization and frequencies use finite differences
- PBE functional and PBE TD-DFT kernel are implemented but not fully validated on grids

## Citation

If you use MOLEKUL, please cite:

```bibtex
@software{yazici2026molekul,
  author  = {Yazici, Enis},
  title   = {{MOLEKUL}: A Pure-Python Ab Initio Quantum Chemistry Platform},
  year    = {2026},
  doi     = {10.5281/zenodo.19763107},
  url     = {https://github.com/yazicienis/molekul},
  version = {v0.1.2}
}
```

## License

MIT — see [LICENSE](LICENSE).
