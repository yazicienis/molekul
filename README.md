# MOLEKUL

A pure-Python ab initio quantum chemistry platform for education and reproducible benchmarking.

[![CI](https://github.com/yazicienis/molekul/actions/workflows/ci.yml/badge.svg)](https://github.com/yazicienis/molekul/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19743306.svg)](https://doi.org/10.5281/zenodo.19743306)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

MOLEKUL implements Restricted Hartree–Fock (RHF) SCF theory and MP2
correlation energy for closed-shell molecules entirely in Python and NumPy.
Every algorithmic step — from primitive integral evaluation to Fock-matrix
diagonalization — is traceable to a named function and a standard
quantum-chemistry reference.

**Validated against PySCF:** RHF/STO-3G energies for 14 molecules agree
within 5×10⁻⁸ Eₕ; MP2 correlation energies for 8 molecules agree within
2×10⁻⁷ Eₕ.

## Features

| Feature | Status |
|---------|--------|
| RHF SCF (DIIS + SAD guess + level shift) | ✅ validated |
| MP2 correlation energy | ✅ validated |
| STO-3G, 6-31G\*, cc-pVDZ (H–F) | ✅ built-in |
| Mulliken & Löwdin population analysis | ✅ |
| Electric dipole moment | ✅ |
| Geometry optimizer (numerical gradients) | ✅ |
| Harmonic frequencies (numerical Hessian) | experimental |
| CIS excited states | experimental |

## Installation

```bash
git clone https://github.com/yazicienis/molekul.git
cd molekul
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy. No SciPy or compiled extensions required.

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
print(f"RHF energy: {rhf_result.energy_total:.8f} Eh")   # -74.96258854 Eh

mp2_result = mp2_energy(mol, basis, rhf_result)
print(f"MP2 energy: {mp2_result.energy_total:.8f} Eh")   # -74.99844967 Eh
```

## Running tests

```bash
pytest tests/          # 606 tests
```

## Validation

```bash
python scripts/benchmark_14mol.py   # RHF vs PySCF, 14 molecules
```

Results are logged to `outputs/logs/benchmark_14mol.json`.

## Project structure

```
src/molekul/       Core library
tests/             606 automated tests
scripts/           Benchmark and validation scripts
outputs/logs/      JSON benchmark logs
examples/          Example XYZ geometries
docs/              Documentation
profiling/         Performance profiling results
```

## Known limitations

- Dense N⁴ ERI storage: practical limit ~N_AO ≤ 50
- Closed-shell RHF only (no UHF/ROHF)
- Element coverage: H–F only
- No integral screening, ECPs, or relativistic corrections
- Geometry optimization and frequencies use finite differences

## Citation

If you use MOLEKUL, please cite:

```bibtex
@software{yazici2026molekul,
  author  = {Yazici, Enis},
  title   = {{MOLEKUL}: A Pure-Python Ab Initio Quantum Chemistry Platform},
  year    = {2026},
  doi     = {10.5281/zenodo.19743306},
  url     = {https://github.com/yazicienis/molekul},
  version = {v0.1.1}
}
```

## License

MIT — see [LICENSE](LICENSE).
