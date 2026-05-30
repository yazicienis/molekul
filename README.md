# MOLEKUL

A pure-Python ab initio quantum chemistry platform for education and reproducible benchmarking.

[![CI](https://github.com/yazicienis/molekul/actions/workflows/ci.yml/badge.svg)](https://github.com/yazicienis/molekul/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19763107.svg)](https://doi.org/10.5281/zenodo.19763107)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

MOLEKUL is a pure-Python ab initio quantum chemistry platform covering
single-reference methods from RHF to CCSD(T), DFT, and excited-state methods.
Every algorithmic step — from primitive integral evaluation to amplitude
equations — is traceable to a named function and a standard reference.

**Validated against PySCF** at each phase of development.

> **Version note:** v0.1.2 (SoftwareX paper, Zenodo DOI) covers Phases 1–9
> (RHF, MP2, geometry, population, basis sets). The current HEAD (`0.2.0.dev0`, Phase 19,
> 641 tests plus 2 CuPy-gated skips) additionally includes CCSD, CCSD(T),
> KS-DFT, CIS, EOM-CCSD, UHF, TD-DFT, semi-numerical RHF gradients,
> and an optional CuPy backend.

## Features

| Feature | Status |
|---------|--------|
| RHF SCF (DIIS + SAD guess + level shift) | ✅ validated |
| UHF (unrestricted, paired DIIS, ⟨S²⟩) | ✅ validated |
| MP2 correlation energy | ✅ validated |
| CCSD spin-orbital | ✅ validated |
| CCSD(T) perturbative triples | ✅ validated |
| KS-DFT (LDA/Slater+VWN validated; PBE experimental) | ✅/experimental |
| CIS excited states | ✅ validated |
| EOM-CCSD excited states (EE) | ✅ validated |
| TD-DFT TDA (LDA kernel validated; PBE experimental) | ✅/experimental |
| STO-3G, 6-31G\*, cc-pVDZ (H–F) | ✅ built-in |
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

**Requirements:** Python ≥ 3.10, NumPy. Core electronic-structure routines are NumPy-only; geometry optimization uses SciPy (`scipy.optimize.minimize`). No compiled extensions required.

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
pytest tests/          # 641 tests, plus 2 CuPy-gated skips on CPU-only systems
```

## Validation

```bash
python scripts/benchmark_14mol.py   # RHF vs PySCF, 14 molecules
```

Results are logged to `outputs/logs/benchmark_14mol.json`.

## Project structure

```
src/molekul/       Core library
tests/             641 automated tests
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
