# MOLEKUL

Experimental ab initio molecular simulation platform, built to benchmark CPU and GPU hardware.

## Goals

- First-principles electronic structure for small-to-medium molecules
- Born-Oppenheimer approximation + Gaussian basis sets + RHF SCF
- Geometry optimization with XYZ trajectory output
- Hardware benchmarking: multi-core CPU and NVIDIA RTX 5090 (GPU phases later)

## Approximations in use

| Level | Approximation |
|-------|--------------|
| Nuclear | Born-Oppenheimer (nuclei fixed during electronic solve) |
| Electronic | Restricted Hartree-Fock (RHF) — no electron correlation |
| Basis | Contracted Gaussian basis sets (STO-3G initially) |
| Relativity | None |
| Spin-orbit | None |

## Project structure

```
src/molekul/       Core library modules
tests/             Unit and integration tests
examples/          Example XYZ input files
outputs/           Runtime logs, JSON results, trajectory XYZ
basis/             Basis set data files
scripts/           Runnable example scripts
docs/              Documentation
profiling/         Performance profiling scripts and results
notebooks/         Jupyter exploration notebooks
```

## Phase status

- [x] Phase 1: Bootstrap — atoms, molecule, XYZ I/O, examples, tests
- [ ] Phase 2: Basis sets + one-electron integrals
- [ ] Phase 3: RHF SCF engine
- [ ] Phase 4: Energy evaluation + validation
- [ ] Phase 5: Geometry optimization
- [ ] Phase 6: Visualization outputs (CUBE, trajectory)
- [ ] Phase 7: Performance tuning (NumPy → Numba → CuPy)

## Quick start

```bash
pip install -e ".[dev]"
python scripts/run_example.py
pytest tests/
```

## Dependencies

- Python 3.10+
- NumPy, SciPy
- pytest (dev)
- CuPy, Numba (optional GPU phases)
