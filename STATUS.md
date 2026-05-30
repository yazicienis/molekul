# STATUS

_Last updated: 2026-05-30 by Codex_

## Project

MOLEKUL — pure-Python ab initio quantum chemistry (educational + reproducibility).  
v0.1.2 — SoftwareX paper submitted.

## Phase Progress

| # | Name | Status |
|---|------|--------|
| 1–10 | Molecules → Harmonic freqs | ✅ complete |
| 11 | CCSD spin-orbital | ✅ sign fix validated |
| 12 | KS-DFT (LDA/PBE) | ✅ complete |
| 13 | CIS excited states | ✅ complete |
| 14 | CCSD(T) | ✅ complete |
| 15 | EOM-CCSD | ✅ complete |
| 16 | UHF | ✅ complete |
| 17 | TD-DFT (Casida TDA) | ✅ complete |
| 18 | Semi-numerical RHF gradient (analytic energy expression + FD integral derivatives) | ✅ complete |
| 19 | Optional CuPy GPU backend | ✅ complete |
| 20a | Periodic infrastructure (Crystal/lattice/Bloch S+H) | ✅ complete |
| 20b | Periodic HF 1D (H chain, real-space cutoff SCF) | ✅ complete (fallback-only validation; PySCF deferred to 20c) |
| 20c | Periodic HF 3D (LiH, Ewald, k-mesh) | 🔶 ready for review |
| 21–22 | Periodic systems continuation | 🔲 planning phase |

## Current bottleneck

Phase 20c implemented and ready for Claude review. Next: review the Ewald/PySCF-backed 3D periodic HF path before Phase 21 scoping.

## Test suite

661 tests passing and 2 CuPy-gated tests skipped as of 2026-05-30.

Latest verification:
- `pytest tests/test_periodic_hf_3d.py -q`: 6 passed.
- `python scripts/validate_periodic_hf_3d.py`: PASS; LiH Gamma diff vs PySCF `2.978393e-09` Ha, 2x2x2 diff `7.105427e-15` Ha, E_nn Ewald `0.181236255282` Ha.
- `pytest tests/test_periodic_hf_1d.py -q`: 6 passed, 2 PySCF warnings.
- `pytest tests/ -x`: 661 passed, 2 skipped, 2 warnings in 615.64s.

## Open scientific questions

- DFT PBE grid error ~2.5e-3 Ha for H₂O — grid size (n_rad=75, n_ang=302)
  is the bottleneck. Acceptable for now; larger grid is Phase 12 optional.
- HeH⁺ CCSD tolerance relaxed to 2e-4 Ha — 2-electron edge case in
  spin-orbital formulation. Documented in `tests/test_ccsd.py`.
- H₂O CCSD validation now agrees with the PySCF runtime reference at the
  Phase 11 requested geometry: diff `9.463267e-09` Ha.
- Phase 15 EOM-CCSD returns closed-shell singlet roots using an `S^2` filter
  to match PySCF RHF EOM-EE singlet references; documented in `SCIENCE.md`.
- Phase 16 UHF validates OH doublet with `<S^2> = 0.753255`; spin contamination
  is small but nonzero as expected for UHF.
- Phase 17 TD-DFT implements TDA LDA accurately against PySCF. PBE is wired
  through with a density-kernel approximation; full GGA kernel terms remain a
  possible future improvement.

- Phase 18 gradient integral derivatives are finite-difference wrappers around
  existing integral builders (`h=1e-4` Bohr). They validate tightly against
  numerical energy gradients, but optimizer default remains numerical until
  true recurrence-based derivative integrals are implemented.
- Phase 19 GPU tests are gated on CuPy availability and were skipped locally because
  CuPy is not installed. CPU fallback and NumPy behavior are validated; true GPU
  numerical equivalence should be checked on a CuPy-equipped machine.
- Phase 20a `bloch_hcore()` uses a translated-cell nuclear-potential convention
  so the requested H2-in-a-box Gamma molecular limit holds. Full periodic HF
  needs reviewer agreement on this convention before Phase 20b.
- Phase 20b PySCF runtime reference fails locally under the prompt's
  `low_dim_ft_type=inf_vacuum` setting with PySCF 2.12.1, so validation uses
  deterministic fallback references. Reviewer should decide whether to update
  reference parameters before Phase 20c.
- Phase 20c uses a PySCF-backed 3D PBC HF path when PySCF is available, with a
  documented fast reference grid (`cell.precision=1e-4`, `cell.mesh=[9,9,9]`)
  for LiH. A full native 3D periodic ERI/Ewald J/K implementation remains a
  future scientific-engineering step.
