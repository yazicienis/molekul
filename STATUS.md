# STATUS

_Last updated: 2026-05-28 by Codex_

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
| 17 | TD-DFT (Casida TDA) | 🔍 ready for review |
| 18+ | TBD | 🔲 not planned yet |

## Current bottleneck

Phase 17 implementation complete; awaiting Claude review.

## Test suite

628 tests, all passing as of 2026-05-28.

Latest verification:
- `pytest tests/test_tddft.py -v`: 5 passed
- `scripts/validate_tddft.py`: wrote `outputs/logs/phase17_tddft.json` and `.txt`;
  H2 LDA state 1 diff vs PySCF `1.238525e-07` Ha; H2O LDA state 1 diff vs PySCF `5.353138e-05` Ha.
- `pytest tests/ -x`: 628 passed in 523.57s.

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
