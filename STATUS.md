# STATUS

_Last updated: 2026-05-27 by Claude_

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
| 16 | UHF | 📋 prompt ready |
| 17 | TD-DFT (Casida TDA) | 📋 prompt ready |
| 18+ | TBD | 🔲 not planned yet |

## Current bottleneck

Phase 15 accepted (Claude review 2026-05-27). Next: Phase 16 UHF — assigned to Codex.

## Test suite

616 tests, all passing as of 2026-05-27.

Latest verification:
- `pytest tests/test_eom_ccsd.py -v`: 5 passed
- `scripts/validate_eom_ccsd.py`: wrote `outputs/logs/phase15_eom_ccsd.json` and
  `.txt`; H2 state 1 diff vs PySCF runtime reference `4.433892e-08` Ha;
  H2O state 1 diff vs PySCF runtime reference `7.154743e-08` Ha.
- `pytest tests/ -x`: 616 passed in 521.89s.

## Open scientific questions

- DFT PBE grid error ~2.5e-3 Ha for H₂O — grid size (n_rad=75, n_ang=302)
  is the bottleneck. Acceptable for now; larger grid is Phase 12 optional.
- HeH⁺ CCSD tolerance relaxed to 2e-4 Ha — 2-electron edge case in
  spin-orbital formulation. Documented in `tests/test_ccsd.py`.
- H₂O CCSD validation now agrees with the PySCF runtime reference at the
  Phase 11 requested geometry: diff `9.463267e-09` Ha.
- Phase 15 EOM-CCSD returns closed-shell singlet roots using an `S^2` filter
  to match PySCF RHF EOM-EE singlet references; documented in `SCIENCE.md`.
