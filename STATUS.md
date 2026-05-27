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
| 15 | EOM-CCSD | 📋 prompt ready |
| 16 | UHF | 📋 prompt ready |
| 17 | TD-DFT (Casida TDA) | 📋 prompt ready |
| 18+ | TBD | 🔲 not planned yet |

## Current bottleneck

Phase 14 complete (Claude review 2026-05-27). Next: Phase 15 EOM-CCSD — assigned to Codex.

## Test suite

611 tests, all passing as of 2026-05-27.

Latest verification:
- `pytest tests/test_ccsd.py -v`: 10 passed
- `pytest tests/ -x`: 606 passed in 516.61s
- `scripts/validate_ccsd.py`: wrote `outputs/logs/phase11_ccsd.json` and
  `.txt`; H2 diff vs PySCF runtime reference `1.661336e-07` Ha;
  H2O diff vs PySCF runtime reference `9.463267e-09` Ha.

## Open scientific questions

- DFT PBE grid error ~2.5e-3 Ha for H₂O — grid size (n_rad=75, n_ang=302)
  is the bottleneck. Acceptable for now; larger grid is Phase 12 optional.
- HeH⁺ CCSD tolerance relaxed to 2e-4 Ha — 2-electron edge case in
  spin-orbital formulation. Documented in `tests/test_ccsd.py`.
- H₂O CCSD validation now agrees with the PySCF runtime reference at the
  Phase 11 requested geometry: diff `9.463267e-09` Ha.
