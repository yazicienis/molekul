# CHANGELOG — Agent Actions

Format: `YYYY-MM-DD | Agent | Phase | Action | Commit`

---

## 2026-05-23 | Claude | setup | Workflow bootstrap

- Created NEXT_AGENT.md, HANDOFF.md, STATUS.md, CHANGELOG_AGENT.md, SCIENCE.md
- Created PHASES.md (phase roadmap), WORKFLOW.md (protocol)
- Created `prompts/` with Codex prompts for phases 11, 14, 15, 16, 17
- No commits made — setup only

---

_Earlier history (pre-workflow) reconstructed from git log:_

| Date | Phase | Action | Commit |
|------|-------|--------|--------|
| 2026-04-18 | 13 | CIS implementation | — |
| 2026-04 | 12 | KS-DFT LDA/PBE | — |
| 2026-04 | 11 | CCSD spin-orbital (sign bug pending) | — |
| 2026-04 | 10 | Harmonic frequencies | — |
| 2026-04 | 1–9 | Core methods | — |
| 2026-05 | paper | SoftwareX V6 submission (human, local) | 4eb8de9 |

---

## 2026-05-23 | Claude | Phase 11 | CCSD sign fix review — ACCEPTED

- Reviewed commit c114255: `ccsd.py` sign/antisymmetriser corrections,
  `scripts/validate_ccsd.py`, `outputs/logs/phase11_ccsd.json/.txt`
- Confirmed SoftwareX paper files absent from commit ✓
- H2 diff 1.66e-7 Ha (< 1e-6), H2O diff 9.46e-9 Ha (< 1e-5) — both pass ✓
- 606/606 tests pass ✓
- Minor: validate_ccsd.py exit code only checks H2 (non-blocking)
- Updated NEXT_AGENT → Codex, Phase 14 CCSD(T)

---

## 2026-05-27 | Claude | Phase 15 | EOM-CCSD review — ACCEPTED

- Reviewed `src/molekul/eom_ccsd.py`: determinant-space similarity-transform H̄
- H2 diff 4.4e-8 Ha, H2O diff 7.2e-8 Ha — both well within 0.001 Ha tolerance ✓
- 616/616 tests pass ✓
- SCIENCE.md entries verified: imaginary threshold 1e-6 + singlet S² < 1e-4, Stanton & Bartlett (1993) ✓
- Runtime PySCF singlet references used ✓
- Minor: local `BOHR` constant in test instead of `ANGSTROM_TO_BOHR` (non-blocking)
- Updated NEXT_AGENT → Codex, Phase 16 UHF

---

## 2026-05-27 | Claude | Phase 14 | CCSD(T) validation + commit (resumed after shutdown)

- Ran `pytest tests/test_ccsdt.py -v`: 5 passed
- Ran `pytest tests/ -x`: 611 passed (606 existing + 5 new)
- Ran `scripts/validate_ccsdt.py`: H2 diff 1.66e-7 Ha, H2O diff 9.43e-9 Ha — both pass
- Generated `outputs/logs/phase14_ccsdt.json` and `.txt`
- Fixed syntax error in `validate_ccsdt.py` (mangled newline escape chars)
- Updated NEXT_AGENT → Codex, Phase 15 EOM-CCSD

---

## 2026-05-23 | Codex | Phase 11 | CCSD sign fix validation + commit

- Confirmed and revised spin-orbital CCSD sign/antisymmetriser corrections in
  `_make_intermediates_so`, `_t1_residual_so`, and `_t2_residual_so`
- Added `scripts/validate_ccsd.py`
- Generated `outputs/logs/phase11_ccsd.json` and `.txt`
- Ran `pytest tests/test_ccsd.py -v`: 10 passed
- Ran `pytest tests/ -x`: 606 passed
- H2 validation diff vs PySCF runtime reference: `1.661336e-07` Ha
- H2O validation diff vs PySCF runtime reference: `9.463267e-09` Ha
- Commit: `c114255`

## 2026-05-28 | Codex | Phase 15 | EOM-CCSD implementation — READY FOR REVIEW

- Added determinant-space EOM-CCSD-EE singlet implementation in `src/molekul/eom_ccsd.py`
- Added `tests/test_eom_ccsd.py` with H2/H2O PySCF reference checks and CIS comparison
- Added `scripts/validate_eom_ccsd.py`
- Generated `outputs/logs/phase15_eom_ccsd.json` and `.txt` with runtime PySCF references
- Documented new EOM numerical thresholds in `SCIENCE.md`
- Ran `pytest tests/test_eom_ccsd.py -v`: 5 passed
- Ran `scripts/validate_eom_ccsd.py`: H2 diff 4.43e-08 Ha, H2O diff 7.15e-08 Ha
- Ran `pytest tests/ -x`: 616 passed in 521.89s
- NEXT_AGENT → Claude, Phase 15 review
