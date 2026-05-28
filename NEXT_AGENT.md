# NEXT_AGENT

**Next:** Claude  
**Task:** Phase 16 — UHF review

## What to do

Review Codex Phase 16 implementation for unrestricted Hartree-Fock.

Files to review:
- `src/molekul/uhf.py`
- `tests/test_uhf.py`
- `scripts/validate_uhf.py`
- `outputs/logs/phase16_uhf.json`
- `outputs/logs/phase16_uhf.txt`
- `SCIENCE.md` Phase 16 entries

Validation already run by Codex on 2026-05-28:
- `pytest tests/test_uhf.py -v`: 7 passed
- `scripts/validate_uhf.py`: H2O diff 9.33e-09 Ha vs PySCF, OH diff 1.08e-08 Ha, H diff 1.18e-08 Ha
- `pytest tests/ -x`: 623 passed in 519.01s

## Acceptance review checklist

- Confirm UHF singlet H2O matches RHF within 1e-8 Ha.
- Confirm OH doublet converges, matches runtime PySCF within 1e-4 Ha, and has 0.75 < S^2 < 1.0.
- Confirm spin-contamination formula and alpha/beta Fock construction are consistent with UHF theory.
- Confirm no paper/unrelated files are included in the Phase 16 commit.
- If accepted, update workflow files and commit Phase 16 files only.
