# NEXT_AGENT

**Next:** Claude  
**Task:** Phase 17 — TD-DFT/TDA review

## What to do

Review Codex Phase 17 implementation for TD-DFT in the Tamm-Dancoff approximation.

Files to review:
- `src/molekul/tddft.py`
- `tests/test_tddft.py`
- `scripts/validate_tddft.py`
- `outputs/logs/phase17_tddft.json`
- `outputs/logs/phase17_tddft.txt`
- `SCIENCE.md` Phase 17 entry

Validation already run by Codex on 2026-05-28:
- `pytest tests/test_tddft.py -v`: 5 passed
- `scripts/validate_tddft.py`: H2 diff 1.24e-07 Ha, H2O diff 5.35e-05 Ha vs runtime PySCF TDA-LDA
- `pytest tests/ -x`: 628 passed in 523.57s

## Acceptance review checklist

- Confirm singlet TDA matrix convention `2*(ia|jb) + 2*(ia|f_xc|jb)` matches PySCF RKS TDA for LDA.
- Confirm H2O LDA TDA state 1 remains within 0.01 Ha of runtime PySCF.
- Confirm oscillator strengths and transition-vector shapes are consistent with the CIS-style interface.
- Confirm no paper/unrelated files are included in the Phase 17 commit.
- If accepted, update workflow files and commit Phase 17 files only.
