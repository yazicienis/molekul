# NEXT_AGENT

**Next:** Claude  
**Task:** Phase 11 — CCSD sign fix review

## What to do

Review the Phase 11 commit from Codex.

Focus:
- `src/molekul/ccsd.py` sign and antisymmetriser corrections:
  - `_make_intermediates_so`: `Wvvvv` P-hat(ab) signs and `Wovvo` T1T1 term
  - `_t1_residual_so`: R1 `ovov` and `ovvv` contraction signs
  - `_t2_residual_so`: modified F intermediates and final T1 antisymmetrisers
- `scripts/validate_ccsd.py`
- `outputs/logs/phase11_ccsd.json` and `.txt`
- Handoff/status/changelog updates
- Confirm unrelated SoftwareX paper files were not included in the Phase 11
  commit.

Reviewer note:
- `pytest tests/test_ccsd.py -v`: 10 passed
- `pytest tests/ -x`: 606 passed
- H2 validation diff vs PySCF runtime reference: `1.661336e-07` Ha
- H2O validation diff vs PySCF runtime reference: `9.463267e-09` Ha

## Acceptance self-check (run before handing off)

- [ ] Review commit diff
- [ ] Confirm `outputs/logs/phase11_ccsd.json` exists
- [ ] Confirm H2 E_corr diff < 1e-6 Ha
- [ ] Confirm H2O E_corr diff < 1e-5 Ha
- [ ] Commit hash recorded in `CHANGELOG_AGENT.md`
