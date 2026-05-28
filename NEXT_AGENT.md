# NEXT_AGENT

**Next:** Codex  
**Task:** Phase 18 — Analytic RHF gradient

## What to do

Implement analytic RHF nuclear gradients. Full specification in
`prompts/phase18_analytic_grad.md`.

Key points:
- Add `rhf_gradient()` to `src/molekul/grad.py` (alongside existing numerical grad)
- Implement derivative integrals for overlap, h_core, ERI
- Update optimizer to use analytic forces by default
- Validate: max |analytic - numerical| < 1e-5 Ha/Bohr for H2, H2O, CO
- Write `scripts/validate_grad.py` and `outputs/logs/phase18_grad.json`
- All existing 628 tests must pass; add `tests/test_grad.py`

## Context

All planned molecular phases (1–17) are complete. The codebase now covers:
- RHF, UHF, MP2, CCSD, CCSD(T), KS-DFT (LDA/PBE)
- CIS, EOM-CCSD, TD-DFT (TDA)
- Geometry optimization, harmonic frequencies, population analysis
- 628 tests, all passing

## Acceptance criteria

- max |analytic - numerical| < 1e-5 Ha/Bohr for all test molecules
- Translational sum of forces < 1e-10 Ha/Bohr
- 628 + new tests pass
- Commit: Phase 18 files only
