# NEXT_AGENT

**Next:** Codex  
**Task:** Phase 16 — UHF (Unrestricted Hartree–Fock)

## What to do

Implement spin-unrestricted HF in a new file `src/molekul/uhf.py`.
Full specification in `prompts/phase16_uhf.md`.

Key points:
- Create `src/molekul/uhf.py` with `uhf_scf()` and `UHFResult`
- Support arbitrary charge and multiplicity; derive n_alpha/n_beta from mol
- Separate alpha/beta Fock matrices with DIIS; reproduce RHF energy for singlets
- Compute ⟨S²⟩ spin contamination
- Validate: H2O singlet matches RHF ± 1e-8; OH doublet within 1e-4 Ha of PySCF
- Write `scripts/validate_uhf.py` and generate `outputs/logs/phase16_uhf.json`
- All existing 616 tests must pass; add `tests/test_uhf.py`

## Acceptance criteria

- `pytest tests/test_uhf.py -v` all pass
- `pytest tests/ -x` 616 + new tests, no regressions
- H2O UHF singlet energy matches RHF within 1e-8 Ha
- OH doublet converged and within 1e-4 Ha of PySCF; 0.75 < ⟨S²⟩ < 1.0
- `outputs/logs/phase16_uhf.json` present with runtime PySCF reference
- Commit: Phase 16 files only
