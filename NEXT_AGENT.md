# NEXT_AGENT

**Next:** Codex  
**Task:** Phase 17 — TD-DFT (Casida TDA)

## What to do

Implement time-dependent DFT in the Tamm-Dancoff approximation (TDA) in a new
file `src/molekul/tddft.py`. Full specification in `prompts/phase17_tddft.md`.

Key points:
- Reuse KS-DFT from `src/molekul/dft.py` for the ground state
- Build the Casida TDA matrix A_{ia,jb} in the MO basis
- Diagonalise with `np.linalg.eigh`; return n_states lowest excitation energies
- Validate against PySCF `TDDFT` for H2O in STO-3G
- Write `scripts/validate_tddft.py` and generate `outputs/logs/phase17_tddft.json`
- All existing 623 tests must pass; add `tests/test_tddft.py`

## Acceptance criteria

- TD-DFT excitation energies within reasonable tolerance of PySCF TDA reference
- `outputs/logs/phase17_tddft.json` present with runtime PySCF reference
- 623 + new TD-DFT tests pass
- Commit: Phase 17 files only
