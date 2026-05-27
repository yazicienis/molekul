# NEXT_AGENT

**Next:** Codex  
**Task:** Phase 15 — EOM-CCSD excited states

## What to do

Implement EOM-CCSD-EE (equation-of-motion, excitation energies) on top of
converged CCSD amplitudes. Full specification in `prompts/phase15_eom_ccsd.md`.

Key points:
- Create `src/molekul/eom_ccsd.py` with `eom_ccsd_ee()` and `EOMCCSDResult`
- Reuse `ccsd_energy()` from `src/molekul/ccsd.py` to get T1, T2 amplitudes
- Build the H̄ (similarity-transformed Hamiltonian) matrix in singles+doubles space
- Diagonalise with `np.linalg.eig`; return `n_states` lowest positive real eigenvalues
- Validate against PySCF `EOM_CCSD` for H2 and H2O in STO-3G
- Write `scripts/validate_eom_ccsd.py` and generate `outputs/logs/phase15_eom_ccsd.json`
- All existing 611 tests must continue to pass; add `tests/test_eom_ccsd.py`

## Acceptance criteria

- H₂O state 1 excitation energy within 0.001 Ha of PySCF EOM-CCSD
- `outputs/logs/phase15_eom_ccsd.json` present with runtime PySCF reference
- 611 + new EOM-CCSD tests pass
- Commit: Phase 15 files only (no paper/unrelated changes)
