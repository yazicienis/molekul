# HANDOFF

## Last action (Claude — reviewer, 2026-05-27, ACCEPTED)

Phase 17 TD-DFT/TDA review passed. All planned molecular phases (1–17) complete.

- TDA matrix diag(ε_a−ε_i) + 2J + 2f_xc correct for closed-shell singlet ✓
- H2 diff 1.2e-7 Ha, H2O diff 5.4e-5 Ha — both within 0.01 Ha ✓
- Oscillator strengths match PySCF ✓
- SCIENCE.md: XC kernel fd step, Casida (1995) + Burke (2005) ✓
- 628/628 tests pass ✓
- NEXT_AGENT → Human, Phase 18+ planning

---

## Previous action (Codex — implementer, 2026-05-28, READY FOR REVIEW)

Phase 17 TD-DFT/TDA implementation complete.

- Added `src/molekul/tddft.py` with `TDDFTResult` and `tddft_tda()`
- Reuses `ks_scf()` for the KS ground state and the existing Becke grid/AO evaluation utilities
- Builds the closed-shell singlet TDA matrix with `2*(ia|jb) + 2*(ia|f_xc|jb)`
- Computes LDA density kernels by finite-differencing `v_xc`; PBE path uses the same density-kernel approximation with fixed ground-state `sigma`
- Added CIS-style oscillator strengths and transition vectors
- Added `tests/test_tddft.py`
- Added `scripts/validate_tddft.py`
- Generated `outputs/logs/phase17_tddft.json` and `.txt` with runtime PySCF references
- Documented the TD-DFT XC-kernel finite-difference step in `SCIENCE.md`

Validation:
- `pytest tests/test_tddft.py -v`: 5 passed
- `scripts/validate_tddft.py`:
  - H2 LDA state 1 diff 1.24e-07 Ha (< 0.01)
  - H2O LDA state 1 diff 5.35e-05 Ha (< 0.01)
- `pytest tests/ -x`: 628 passed in 523.57s

Note for reviewer:
- The prompt suggested a “TDDFT better than CIS vs EOM-CCSD” test, but for STO-3G H2O LDA state 1 the actual numbers are TDDFT 0.4225 Ha, EOM-CCSD 0.4571 Ha, CIS 0.4851 Ha, so CIS is closer to EOM-CCSD. The implemented regression instead checks TDDFT is closer than CIS to the PySCF TDA-LDA reference, while validation enforces the stated PySCF acceptance criterion.

NEXT_AGENT → Claude, Phase 17 review.

---

## Last action (Claude — reviewer, 2026-05-27, ACCEPTED)

Phase 16 UHF review passed.

- H2O diff 9.3e-9 Ha; UHF-RHF diff 2.8e-14 Ha (machine zero) ✓
- OH diff 1.1e-8 Ha; ⟨S²⟩ = 0.7533 ∈ (0.75, 1.0) ✓
- H atom ⟨S²⟩ = 0.75 exact ✓
- 623/623 tests pass ✓
- SCIENCE.md: max_iter + diis_start, Pulay (1980) ✓
- NEXT_AGENT → Codex, Phase 17 TD-DFT

---

## Previous action (Codex — implementer, 2026-05-28, READY FOR REVIEW)

Phase 16 UHF implementation complete.

- Added `src/molekul/uhf.py` with `UHFResult` and `uhf_scf()`
- Supports arbitrary charge and multiplicity through `Molecule.n_alpha` and `Molecule.n_beta`
- Builds separate alpha/beta Fock matrices with shared Coulomb and spin-specific exchange
- Uses paired alpha/beta Pulay DIIS and computes UHF spin contamination `<S^2>`
- Added `tests/test_uhf.py`
- Added `scripts/validate_uhf.py`
- Generated `outputs/logs/phase16_uhf.json` and `.txt` with runtime PySCF references
- Documented UHF numerical parameters in `SCIENCE.md`

---

## Previous action (Claude — reviewer, 2026-05-27, ACCEPTED)

Phase 15 EOM-CCSD-EE review passed.

- H2 state 1 diff 4.4e-8 Ha (< 0.001) ✓
- H2O state 1 diff 7.2e-8 Ha (< 0.001) ✓
- 616/616 tests pass ✓
- SCIENCE.md: imaginary threshold 1e-6 + singlet S² < 1e-4, Stanton & Bartlett (1993) ✓
