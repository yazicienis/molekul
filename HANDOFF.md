# HANDOFF

## Last action (Codex — implementer, 2026-05-28, READY FOR REVIEW)

Phase 16 UHF implementation complete.

- Added `src/molekul/uhf.py` with `UHFResult` and `uhf_scf()`
- Supports arbitrary charge and multiplicity through `Molecule.n_alpha` and `Molecule.n_beta`
- Builds separate alpha/beta Fock matrices with shared Coulomb and spin-specific exchange
- Uses paired alpha/beta Pulay DIIS and computes UHF spin contamination `<S^2>`
- Added `tests/test_uhf.py`
- Added `scripts/validate_uhf.py`
- Generated `outputs/logs/phase16_uhf.json` and `.txt` with runtime PySCF references
- Documented UHF numerical parameters in `SCIENCE.md`

Validation:
- `pytest tests/test_uhf.py -v`: 7 passed
- `scripts/validate_uhf.py`:
  - H2O diff vs PySCF 9.33e-09 Ha; UHF-RHF diff -2.84e-14 Ha
  - OH diff vs PySCF 1.08e-08 Ha; `<S^2>` = 0.753255
  - H diff vs PySCF 1.18e-08 Ha; `<S^2>` = 0.750000
- `pytest tests/ -x`: 623 passed in 519.01s

NEXT_AGENT → Claude, Phase 16 review.

---

## Last action (Claude — reviewer, 2026-05-27, ACCEPTED)

Phase 15 EOM-CCSD-EE review passed.

- H2 state 1 diff 4.4e-8 Ha (< 0.001) ✓
- H2O state 1 diff 7.2e-8 Ha (< 0.001) ✓
- 616/616 tests pass ✓
- SCIENCE.md: imaginary threshold 1e-6 + singlet S² < 1e-4, Stanton & Bartlett (1993) ✓
- Runtime PySCF singlet references used ✓
- Minor non-blocking: local `BOHR` constant in test_eom_ccsd.py instead of ANGSTROM_TO_BOHR
- NEXT_AGENT → Codex, Phase 16 UHF

---

## Previous action (Codex — implementer, 2026-05-28, READY FOR REVIEW)

Phase 15 EOM-CCSD-EE implementation complete.

- Added `src/molekul/eom_ccsd.py` with `EOMCCSDResult` and `eom_ccsd_ee()`
- Reuses converged CCSD amplitudes from `ccsd_energy()`
- Builds the spin-orbital determinant-space `exp(-T) H exp(T) - E_CCSD`
  matrix in the 1h1p + 2h2p space
- Diagonalises with `np.linalg.eig` and returns closed-shell singlet roots via
  an `S^2` classifier, matching PySCF RHF EOM-EE singlet references
- Added `tests/test_eom_ccsd.py`
- Added `scripts/validate_eom_ccsd.py`
- Generated `outputs/logs/phase15_eom_ccsd.json` and `.txt` with runtime PySCF references
- Documented EOM imaginary-part and singlet `S^2` thresholds in `SCIENCE.md`

---

## Previous action (Claude — reviewer/completer, 2026-05-27, ACCEPTED)

Phase 14 CCSD(T) complete. Computer had shut down mid-session; Codex had
written all code but hadn't run validation or committed.

- Fixed syntax error in `scripts/validate_ccsdt.py` (mangled `\n` escape chars)
- Ran `pytest tests/test_ccsdt.py -v`: 5/5 passed
- Ran `pytest tests/ -x`: 611/611 passed (606 existing + 5 new)
- H2: diff 1.66e-7 Ha (< 1e-6) ✓ — H2O: diff 9.43e-9 Ha (< 1e-5) ✓
- NEXT_AGENT → Codex, Phase 15 EOM-CCSD
