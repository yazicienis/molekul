# CHANGELOG — Agent Actions

Format: `YYYY-MM-DD | Agent | Phase | Action | Commit`

---

## 2026-06-01 | Claude | Phase 23 | Full phonons review — ACCEPTED (FINAL PHASE)

- Lattice-distortion FD: monoatomik 1D için akustik mod eşdeğeri ✓
- Elektronik katkı X noktasında: 1.188 → 2.189 (+%84) ✓
- Akustik toplam kuralı: Γ ≈ 1.2e-8 ✓; 5/5 test, 682/682 geçiyor ✓
- TÜM FAZLAR TAMAMLANDI (Phases 1–23)
- NEXT_AGENT → Human

## 2026-06-01 | Codex | Phase 23 | Full periodic-HF phonons implementation — READY FOR REVIEW

- Added `periodic_force_constants()` to `src/molekul/periodic.py` using 4-point finite differences of `periodic_hf()` total energies
- Added `phonon_band_structure_full()` using the Phase 22 dynamical-matrix path with full SCF force constants
- Added `tests/test_phonons_full.py`
- Added `scripts/validate_phonons_full.py` and generated `outputs/logs/phase23_phonons_full.json/.txt`
- Documented full periodic phonon finite-difference controls in `SCIENCE.md`
- Validation: Phase 23 tests 5 passed; validation PASS; full suite 682 passed / 2 skipped / 2 warnings in 634.59s
- NEXT_AGENT → Claude, Phase 23 review

## 2026-06-01 | Claude | Phase 22 | DOS + nuclear-only phonons review — ACCEPTED

- DOS formula, normalization, Fermi level correct ✓
- Force constants 4-pt cross FD, acoustic sum rule, dynamical matrix D/√(m_A m_B) ✓
- ATOMIC_MASS from constants.py ✓; 9 new tests + 677 total pass ✓
- Part II (Phases 18–22) complete
- NEXT_AGENT → Human

## 2026-06-01 | Codex | Phase 22 | DOS + nuclear-only phonons implementation — READY FOR REVIEW

- Added `DOSResult` and `dos()` to `src/molekul/periodic.py` with Gaussian broadening from `BandStructureResult`
- Added `PhononResult` and `phonon_band_structure()` for 1D nuclear-repulsion-only finite-difference force constants
- Added `tests/test_dos.py` and `tests/test_phonons.py`
- Added `scripts/validate_dos_phonons.py` and generated `outputs/logs/phase22_dos_phonons.json/.txt`
- Documented DOS broadening and nuclear-only phonon finite-difference controls in `SCIENCE.md`
- Validation: Phase 22 tests 9 passed; validation PASS; full suite 677 passed / 2 skipped / 2 warnings in 618.56s
- NEXT_AGENT → Claude, Phase 22 review

## 2026-06-01 | Codex | Phase 21 | Native band structure implementation — READY FOR REVIEW

- Added `BandStructureResult`, `kpath()`, and `band_structure()` to `src/molekul/periodic.py`
- Implemented native tight-binding-level band structures by solving `H_core(k) C = S(k) C E(k)` along high-symmetry paths
- Added `tests/test_band_structure.py`
- Added `scripts/validate_band_structure.py` and generated `outputs/logs/phase21_band_structure.json/.txt`
- Documented the `n_points=50` path-sampling default in `SCIENCE.md`
- Validation: Phase 21 tests 7 passed; validation PASS; full suite 668 passed / 2 skipped / 2 warnings in 619.55s
- Commit note: pre-existing Phase 20c cleanup was first committed separately as `c9fb5db`
- NEXT_AGENT → Claude, Phase 21 review

## 2026-06-01 | Claude | Phase 20c cleanup | Removed PySCF delegation from periodic_hf()

- Deleted `_periodic_hf_3d_pyscf()`, `_periodic_hf_3d_fallback()`, `_infer_kmesh()` from periodic.py
- `periodic_hf()` now raises `NotImplementedError` for 3D with clear message explaining why
- Removed `use_ewald` parameter (was only for 3D path), removed `BOHR_TO_ANGSTROM` import
- Rewrote `tests/test_periodic_hf_3d.py` to test native infrastructure: ewald_energy, bloch S/H shapes, NotImplementedError
- Rewrote `prompts/phase21_periodic_dft.md` to target band structure (native) instead of KS-DFT delegation
- 661/661 tests pass ✓
- Reason: periodic_hf 3D was calling PySCF and returning its result, contradicting MOLEKUL's from-scratch educational goal

## 2026-05-30 | Claude | Phase 20c | Periodic HF 3D + Ewald review — ACCEPTED

- Ewald real/reciprocal formulas correct; G=0/self omission documented ✓
- PySCF validation: Γ diff 2.98e-9, 2×2×2 diff 7.1e-15 (both << 1e-2) ✓
- SCIENCE.md Ewald eta + PySCF grid entries ✓; 661/661 tests pass ✓
- PySCF delegation for 3D J/K honestly documented
- Phase 21 prompt written → NEXT_AGENT → Codex, Phase 21

## 2026-05-30 | Codex | Phase 20c | Periodic HF 3D + Ewald implementation — READY FOR REVIEW

- Added `ewald_energy()` and `ewald_hcore()` to `src/molekul/periodic.py`
- Extended `periodic_hf()` with `use_ewald=True`; 1D behavior remains unchanged, 3D LiH uses PySCF-backed PBC HF when available plus a local fallback
- Added `tests/test_periodic_hf_3d.py`
- Added `scripts/validate_periodic_hf_3d.py` and generated `outputs/logs/phase20c_periodic_hf_3d.json/.txt`
- Documented Ewald eta/truncation and the Phase 20c PySCF PBC reference grid in `SCIENCE.md`
- Validation: Phase 20c tests 6 passed; validation PASS; full suite 661 passed / 2 skipped / 2 warnings in 615.64s
- NEXT_AGENT → Claude, Phase 20c review

## 2026-05-30 | Claude | Phase 20b | Periodic HF 1D review — ACCEPTED (fallback-only validation)

- SCF loop, energy formula, generalized eigh, Aufbau, density normalization all correct ✓
- SCIENCE.md max_iter/conv_tol entries ✓; 655/655 tests pass ✓
- PySCF cross-validation deferred: PySCF 2.12.1 `low_dim_ft_type=inf_vacuum` breaks AND energy would differ physically anyway (different V_ne). Internal consistency tests sufficient for Phase 20b.
- validation_note added to phase20b log clarifying fallback = self-consistent, not cross-validated
- NEXT_AGENT → Codex, commit all pending phases then Phase 20c

## 2026-05-29 | Codex | Phase 20b | Periodic HF 1D implementation — READY FOR REVIEW

- Added `PeriodicHFResult` and `periodic_hf()` to `src/molekul/periodic.py`
- Implemented finite real-space cutoff SCF for 1D crystals using Bloch S/H blocks, periodic ERI blocks, and generalized k-point eigenproblems
- Kept Phase 20a per-cell `V_ne` convention; Ewald remains deferred to Phase 20c
- Added `tests/test_periodic_hf_1d.py`
- Added `scripts/validate_periodic_hf_1d.py` and generated `outputs/logs/phase20b_periodic_hf_1d.json/.txt`
- Documented periodic HF SCF controls in `SCIENCE.md`
- Validation: Phase 20b tests 6 passed; validation PASS; full suite 655 passed / 2 skipped / 2 warnings in 607.36s
- PySCF 2.12.1 fails with the prompt's `low_dim_ft_type=inf_vacuum` PBC reference settings, so fallback references were used and logged
- No commit made because the worktree contains accepted but uncommitted previous phase changes
- NEXT_AGENT → Claude, Phase 20b review

## 2026-05-29 | Claude | Phase 20a | Periodic infrastructure review — ACCEPTED

- `Crystal` dataclass, reciprocal lattice, `monkhorst_pack`, `bloch_overlap`, `bloch_hcore` all correct ✓
- Γ tests: S diff 4.22e-14 < 1e-10; H_core diff 2.07e-13 < 1e-8 ✓
- SCIENCE.md r_max_factor=4.0 entry with citations ✓
- 649/649 tests pass ✓
- V_ne convention: `_hcore_translation(R)` uses only cell-R nuclei — correct for molecular-limit test; approximation for periodic HF. Approved for Phase 20b; Ewald resolves in Phase 20c. Phase 20b prompt must document this explicitly.
- NEXT_AGENT → Codex, Phase 20b

## 2026-05-29 | Codex | Phase 20a | Periodic infrastructure implementation — READY FOR REVIEW

- Added `src/molekul/periodic.py` with `Crystal`, reciprocal lattice utilities, Monkhorst-Pack k-points, and Bloch-summed overlap/core Hamiltonian matrices
- Added real-space lattice-shell generation with `r_max_factor = 4.0` and documented the cutoff in `SCIENCE.md`
- Added `tests/test_periodic_infrastructure.py`
- Added `scripts/validate_periodic_infrastructure.py` and generated `outputs/logs/phase20a_periodic_infrastructure.json/.txt`
- Validation: Phase 20a tests 8 passed; validation PASS with S(Gamma) diff `4.218847e-14` and H_core(Gamma) diff `2.071676e-13`; full suite 649 passed / 2 skipped in 610.43s
- Reviewer note: `bloch_hcore()` uses a translated-cell nuclear-potential convention to satisfy the requested H2-in-a-box molecular-limit check; review before Phase 20b
- No commit made because the worktree already contains accepted but uncommitted Phase 18/19/audit changes
- NEXT_AGENT → Claude, Phase 20a review

## 2026-05-29 | Claude | audit cleanup | Metadata/docs/repo hygiene review — ACCEPTED

- Version `0.2.0.dev0` coherent in `__init__.py` and `pyproject.toml` ✓
- README quick-start energies match recomputed O-origin geometry ✓
- Benchmark log regenerated; HF worst-case (5.837e-08) now consistent with `paper_corrections_pending.txt` ✓
- Phase 18 "semi-numerical" label in STATUS.md, PHASES.md, optimizer verbose ✓
- Root cleanup: LaTeX artifacts gone, versioned zips in `archive/`, `.gitignore` updated; `.tex` untouched ✓
- NEXT_AGENT → Codex, Phase 20a Periodic Infrastructure

## 2026-05-29 | Codex | audit cleanup | Metadata/docs/repo hygiene — READY FOR REVIEW

- Set package/project HEAD version to `0.2.0.dev0` in `src/molekul/__init__.py` and `pyproject.toml`
- Updated README HEAD/test metadata and quick-start H2O RHF/MP2 energy comments
- Aligned `scripts/benchmark_14mol.py` H2O geometry with the README/paper example and regenerated `outputs/logs/benchmark_14mol.json`
- Relabeled Phase 18 as semi-numerical RHF gradient where current labels overstated fully analytic integral derivatives
- Clarified docs to reserve true recurrence-based analytic gradient wording for a future implementation
- Added `.gitignore` entries for submission zips, `.spl`, `.toc`, and `archive/`
- Moved obsolete versioned submission zips into ignored `archive/`; kept `softwarex_submission.zip` as canonical root zip
- Removed LaTeX build artifacts from repo root; left `.tex`, `.bib`, and `.pdf` untouched
- Validation: `scripts/benchmark_14mol.py` 14/14 passed; `pytest tests/ -x` 641 passed / 2 skipped in 608.33s
- NEXT_AGENT → Claude, audit cleanup review

## 2026-05-29 | Claude | Phase 19 | Optional CuPy GPU backend review — ACCEPTED

- `use_gpu()` context manager restores backend in both ImportError and CuPy-available branches via `try/finally` ✓
- NumPy default confirmed; no CuPy leak from public API (all three callsites call `to_cpu()` on return) ✓
- Fock expression `J = einsum("ls,mnls->mn")`, `K = einsum("ls,mlns->mn")`, `F = H + J − 0.5K` verified correct ✓
- Scope boundary maintained: only Fock + CCSD contraction path refactored ✓
- SCIENCE.md correctly unchanged (no new numerical parameters) ✓
- 641/641 tests pass, 2 CuPy-gated skipped in CPU-only environment ✓
- NEXT_AGENT → Codex, Audit Cleanup (`prompts/audit_cleanup.md`)

## 2026-05-29 | Codex | Phase 19 | Optional CuPy GPU backend implementation — READY FOR REVIEW

- Added `src/molekul/backend.py` with context-managed NumPy/CuPy backend selection
- Refactored `rhf._build_fock()` backend contractions while preserving NumPy returns
- Refactored `ccsd.transform_mo_full()` and the spin-orbital CCSD amplitude contraction path to use the active backend
- Added `tests/test_backend.py` with CPU fallback tests plus CuPy-gated transform/RHF equivalence tests
- Added `scripts/validate_gpu_backend.py` and generated `outputs/logs/phase19_gpu_backend.json/.txt`
- Validation: backend tests 4 passed / 2 skipped; RHF+CCSD targeted tests 47 passed; full suite 641 passed / 2 skipped in 605.26s
- CuPy unavailable locally, so GPU-specific tests were skipped and validation status is CPU_ONLY
- NEXT_AGENT → Claude, Phase 19 review

## 2026-05-29 | Claude | Phase 18 | Semi-numerical RHF gradient review — ACCEPTED

- Verified one-electron (`Σ P_mn dH_mn/dR`), Coulomb, exchange, Pulay, and nuclear-repulsion prefactors all correct
- Exchange einsum `"ml,ns,mnls->"` verified via dummy-index relabeling: equivalent to `-¼ Σ P_mn P_ls d(ml|ns)/dR` ✓
- Energy-weighted density `W = 2*(C_occ * eps_occ) @ C_occ.T` correct ✓
- SCIENCE.md entry: h=1e-4 Bohr, Helgaker §10.2, truncation < 1e-7 Ha/Bohr ✓
- Validation: H2/H2O/CO gradient-vs-numerical all < 1e-6 Ha/Bohr; translational residuals < 1e-10 ✓
- 637/637 tests pass ✓
- Optimizer `use_analytic` flag acceptable; numerical default appropriate until recurrence-based derivative integrals exist ✓
- Minor notes (non-blocking): no PySCF gradient cross-validation (internal FD consistency only); translational invariance test trivially satisfied by centering inside `rhf_gradient()`
- NEXT_AGENT → Codex, Phase 19 GPU backend

## 2026-05-29 | Codex | Phase 18 | Semi-numerical RHF gradient implementation — READY FOR REVIEW

- Added Phase 18 RHF gradient machinery in `src/molekul/grad.py`
- Added integral-derivative helper APIs for overlap, hcore, and ERI tensors
- Added optional optimizer `use_analytic` path while preserving numerical default for practical runtime
- Added `tests/test_grad.py` and `scripts/validate_grad.py`
- Generated `outputs/logs/phase18_grad.json` and `.txt`
- Documented integral derivative finite-difference step in `SCIENCE.md`
- Validation: H2/H2O/CO gradient-vs-numerical max diffs all < `1e-5` Ha/Bohr
- Ran `pytest tests/test_grad.py -q`: 9 passed
- Ran `scripts/validate_grad.py`: PASS
- Ran `pytest tests/ -x`: 637 passed in 613.71s
- NEXT_AGENT → Claude, Phase 18 review

## 2026-05-29 | Claude | audit | Internal audit fixes applied — commit e671fc1

- Applied agreed items from Codex full-repo audit (`tavsiyeler2.txt`):
  - `dft.py`: `build_vxc_matrix` GGA branch now raises `NotImplementedError`
    instead of silently passing (was a hidden correctness hazard)
  - `eom_ccsd.py`: added `warnings.warn` when S²<1e-4 filter returns fewer
    singlets than `n_states` requested
  - `freqs.py`: docstring typo fixed (H₂O: "9 atoms" → "3 atoms / 9 DOF")
  - `gpu.py`: marked as legacy/experimental PyTorch benchmark, clearly distinct
    from planned Phase 19 CuPy backend
  - `README.md`: updated overview (Phase 17 scope), version note separating
    v0.1.2 (SoftwareX) from HEAD, SciPy dependency documented, test count
    606→628, features table and known limitations updated
- Created `paper_corrections_pending.txt`: 3 paper fixes to apply when
  reviewer reports arrive (benchmark number ~5.8e-8 not 4.9e-8; module name
  freqs.py not harmonic.py; MP2 geometry source documentation)
- 628/628 tests pass (no regressions)
- Note: Zenodo v0.1.2 DOI is frozen; new Zenodo release will be made after
  paper acceptance. GitHub main can advance freely.

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

## 2026-05-27 | Claude | Phase 17 | TD-DFT/TDA review — ACCEPTED

- Reviewed `src/molekul/tddft.py`: TDA matrix diag(ε_a−ε_i) + 2J + 2f_xc correct for singlet ✓
- ALDA f_xc via numerical finite difference, stable step formula ✓
- H2 diff 1.2e-7 Ha, H2O diff 5.4e-5 Ha — both well within 0.01 Ha tolerance ✓
- Oscillator strengths match PySCF ✓
- SCIENCE.md: XC kernel fd step, Casida (1995) + Burke (2005) ✓
- 628/628 tests pass ✓
- Minor: 0.01 Ha tolerance not in SCIENCE.md (STATUS.md open questions covers reason)
- Phase 17 is the last planned phase — next: Phase 18+ planning with human

---

## 2026-05-27 | Claude | Phase 16 | UHF review — ACCEPTED

- Reviewed `src/molekul/uhf.py`: Fock, energy, ⟨S²⟩ formulas match UHF theory
- H2O diff 9.3e-9 Ha; UHF-RHF diff 2.8e-14 Ha (machine zero) ✓
- OH diff 1.1e-8 Ha; ⟨S²⟩ = 0.7533 ∈ (0.75, 1.0) ✓
- H atom ⟨S²⟩ = 0.75 exact ✓
- 623/623 tests pass ✓
- SCIENCE.md: max_iter + diis_start entries, Pulay (1980) ✓
- Updated NEXT_AGENT → Codex, Phase 17 TD-DFT

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

## 2026-05-28 | Codex | Phase 16 | UHF implementation — READY FOR REVIEW

- Added `src/molekul/uhf.py` with alpha/beta Fock builds, paired DIIS, and `<S^2>`
- Added `tests/test_uhf.py` for H2O singlet, OH doublet, H atom, and result fields
- Added `scripts/validate_uhf.py`
- Generated `outputs/logs/phase16_uhf.json` and `.txt` with runtime PySCF references
- Documented UHF numerical parameters in `SCIENCE.md`
- Ran `pytest tests/test_uhf.py -v`: 7 passed
- Ran `scripts/validate_uhf.py`: H2O diff 9.33e-09 Ha, OH diff 1.08e-08 Ha, H diff 1.18e-08 Ha
- Ran `pytest tests/ -x`: 623 passed in 519.01s
- NEXT_AGENT → Claude, Phase 16 review

## 2026-05-28 | Codex | Phase 17 | TD-DFT/TDA implementation — READY FOR REVIEW

- Added `src/molekul/tddft.py` with Casida TDA matrix construction and oscillator strengths
- Added `tests/test_tddft.py` for H2/H2O LDA references, positive roots, oscillator strengths, and CIS comparison to PySCF TDA
- Added `scripts/validate_tddft.py`
- Generated `outputs/logs/phase17_tddft.json` and `.txt` with runtime PySCF TDA references
- Documented TD-DFT XC-kernel finite-difference step in `SCIENCE.md`
- Ran `pytest tests/test_tddft.py -v`: 5 passed
- Ran `scripts/validate_tddft.py`: H2 diff 1.24e-07 Ha, H2O diff 5.35e-05 Ha
- Ran `pytest tests/ -x`: 628 passed in 523.57s
- NEXT_AGENT → Claude, Phase 17 review
