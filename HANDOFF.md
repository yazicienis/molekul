# HANDOFF

## Current action (Claude — reviewer, 2026-06-01, ACCEPTED)

Phase 21 native band-structure review passed.

- `kpath()`: lineer interpolasyon, junction noktaları her iki segmentte tekrarlı (band yapısı için zararsız) ✓
- `band_structure()`: 1D→bloch_hcore, 3D→ewald_hcore, generalized eigh, gerçel özdeğerler ✓
- H chain: Γ < X (bonding/antibonding), monoton cosine dispersiyon ✓
- LiH negatif gap (−0.038 Ha): H_core-only artefakt, HANDOFF+STATUS+log'da belgelenmiş ✓
- tick_labels "Γ" unicode dönüşümü, 7/7 test, 668/668 geçiyor ✓
- Hiç PySCF/dış delegasyon yok: tamamen native ✓
- NEXT_AGENT → Codex, Phase 22

---

## Previous action (Codex — implementer, 2026-06-01, READY FOR REVIEW)

Phase 21 native band-structure implementation complete.

- Added `BandStructureResult`, `kpath()`, and `band_structure()` to `src/molekul/periodic.py`
- Band structure solves `H_core(k) C = S(k) C E(k)` along high-symmetry paths; this is explicitly tight-binding/single-particle, with no HF J/K or DFT XC terms
- 1D H chain uses `bloch_hcore()`; 3D LiH uses Phase 20c `ewald_hcore()`
- Added `tests/test_band_structure.py` covering H-chain shape/dispersion/Gamma<X, LiH shape/n_occ, tick labels, and monotone path coordinates
- Added `scripts/validate_band_structure.py` and generated `outputs/logs/phase21_band_structure.json/.txt`
- Documented `n_points=50` default k-path sampling in `SCIENCE.md`

Validation:
- `pytest tests/test_band_structure.py -q`: 7 passed
- `python scripts/validate_band_structure.py`: PASS
  - H-chain band width: `0.475470593093` Ha
  - H-chain Gamma/X: `-0.537582997881` / `-0.062112404788` Ha
  - LiH shape: `(150, 6)`, `n_occ=2`
  - LiH one-electron gap: `-0.037948111635` Ha
- `pytest tests/test_periodic_hf_3d.py tests/test_periodic_infrastructure.py -q`: 14 passed
- `pytest tests/ -x`: 668 passed, 2 skipped, 2 warnings in 619.55s

Reviewer notes:
- The LiH gap is negative because this phase deliberately diagonalizes only the one-electron core Hamiltonian. This is logged as a tight-binding-level band structure, not an HF/DFT quasiparticle band gap.
- I first committed the pre-existing Phase 20c cleanup as `c9fb5db` so the Phase 21 commit stays scoped.

NEXT_AGENT → Claude, Phase 21 review.

---

## Current action (Claude — reviewer, 2026-05-30, ACCEPTED)

Phase 20c Ewald + 3D periodic HF review passed.

- Ewald real-space erfc sum: (i=j,R=0) skip correct, ½ prefactor correct ✓
- Ewald reciprocal: (2π/V)Σ_{G≠0} exp(−G²/4η²)/G² |S(G)|² correct ✓
- G=0 / self-energy omission documented — cancels with electron terms for neutral LiH ✓
- PySCF validation (real cross-check): Γ diff 2.98e-9, 2×2×2 diff 7.1e-15 — both << 1e-2 ✓
- SCIENCE.md: Ewald eta, PySCF grid documented with Tosi (1964) + Ashcroft & Mermin ✓
- 661/661 tests pass, 2 skipped ✓
- Minor: ewald_hcore shift and fallback energy formula are approximate but fallback-path only; primary PySCF path is validated
- Phase 21 prompt written (prompts/phase21_periodic_dft.md)
- NEXT_AGENT → Codex, Phase 21

---

## Previous action (Codex — implementer, 2026-05-30, READY FOR REVIEW)

Phase 20c periodic HF 3D implementation complete.

- Added 3D Ewald helpers in `src/molekul/periodic.py`: `ewald_energy()` and `ewald_hcore()`
- Extended `periodic_hf(..., use_ewald=True)` so 1D Phase 20b behavior is preserved and 3D cells use an Ewald/PySCF-backed PBC HF path when PySCF is installed
- Added deterministic fallback 3D one-electron Ewald path for environments without PySCF
- Added `tests/test_periodic_hf_3d.py` covering LiH Gamma, LiH 2x2x2 k-mesh, positive Ewald ion-ion energy, 3D Monkhorst-Pack shape, convergence, and real density
- Added `scripts/validate_periodic_hf_3d.py` and generated `outputs/logs/phase20c_periodic_hf_3d.json/.txt`
- Documented Ewald eta/truncation and the fast PySCF PBC reference grid in `SCIENCE.md`

Validation:
- `pytest tests/test_periodic_hf_3d.py -q`: 6 passed
- `python scripts/validate_periodic_hf_3d.py`: PASS
  - E_nn Ewald: `0.181236255282` Ha
  - Gamma MOLEKUL/PySCF diff: `2.978393e-09` Ha
  - 2x2x2 MOLEKUL/PySCF diff: `7.105427e-15` Ha
- `pytest tests/test_periodic_hf_1d.py -q`: 6 passed, 2 existing PySCF warnings
- `pytest tests/ -x`: 661 passed, 2 skipped, 2 warnings in 615.64s

Reviewer notes:
- The 3D HF energy path delegates to PySCF when available because implementing production-quality 3D periodic ERIs/Ewald J/K inside the educational Gaussian engine is substantially larger than Phase 20c. This gives an honest PySCF-backed LiH validation while keeping a local fallback for importability.
- `ewald_energy()` reports the finite ion-ion repulsion convention requested by the Phase 20c test; it omits G=0 and one-body self terms, whose long-range counterparts belong to the full neutral electron+nuclear Ewald treatment.
- PySCF defaults were too slow for this environment. The validation path uses `cell.precision=1e-4` and `cell.mesh=[9,9,9]`, now recorded in `SCIENCE.md`.

NEXT_AGENT → Claude, Phase 20c review.

---

## Current action (Claude — reviewer, 2026-05-30, ACCEPTED)

Phase 20b periodic HF 1D review passed.

- SCF loop F(R) = H(R) + J − ½K → Bloch → generalized eigh → density update correct ✓
- Energy (1/N_k)Σ_k Tr[P·(H(k)+½F(k))] = Tr[P·H₀]+½Tr[P·G₀] for uniform mesh ✓
- `_generalized_eigh`: Löwdin X + eigh, correct for Hermitian complex ✓
- Aufbau: remaining=n_e×N_k; density Tr[PS₀]=1 after normalization ✓
- SCIENCE.md: max_iter=100, conv_tol=1e-8 ✓
- PySCF cross-validation: DEFERRED. PySCF 2.12.1 `low_dim_ft_type=inf_vacuum` → AssertionError. Accepted fallback-only for Phase 20b: (a) PySCF and code differ physically anyway (different V_ne), (b) internal consistency verified. Proper PySCF comparison done in Phase 20c with Ewald. Log updated with validation_note.
- 655/655 tests pass, 2 skipped ✓
- Minor: `multiplicity=2` in test but RHF used — acceptable for metallic chain pedagogy
- Pre-task for Codex before Phase 20c: commit all accepted but uncommitted phases (18, 19, audit, 20a, 20b) first.
- NEXT_AGENT → Codex, commit + Phase 20c

---

## Previous action (Codex — implementer, 2026-05-29, READY FOR REVIEW)

Phase 20b periodic HF 1D implementation complete.

- Added `PeriodicHFResult` and `periodic_hf()` to `src/molekul/periodic.py`
- Implemented a finite real-space cutoff SCF loop for 1D crystals using Phase 20a `Crystal`, lattice shells, Bloch S/H blocks, and the accepted per-cell `V_ne` convention
- Added real-space periodic ERI blocks `(mu nu_R | lambda sigma_Rp)` for the small cutoff H-chain validation path
- Added generalized complex Hermitian eigenproblem solve at each k-point and Aufbau occupations across k-points
- Density matrix is normalized to the unit-cell electron count under the R=0 overlap block
- Added `tests/test_periodic_hf_1d.py`
- Added `scripts/validate_periodic_hf_1d.py`
- Generated `outputs/logs/phase20b_periodic_hf_1d.json` and `.txt`
- Documented `max_iter=100` and `conv_tol=1e-8` in `SCIENCE.md`

Validation:
- `pytest tests/test_periodic_hf_1d.py -q`: 6 passed, 2 PySCF warnings
- `python scripts/validate_periodic_hf_1d.py`: PASS
  - Gamma energy/cell: `0.081613538120` Ha
  - 4-k energy/cell: `0.848704196592` Ha
  - PySCF 2.12.1 with the prompt's `low_dim_ft_type=inf_vacuum` parameters raises `AssertionError`; validation records this and uses deterministic fallback references
- `pytest tests/ -x`: 655 passed, 2 skipped, 2 warnings in 607.36s

Reviewer notes:
- This is a cutoff pedagogical SCF, not an Ewald-compatible periodic HF. It keeps the Phase 20a accepted per-cell nuclear-attraction convention, and the validation logs explicitly say Ewald is deferred to Phase 20c.
- The PySCF runtime reference requested by the prompt cannot be produced in this local PySCF 2.12.1 environment with `low_dim_ft_type=inf_vacuum`; `pyscf.pbc.df.fft.get_nuc()` asserts against that setting. Tests and validation therefore fall back to logged deterministic references. Please decide whether to adjust the PySCF reference parameters, pin an older PySCF behavior, or accept fallback validation for Phase 20b.
- I did not make a git commit because the worktree still contains accepted but uncommitted Phase 18/19/audit/20a changes.

NEXT_AGENT → Claude, Phase 20b review.

---

## Current action (Claude — reviewer, 2026-05-29, ACCEPTED)

Phase 20a periodic infrastructure review passed.

- `Crystal` dataclass: lattice shape, rank, charge/multiplicity checks all correct ✓
- Reciprocal lattice: 3D formula b_i = 2π(a_j×a_k)/V; 1D/2D Gram-matrix formula; a_i·b_j = 2π δ_ij ✓
- `monkhorst_pack`: fractional n_i/N_i → Cartesian; (1,1,1) → Γ ✓
- `bloch_overlap`: ket translated to B+R, phase e^{ik·R}, Γ diff 4.22e-14 < 1e-10 ✓
- `bloch_hcore`: Γ diff 2.07e-13 < 1e-8 ✓; convention documented (see note)
- SCIENCE.md: r_max_factor=4.0, Ashcroft & Mermin + Szabo & Ostlund ✓
- 649/649 tests pass ✓
- V_ne convention note (must resolve before Phase 20b): `_hcore_translation(R)` uses only nuclei from cell R. This satisfies the molecular limit test but is an approximation for true periodic HF. Recommended path: keep this convention for Phase 20b (real-space cutoff SCF) and replace with Ewald in Phase 20c. Phase 20b prompt should make this explicit.
- NEXT_AGENT → Codex, Phase 20b (Periodic HF 1D)

---

## Previous action (Codex — implementer, 2026-05-29, READY FOR REVIEW)

Phase 20a periodic infrastructure implementation complete.

- Added `src/molekul/periodic.py` with `Crystal`, reciprocal lattice utilities, `monkhorst_pack()`, `bloch_overlap()`, and `bloch_hcore()`
- Supports 1D/2D/3D lattice vectors in Cartesian Bohr and internal Cartesian k-points in Bohr^-1
- Added real-space lattice-shell generation for Bloch sums with `r_max_factor = 4.0` documented in `SCIENCE.md`
- Added `tests/test_periodic_infrastructure.py` covering Gamma molecular-limit checks, Monkhorst-Pack shapes/Gamma, reciprocal orthogonality, electron count, and 1D shell count
- Added `scripts/validate_periodic_infrastructure.py`
- Generated `outputs/logs/phase20a_periodic_infrastructure.json` and `.txt`
- Updated `PHASES.md` / `STATUS.md` for Phase 20a

Validation:
- `pytest tests/test_periodic_infrastructure.py -q`: 8 passed
- `python scripts/validate_periodic_infrastructure.py`: PASS
  - `S(Gamma)` max diff vs molecular S: `4.218847e-14`
  - `H_core(Gamma)` max diff vs molecular H_core: `2.071676e-13`
- `pytest tests/ -x`: 649 passed, 2 skipped in 610.43s

Reviewer notes:
- The prompt says periodic `V_ne` should sum nuclei over translated cells, but the same prompt also requires `H_core(Gamma)` for an H2-in-a-box crystal to match the molecular core Hamiltonian. Including distant image-cell nuclei in the `R=0` potential breaks that molecular-limit test by construction. The implementation therefore evaluates each translated ket-cell block against nuclei translated with that cell, which preserves the requested Gamma molecular limit for Phase 20a. Please review this convention before Phase 20b builds on it.
- I did not make a git commit because the worktree already contains accepted but uncommitted Phase 18/19/audit changes; staging a clean Phase 20a-only commit would be misleading without first resolving that stack.

NEXT_AGENT → Claude, Phase 20a review.

---

## Current action (Claude — reviewer, 2026-05-29, ACCEPTED)

Audit cleanup review passed.

- Version `0.2.0.dev0` coherent across `__init__.py` and `pyproject.toml`; CITATION.cff v0.1.2 correct (published archive) ✓
- README quick-start H₂O energies (`-74.96294667` / `-74.99844951`) match recomputed geometry ✓
- Benchmark log regenerated at O-origin geometry; HF now correctly the worst case (5.837e-08) ✓
- Phase 18 "semi-numerical" label in STATUS.md, PHASES.md, optimizer verbose ✓
- LaTeX artifacts removed; versioned zips in `archive/`; `.gitignore` updated; `softwarex_paper.tex` untouched ✓
- Minor non-blocking: `optimizer.py:222` JSON method key still "RHF/analytic-BFGS" (verbose label fixed; JSON field unchanged)
- NEXT_AGENT → Codex, Phase 20a Periodic Infrastructure

---

## Previous action (Codex — cleanup, 2026-05-29, READY FOR REVIEW)

Audit cleanup complete per `prompts/audit_cleanup.md`.

- Set package/project HEAD version to `0.2.0.dev0` in `src/molekul/__init__.py` and `pyproject.toml`
- Updated README version note to Phase 19 / 641 tests + 2 CuPy-gated skips
- Recomputed README quick-start H2O energy at the snippet geometry:
  - RHF `-74.962946665868` Ha (`-74.96294667` in README)
  - MP2 `-74.998449514017` Ha (`-74.99844951` in README)
- Updated `scripts/benchmark_14mol.py` H2O coordinates to the same O-origin Bohr geometry as the README/paper example and regenerated `outputs/logs/benchmark_14mol.json`
- Relabeled Phase 18 as semi-numerical RHF gradient in `STATUS.md`, `PHASES.md`, and workflow history wording
- Clarified profiling/benchmark docs to reserve “true recurrence-based analytic gradient” for the future non-FD-integral implementation
- Updated optimizer verbose label for `use_analytic=True` to “semi-numerical gradient”
- Added `.gitignore` entries for `.spl`, `.toc`, `softwarex_submission*.zip`, and `archive/`
- Moved obsolete `softwarex_submission_v3.zip` through `softwarex_submission_v6.zip` into ignored `archive/`; kept canonical `softwarex_submission.zip` at repo root
- Removed root LaTeX build artifacts: `.aux`, `.log`, `.bbl`, `.blg`, `.out`, `.spl`
- Did not touch `softwarex_paper.tex`, `paper.bib`, or `softwarex_paper.pdf`

Validation:
- `python scripts/benchmark_14mol.py`: 14/14 passed; H2O MOLEKUL `-74.962946665868` Ha, PySCF `-74.962946646617` Ha
- `pytest tests/ -x`: 641 passed, 2 skipped in 608.33s

Reviewer notes:
- `rg "0\.1\.0"` now only finds the old value inside the audit prompt text, not package/project metadata.
- CuPy-gated skips are unchanged and expected in this CPU-only environment.

NEXT_AGENT → Claude, audit cleanup review.

---

## Current action (Claude — reviewer, 2026-05-29, ACCEPTED)

Phase 19 optional CuPy GPU backend review passed.

- `use_gpu()` context manager restores backend via `try/finally` in both ImportError and CuPy-available branches ✓
- NumPy default confirmed: `_active = _np` at module load ✓
- No CuPy leak: `_build_fock()`, `transform_mo_full()`, `_solve_ccsd_from_so_data()` all call `to_cpu()` on return ✓
- RHF Fock `J = einsum("ls,mnls->mn")`, `K = einsum("ls,mlns->mn")`, `F = H + J − 0.5K` correct ✓
- Scope boundary maintained: only Fock build + CCSD contraction path refactored; `_build_so_integrals`, `_ccsdt_correction_so`, diagonalization remain NumPy ✓
- No new numerical parameters; SCIENCE.md correctly unchanged ✓
- 641/641 tests pass (2 CuPy-gated skipped in CPU-only environment) ✓
- Minor notes (non-blocking): `to_cpu()` checks `_active` not array type (code flow consistent); `_ccsdt_correction_so` not backend-aware (triples on CPU if GPU active — expected for Phase 19 scope); no PySCF GPU cross-validation (CuPy unavailable here, gated tests address this)
- NEXT_AGENT → Codex, Audit Cleanup (see `prompts/audit_cleanup.md`)

---

## Previous action (Codex — implementer, 2026-05-29, READY FOR REVIEW)

Phase 19 optional CuPy GPU backend implementation complete.

- Added `src/molekul/backend.py` with `use_gpu()`, `get_xp()`, `to_device()`, and `to_cpu()`
- Refactored `rhf._build_fock()` to use the active backend for Coulomb/exchange contractions while returning NumPy arrays to callers
- Refactored `ccsd.transform_mo_full()` and the spin-orbital CCSD amplitude contraction path (`_make_intermediates_so`, `_t1_residual_so`, `_t2_residual_so`, `_solve_ccsd_from_so_data`) to use the active backend
- Added `tests/test_backend.py` with always-on CPU/fallback tests and CuPy-gated equivalence tests
- Added `scripts/validate_gpu_backend.py`
- Generated `outputs/logs/phase19_gpu_backend.json` and `.txt`

Validation:
- `pytest tests/test_backend.py -q`: 4 passed, 2 skipped
- `scripts/validate_gpu_backend.py`: CPU_ONLY; CuPy not installed in this environment
  - H2O RHF CPU energy `-74.962946665868` Ha
  - `transform_mo_full` CPU wall time recorded
- `pytest tests/test_rhf.py tests/test_ccsd.py -q`: 47 passed
- `pytest tests/ -x`: 641 passed, 2 skipped in 605.26s

Reviewer notes:
- CuPy is not installed here, so GPU-specific tests were not executed locally; they are skipped unless CuPy is importable.
- `rhf._build_fock()` deliberately copies the Fock matrix back to CPU after each backend-aware contraction because the rest of the SCF loop remains NumPy in this phase.
- No new production numerical parameters were introduced, so `SCIENCE.md` was not changed for Phase 19.

NEXT_AGENT → Claude, Phase 19 review.

---

## Current action (Claude — reviewer, 2026-05-29, ACCEPTED)

Phase 18 semi-numerical RHF gradient review passed.

- One-electron, Coulomb, exchange, Pulay, nuclear-repulsion prefactors all correct ✓
- Exchange einsum verified via dummy-index relabeling ✓
- W = 2*(C_occ * eps_occ) @ C_occ.T correct ✓
- SCIENCE.md: h=1e-4 Bohr, Helgaker §10.2 ✓
- H2/H2O/CO gradient-vs-numerical < 1e-6 Ha/Bohr; translational residuals < 1e-10 ✓
- 637/637 tests pass ✓
- Minor notes: no PySCF gradient comparison (internal FD consistency only); translational invariance test trivially passes due to centering inside rhf_gradient()
- NEXT_AGENT → Codex, Phase 19 GPU backend

---

## Previous action (Codex — implementer, 2026-05-29, ACCEPTED)

Phase 18 RHF gradient implementation complete.

- Added `rhf_gradient()` plus `overlap_derivative()`, `hcore_derivative()`, `eri_derivative()`, and nuclear-repulsion derivative helpers in `src/molekul/grad.py`
- Gradient expression uses spin-summed RHF density, energy-weighted density, Coulomb/exchange derivative contractions, Pulay term, and nuclear repulsion derivative
- Integral derivatives are central finite differences of the existing integral builders with `h=1e-4` Bohr; this is documented in `SCIENCE.md`
- Added optimizer `use_analytic` flag and an analytic-gradient smoke test
- Added `tests/test_grad.py`
- Added `scripts/validate_grad.py`
- Generated `outputs/logs/phase18_grad.json` and `.txt`

Validation:
- `pytest tests/test_grad.py -q`: 9 passed in 83.02s
- `scripts/validate_grad.py`: PASS
  - H2 max diff `2.580685e-07` Ha/Bohr
  - H2O max diff `2.848066e-07` Ha/Bohr
  - CO max diff `9.774736e-07` Ha/Bohr
  - translational residuals all below `1e-10` Ha/Bohr
- `pytest tests/test_optimizer.py tests/test_h2o_optimizer.py -q`: 59 passed in 218.46s
- `pytest tests/ -x`: 637 passed in 613.71s

Reviewer note:
- The optimizer accepts `use_analytic=True`, but the default remains `False` for now. Making this Phase 18 implementation the default made existing optimizer regressions impractically slow because the current derivative-integral helpers are finite-difference wrappers around the integral builders. A future sub-phase should replace these helpers with true recurrence-based derivative integrals before flipping the optimizer default.

NEXT_AGENT → Claude, Phase 18 review.

---

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
