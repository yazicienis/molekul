# SCIENCE.md — Numerical Parameter Registry

Every non-trivial numerical parameter used in production code must have an
entry here. Codex appends when setting a new parameter. Claude verifies citations.

Format:
```
## <parameter_name>
- Value: ...
- File: src/molekul/...
- Set by: Agent | Human | Date
- Justification: ...
- Citation: ...
```

---

## SCF convergence threshold
- Value: `conv_tol = 1e-9` (max |ΔP| element)
- File: `src/molekul/rhf.py`, `src/molekul/uhf.py`
- Set by: Human (initial design)
- Justification: Standard tight convergence for energy accuracy to ~1e-8 Ha.
  Consistent with PySCF default `conv_tol=1e-9`.
- Citation: Szabo & Ostlund, §3.4; PySCF docs

## DIIS subspace size
- Value: `diis_size = 8`
- File: `src/molekul/rhf.py`, `src/molekul/uhf.py`
- Set by: Human
- Justification: 6–10 vectors is standard; 8 balances memory vs convergence
  speed for molecules up to ~20 basis functions.
- Citation: Pulay (1980) Chem. Phys. Lett. 73, 393

## ERI screening threshold
- Value: None (all integrals computed)
- File: `src/molekul/eri.py`
- Set by: Human
- Justification: Educational clarity. Screening would obscure the algorithm.
  Not suitable for large molecules (> ~50 basis functions).
- Citation: N/A

## DFT Becke integration grid
- Value: `n_rad=75`, `n_ang=302` (Lebedev)
- File: `src/molekul/dft.py`
- Set by: Human (Phase 12)
- Justification: Matches PySCF `grids.level=3` default. Gives ~1e-5 Ha error
  for LDA, ~1e-3 Ha for PBE (GGA gradient correction amplifies grid error).
  See STATUS.md open issues.
- Citation: Becke (1988) J. Chem. Phys. 88, 2547; Lebedev (1976)

## CCSD convergence threshold
- Value: `conv_tol = 1e-9` (max residual element)
- File: `src/molekul/ccsd.py`
- Set by: Human (Phase 11)
- Justification: Matches RHF threshold; gives E_corr reproducible to ~1e-8 Ha.
- Citation: Stanton et al. (1991) J. Chem. Phys. 94, 4334

## CCSD max iterations
- Value: `max_iter = 100`
- File: `src/molekul/ccsd.py`
- Set by: Human
- Justification: Small basis (STO-3G) converges in < 20 iterations for tested
  molecules. 100 is a safe ceiling.
- Citation: N/A

## Numerical gradient step size
- Value: `h = 0.001` Bohr (central differences)
- File: `src/molekul/grad.py`
- Set by: Human (Phase 5)
- Justification: Central difference error ~h²; at h=0.001 Bohr, gradient error
  ~1e-6 Eh/Bohr, consistent with SCF convergence.
- Citation: Helgaker, Jørgensen & Olsen §10.1

## STO-3G basis
- Value: 3 Gaussians per STO, standard Hehre/Stewart parameters
- File: `src/molekul/basis_sto3g.py`
- Set by: Human (Phase 3)
- Justification: Minimal basis for educational use; fast enough to run all
  tests in < 10 minutes.
- Citation: Hehre, Stewart & Pople (1969) J. Chem. Phys. 51, 2657

## EOM-CCSD eigenvalue imaginary-part warning threshold
- Value: `1e-6` Ha
- File: `src/molekul/eom_ccsd.py`
- Set by: Codex | 2026-05-28
- Justification: EOM-CCSD uses a real non-Hermitian matrix, so physically
  accepted excitation energies should be real; imaginary components at or below
  this scale are treated as numerical diagonalisation noise.
- Citation: Stanton & Bartlett (1993) J. Chem. Phys. 98, 7029

## EOM-CCSD singlet S^2 filter threshold
- Value: `abs(<S^2>) < 1e-4`
- File: `src/molekul/eom_ccsd.py`
- Set by: Codex | 2026-05-28
- Justification: Closed-shell RHF EOM-EE validation targets singlet roots.
  Numerical right eigenvectors classified with this threshold separate singlets
  (`S(S+1)=0`) from triplets (`S(S+1)=2`) by several orders of magnitude in the
  STO-3G validation systems.
- Citation: Stanton & Bartlett (1993) J. Chem. Phys. 98, 7029

## UHF max iterations
- Value: `max_iter = 100`
- File: `src/molekul/uhf.py`
- Set by: Codex | 2026-05-28
- Justification: Mirrors the existing RHF and CCSD iteration ceiling in this
  small-basis educational code; Phase 16 validation systems converge in 13 or
  fewer iterations.
- Citation: N/A

## UHF DIIS start iteration
- Value: `diis_start = 2`
- File: `src/molekul/uhf.py`
- Set by: Codex | 2026-05-28
- Justification: Matches the existing RHF DIIS warm-up behavior, allowing one
  prior Fock/error pair before Pulay extrapolation.
- Citation: Pulay (1980) Chem. Phys. Lett. 73, 393

## TD-DFT XC kernel finite-difference step
- Value: `max(1e-6 * max(rho, 1.0), 1e-8)`
- File: `src/molekul/tddft.py`
- Set by: Codex | 2026-05-28
- Justification: The adiabatic LDA/GGA TDA kernel needs `d v_xc / d rho`.
  A relative step with an absolute floor is stable on low-density grid points
  and reproduces PySCF STO-3G LDA TDA references within `5.4e-5` Ha for H2O.
- Citation: Casida (1995), in Recent Advances in Density Functional Methods;
  Burke et al. (2005) J. Chem. Phys. 123, 062206


## Integral derivative finite-difference step
- Value: `h = 1e-4` Bohr for central differences of overlap, core-Hamiltonian,
  and ERI tensors used by the Phase 18 RHF gradient expression.
- File: `src/molekul/grad.py`
- Set by: Codex | 2026-05-29
- Justification: The derivative is applied to smooth Gaussian integral tensors;
  `1e-4` Bohr keeps truncation error below the `1e-7` integral-derivative test
  tolerance while avoiding cancellation seen at substantially smaller steps.
- Citation: Helgaker, Jørgensen & Olsen §10.2

## Periodic Bloch lattice-sum cutoff
- Value: `r_max_factor = 4.0`; lattice vectors with `|R| <= 4 * max(|a_i|)`
  are included in Phase 20a Bloch-summed one-electron matrices.
- File: `src/molekul/periodic.py`
- Set by: Codex | 2026-05-29
- Justification: This finite real-space shell is large enough for the Phase 20a
  H2-in-a-box molecular-limit validation while keeping the educational lattice
  sum explicit and inexpensive. Later periodic HF phases should replace this
  with physically converged real/reciprocal-space treatments where needed.
- Citation: Ashcroft & Mermin, Ch. 8; Szabo & Ostlund, Appendix A for Gaussian
  one-electron integral decay.

## Periodic HF 1D SCF controls
- Value: `max_iter = 100`, `conv_tol = 1e-8` on max density-matrix change.
- File: `src/molekul/periodic.py`
- Set by: Codex | 2026-05-29
- Justification: Mirrors the molecular SCF iteration ceiling while using a
  density convergence threshold appropriate for the small STO-3G H-chain
  validation. The Phase 20b validation converges in 2 iterations for the tested
  Gamma and 4-k meshes.
- Citation: Szabo & Ostlund, §3.4; Ashcroft & Mermin, Ch. 8.
## Ewald splitting parameter eta
- Value: `eta = sqrt(pi) / min(|a_i|)` for 3D crystals; real-space terms
  are truncated at `4 / eta` and reciprocal-space terms at `4 * eta`.
- File: `src/molekul/periodic.py`
- Set by: Codex | 2026-05-30
- Justification: The default eta balances real-space and reciprocal-space
  convergence for compact 3D cells. The paired `4 / eta` and `4 * eta`
  truncations follow the same Gaussian-screening decay scale and keep the
  Phase 20c LiH Ewald nuclear repulsion stable for the validation tolerance.
- Citation: Tosi (1964) Solid State Physics 16, 1; Ashcroft & Mermin, Appendix B.

## Phase 20c PySCF 3D PBC reference grid
- Value: `cell.precision = 1e-4`, `cell.mesh = [9, 9, 9]` for the LiH
  STO-3G PySCF PBC reference path used by `periodic_hf()` when PySCF is
  available and by `scripts/validate_periodic_hf_3d.py`.
- File: `src/molekul/periodic.py`, `scripts/validate_periodic_hf_3d.py`,
  `tests/test_periodic_hf_3d.py`
- Set by: Codex | 2026-05-30
- Justification: The default PySCF PBC mesh for the prompt's LiH cell is too
  slow for the repository test budget in this environment. This grid gives a
  deterministic 3D PBC HF reference and validates MOLEKUL's Phase 20c wrapper
  within the requested `1e-2` Ha tolerance.
- Citation: PySCF PBC documentation; Sun et al. (2020) J. Chem. Phys. 153, 024109.
## Band-structure k-path default sampling
- Value: `n_points = 50` per high-symmetry path segment.
- File: `src/molekul/periodic.py`
- Set by: Codex | 2026-06-01
- Justification: Provides a smooth educational band path for the small H-chain
  and LiH STO-3G examples while keeping validation fast and data files compact.
  The value is a plotting/data-resolution default, not a convergence threshold.
- Citation: Ashcroft & Mermin, Ch. 8; standard band-structure plotting practice.
## DOS Gaussian broadening
- Value: `sigma = 0.02` Ha and `n_grid = 500` for Gaussian-broadened density
  of states.
- File: `src/molekul/periodic.py`
- Set by: Codex | 2026-06-01
- Justification: A 0.02 Ha width smooths the sparse educational H-chain band
  samples without hiding the band range; 500 grid points keeps the DOS curve
  smooth while remaining inexpensive. These are visualization defaults, not
  electronic-structure convergence thresholds.
- Citation: Ashcroft & Mermin, Ch. 8; standard Gaussian broadening practice for DOS plotting.

## Nuclear-only phonon finite-difference controls
- Value: `h = 0.01` Bohr and `n_points = 30` q-points per high-symmetry path
  segment for `phonon_band_structure()`.
- File: `src/molekul/periodic.py`
- Set by: Codex | 2026-06-01
- Justification: The displacement is large enough to avoid cancellation in
  second finite differences of nuclear repulsion while remaining small relative
  to the 1.8 Bohr H-chain lattice spacing. The q-path sampling is a compact
  plotting default. Phase 22 phonons intentionally include only nuclear
  repulsion force constants; electronic Hellmann-Feynman/Pulay contributions
  are omitted and the frequencies are not physical production phonons.
- Citation: Born & Huang, Dynamical Theory of Crystal Lattices; Ashcroft & Mermin, Ch. 22.
## Full periodic phonon finite-difference controls
- Value: `h = 0.01` Bohr and `n_points = 30` q-points per high-symmetry path
  segment for `phonon_band_structure_full()`; `r_max_factor = 4.0` inherited
  from the periodic Bloch lattice-sum cutoff.
- File: `src/molekul/periodic.py`
- Set by: Codex | 2026-06-01
- Justification: Reuses the Phase 22 displacement scale so nuclear-only and
  full periodic-HF force constants are directly comparable. Phase 23 evaluates
  finite differences of the periodic HF total energy, so both electronic and
  nuclear energy responses are included for the 1D H-chain validation.
- Citation: Born & Huang, Dynamical Theory of Crystal Lattices; Ashcroft & Mermin, Ch. 22.

