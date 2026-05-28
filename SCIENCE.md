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

