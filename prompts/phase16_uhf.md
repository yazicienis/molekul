# Phase 16: UHF — Unrestricted Hartree–Fock

## Context

RHF is implemented in `src/molekul/rhf.py` and exports:
```python
@dataclass
class RHFResult:
    energy_total, energy_hf, C, eps, P, converged, n_iter, n_occ, n_basis

def rhf_scf(mol, basis_fn, verbose=False, max_iter=100, conv_tol=1e-9) -> RHFResult
```

One-electron integrals (`S, T, V_ne`) and ERI are built in
`integrals.py` / `eri.py` and work for any molecule.

The current codebase only handles closed-shell (even electron count,
singlet). UHF will extend MOLEKUL to open-shell doublets and triplets.

## Objective

Implement spin-unrestricted HF (UHF) in a new file `src/molekul/uhf.py`.
Support arbitrary charge and multiplicity (2S+1).

## Theory

Szabo & Ostlund §3.8; Helgaker, Jørgensen & Olsen Ch. 10.

Separate alpha/beta Fock matrices:
```
F^α = H_core + J^α + J^β - K^α
F^β = H_core + J^α + J^β - K^β
```
Density matrices:
```
P^α_{μν} = Σ_i C^α_{μi} C^α_{νi}   (sum over n_alpha occupied)
P^β_{μν} = Σ_i C^β_{μi} C^β_{νi}   (sum over n_beta  occupied)
```
Total energy:
```
E = ½ Tr[(P^α + P^β)(H_core + F^α) + (P^α + P^β)(H_core + F^β)]
  + V_nn
```
UHF SCF: diagonalise F^α and F^β separately each iteration.
Use same DIIS strategy as RHF (separate error vectors per spin).

Spin contamination:
```
⟨S²⟩ = S(S+1) + n_β - Σ_{ij} |⟨ψ^α_i|ψ^β_j⟩|²
```
where the overlap is `Σ_{μν} C^α_{μi} S_{μν} C^β_{νj}`.

## Implementation

File: `src/molekul/uhf.py` (new file)

```python
@dataclass
class UHFResult:
    energy_total: float
    energy_hf: float          # same as energy_total (no post-HF here)
    Ca: np.ndarray            # alpha MO coefficients, shape (n_basis, n_basis)
    Cb: np.ndarray            # beta  MO coefficients
    epsa: np.ndarray          # alpha orbital energies
    epsb: np.ndarray          # beta  orbital energies
    Pa: np.ndarray            # alpha density matrix
    Pb: np.ndarray            # beta  density matrix
    S2: float                 # ⟨S²⟩ expectation value
    converged: bool
    n_iter: int
    n_occ_a: int
    n_occ_b: int
    n_basis: int

def uhf_scf(
    mol: Molecule,
    basis_fn,
    verbose: bool = False,
    max_iter: int = 100,
    conv_tol: float = 1e-9,
) -> UHFResult:
    """Unrestricted HF SCF. mol.multiplicity determines n_alpha, n_beta."""
    ...
```

- Derive `n_alpha`, `n_beta` from `mol.charge` and `mol.multiplicity`:
  `n_elec = sum(Z) - charge`, `n_alpha = (n_elec + (mult-1))//2`.
- For singlet closed-shell (n_alpha == n_beta), UHF must reproduce the RHF
  energy exactly (use as a sanity check in tests).
- Core Hamiltonian guess (set Pa = Pb = 0 initially, build Fock from H_core).

## Tests

File: `tests/test_uhf.py`

| Molecule | Charge | Mult | Basis | Property | PySCF ref | Tolerance |
|----------|--------|------|-------|----------|-----------|-----------|
| H₂O | 0 | 1 | STO-3G | E_total | −74.9626 | 1e-4 |
| H₂O | 0 | 1 | STO-3G | ⟨S²⟩ | 0.0 | 1e-10 |
| OH radical | 0 | 2 | STO-3G | E_total | TBD PySCF | 1e-4 |
| OH radical | 0 | 2 | STO-3G | ⟨S²⟩ | ~0.75+contamination | 0.01 |
| H atom | 0 | 2 | STO-3G | E_total | TBD PySCF | 1e-5 |

Required tests:
- `test_h2o_uhf_matches_rhf` — closed-shell UHF energy == RHF energy ± 1e-8
- `test_h2o_s2_zero` — ⟨S²⟩ < 1e-10 for singlet
- `test_oh_converged`
- `test_oh_energy`
- `test_oh_s2_reasonable` — 0.75 < ⟨S²⟩ < 1.0
- `test_uhf_result_fields`

## Validation Script

File: `scripts/validate_uhf.py`

Output:
- `outputs/logs/phase16_uhf.json`
- `outputs/logs/phase16_uhf.txt`

## Acceptance Criteria

- `pytest tests/test_uhf.py -v` all pass
- `pytest tests/ -x` no regressions
- UHF H₂O singlet energy matches RHF within 1e-8 Ha
- OH doublet converged and within 1e-4 Ha of PySCF
- ⟨S²⟩ values correct
- Log committed
