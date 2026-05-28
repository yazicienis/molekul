# Phase 18: Analytic RHF Gradient

## Context

RHF is in `src/molekul/rhf.py`. Numerical gradients (finite difference) are in
`src/molekul/grad.py` and used by the geometry optimizer in `src/molekul/optimizer.py`.
One-electron integrals are in `src/molekul/integrals.py`; ERIs in `src/molekul/eri.py`.
All integrals are over contracted Gaussian basis functions.

The numerical gradient step `h = 0.001` Bohr gives ~1e-6 Ha/Bohr accuracy.
Analytic gradients are exact and ~100× faster for large molecules.

## Objective

Implement analytic RHF nuclear gradients in `src/molekul/grad.py` (alongside the
existing numerical gradient). Replace the optimizer's finite-difference forces with
analytic forces.

## Theory

Szabo & Ostlund §3.4; Helgaker, Jørgensen & Olsen §10.2.

The analytic gradient of the RHF energy with respect to nuclear coordinate R_I is:

```
dE/dR_I = Σ_μν P_μν * dh_μν/dR_I
         + Σ_μνλσ (P_μν P_λσ - ½ P_μλ P_νσ) * d(μν|λσ)/dR_I
         - Σ_μν W_μν * dS_μν/dR_I
         + dV_nn/dR_I
```

where the energy-weighted density matrix is:

```
W_μν = Σ_i^occ ε_i C_μi C_νi
```

Integral derivatives for contracted GTOs:

For a primitive Gaussian g(r; α, A) = (r-A)^l exp(-α|r-A|²), its derivative
with respect to A_x is:

```
∂g/∂A_x = 2α g(r; α, A, l_x+1) - l_x g(r; α, A, l_x-1)
```

This recurrence connects derivative integrals to standard integrals with shifted
angular momenta. Implement derivative integrals for:
- dS_μν/dR_I (overlap derivative)
- dT_μν/dR_I (kinetic derivative)
- dV_μν/dR_I (nuclear attraction derivative)
- d(μν|λσ)/dR_I (ERI derivative)

Only s-type (l=0) and p-type (l=1) functions are needed for STO-3G, 6-31G*, cc-pVDZ.

## Implementation

File: `src/molekul/grad.py` (extend existing file)

```python
def rhf_gradient(
    mol: Molecule,
    basis_fn: BasisSet,
    rhf_result: RHFResult,
) -> np.ndarray:
    """
    Analytic RHF nuclear gradient.
    Returns shape (n_atoms, 3) in Hartree/Bohr.
    """
    ...
```

Internal helpers (add to `integrals.py` or `grad.py`):
```python
def overlap_derivative(basis, mol) -> np.ndarray:
    """dS_μν/dR_I, shape (n_atoms, 3, n_basis, n_basis)"""

def hcore_derivative(basis, mol) -> np.ndarray:
    """d(T+V)_μν/dR_I, shape (n_atoms, 3, n_basis, n_basis)"""

def eri_derivative(basis, mol) -> np.ndarray:
    """d(μν|λσ)/dR_I, shape (n_atoms, 3, n_basis, n_basis, n_basis, n_basis)"""
```

Also update `src/molekul/optimizer.py` to accept an optional `use_analytic=True`
flag that routes to `rhf_gradient` instead of finite differences.

## Tests

File: `tests/test_grad.py`

Validation strategy: analytic gradient must agree with numerical gradient to 1e-5 Ha/Bohr.

| Molecule | Max |analytic - numerical| | Tolerance |
|----------|--------------------------|-----------|
| H2 (r=0.74Å) | per component | 1e-5 |
| H2O (std geom) | per component | 1e-5 |
| CO (r=1.13Å) | per component | 1e-5 |

Required tests:
- `test_h2_gradient_vs_numerical`
- `test_h2o_gradient_vs_numerical`
- `test_co_gradient_vs_numerical`
- `test_gradient_shape` — shape is (n_atoms, 3)
- `test_gradient_translational_invariance` — sum of forces ≈ 0 (Newton 3rd law)

## Validation Script

File: `scripts/validate_grad.py`

Output:
- `outputs/logs/phase18_grad.json`
- `outputs/logs/phase18_grad.txt`

Log each molecule: analytic gradient, numerical gradient, max absolute difference.

## Acceptance Criteria

- All analytic–numerical differences < 1e-5 Ha/Bohr
- Translational sum of forces < 1e-10 Ha/Bohr
- `pytest tests/ -x` no regressions (628 + new tests)
- Commit: Phase 18 files only
