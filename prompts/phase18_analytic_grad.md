# Phase 18: Analytic RHF Gradient

## Context

RHF is in `src/molekul/rhf.py`. Numerical gradients (finite difference, h=0.001 Bohr)
are in `src/molekul/grad.py` and used by the geometry optimizer in
`src/molekul/optimizer.py`. One-electron integrals are in `src/molekul/integrals.py`;
ERIs in `src/molekul/eri.py`.

**Scope:** STO-3G basis only in this phase. STO-3G uses only s-type and p-type
Gaussian primitives, so derivative recurrence relations stay within s/p angular
momentum. Support for d-type functions (6-31G*, cc-pVDZ) is deferred to a later
sub-phase.

## Objective

Implement analytic RHF nuclear gradients in `src/molekul/grad.py`. Validate against
the existing numerical gradient. Update the optimizer to accept analytic forces.

## Theory

Szabo & Ostlund §3.4; Helgaker, Jørgensen & Olsen §10.2.

### Gradient formula

```
dE/dR_I = Σ_μν P_μν * dh_μν/dR_I
         + Σ_μνλσ Γ_μνλσ * d(μν|λσ)/dR_I
         - Σ_μν W_μν * dS_μν/dR_I
         + dV_nn/dR_I
```

### Density matrices

RHF density: `P_μν = 2 Σ_i^occ C_μi C_νi`  (factor 2 for closed-shell spin sum)

Two-particle density: `Γ_μνλσ = P_μν P_λσ - ½ P_μλ P_νσ`

Energy-weighted density (not spin-doubled):
```
W_μν = 2 Σ_i^occ ε_i C_μi C_νi
```
Note: W uses the same occupied C as P but weighted by orbital energies, NOT by 2ε.
The factor 2 in P accounts for spin; W should also carry a factor 2 for consistency
with the spin-summed P convention. Verify: for H2/STO-3G, the Pulay term
`-Σ W_μν dS_μν/dR` should equal approximately -0.4 Ha/Bohr at equilibrium.

### Integral derivatives for s/p GTOs

For a primitive Gaussian g(r; α, A, l) = (r-A)^l exp(-α|r-A|²), derivative
w.r.t. A_x:

```
∂g/∂A_x = 2α g(r; α, A, l_x+1) - l_x g(r; α, A, l_x-1)
```

This shifts angular momentum by ±1. For s→p and p→d only d-type integrals appear
as intermediates; those can be computed using the same integral engine with l=2.

## Implementation

File: `src/molekul/grad.py` (extend existing file)

### Public API

```python
def rhf_gradient(
    mol: Molecule,
    basis_fn: BasisSet,
    rhf_result: RHFResult,
) -> np.ndarray:
    """
    Analytic RHF nuclear gradient (STO-3G s/p only).
    Returns shape (n_atoms, 3), Hartree/Bohr.
    """
```

### Internal helpers (can live in grad.py or a new integrals_deriv.py)

```python
def overlap_derivative(basis, mol) -> np.ndarray:
    """dS_μν/dR_{Ix}, shape (n_atoms, 3, n_basis, n_basis)"""

def hcore_derivative(basis, mol) -> np.ndarray:
    """d(T+V)_μν/dR_{Ix}, shape (n_atoms, 3, n_basis, n_basis)"""

def eri_derivative(basis, mol) -> np.ndarray:
    """d(μν|λσ)/dR_{Ix}, shape (n_atoms, 3, n_basis, n_basis, n_basis, n_basis)"""
```

### Optimizer update

Add `use_analytic: bool = True` to `optimizer.py`. Default to analytic only after
gradient tests pass; use `use_analytic=False` as fallback to numerical.

## Tests

File: `tests/test_grad.py`

### Level 1 — Individual integral derivatives (most important)

Each integral derivative must be validated against finite differences **independently**
before the total gradient test runs. This catches sign errors and prefactor mistakes
early.

```python
def test_overlap_derivative_h2():
    # Compute dS/dR numerically (finite diff on build_overlap) and analytically.
    # Max |analytic - numerical| < 1e-7

def test_hcore_derivative_h2():
    # Same for dh_core/dR.
    # Max |analytic - numerical| < 1e-7

def test_eri_derivative_h2():
    # Same for d(μν|λσ)/dR.
    # Max |analytic - numerical| < 1e-7
```

### Level 2 — Total gradient

| Molecule | Max |analytic - numerical| | Tolerance |
|----------|--------------------------|-----------|
| H2 (r=0.74 Å, STO-3G) | per component | 1e-5 |
| H2O (O at origin, std geom, STO-3G) | per component | 1e-5 |
| CO (r=1.128 Å, STO-3G) | per component | 1e-5 |

Required tests:
- `test_overlap_derivative_h2`
- `test_hcore_derivative_h2`
- `test_eri_derivative_h2`
- `test_h2_gradient_vs_numerical`
- `test_h2o_gradient_vs_numerical`
- `test_co_gradient_vs_numerical`
- `test_gradient_shape` — shape is (n_atoms, 3)
- `test_gradient_translational_invariance` — Σ_I dE/dR_I ≈ 0 per component, tol 1e-10

## Validation Script

File: `scripts/validate_grad.py`

Output:
- `outputs/logs/phase18_grad.json`
- `outputs/logs/phase18_grad.txt`

Log each molecule: analytic gradient (full tensor), max |analytic−numerical|,
translational residual.

## Acceptance Criteria

- Level 1 integral derivative tests pass (max diff < 1e-7)
- Total gradient: max |analytic − numerical| < 1e-5 Ha/Bohr for all molecules
- Translational sum of forces < 1e-10 Ha/Bohr
- `pytest tests/ -x` no regressions (628 + new tests)
- Commit: Phase 18 files only
- **Note:** d-function derivative support (for 6-31G*, cc-pVDZ) is explicitly
  out of scope for this phase.
