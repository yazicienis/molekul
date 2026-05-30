# Phase 15: EOM-CCSD Excited States

## Context

CCSD is implemented in `src/molekul/ccsd.py`.
CIS is implemented in `src/molekul/cis.py` and provides a template for the
excited-state interface: `cis_states(mol, basis_fn, rhf_result) -> CISResult`.

EOM-CCSD provides coupled-cluster quality excited states. It should supersede
CIS for molecules where CCSD is affordable.

Available imports: `numpy`, all modules in `src/molekul/`.

## Objective

Implement EOM-CCSD-IP (ionisation potential) and EOM-CCSD-EE (excitation
energy) in a new file `src/molekul/eom_ccsd.py`.

Start with EOM-CCSD-EE only (neutral excited states). IP variant is optional
and can be Phase 15b if needed.

## Theory

Stanton & Bartlett (1993), J. Chem. Phys. 98, 7029.

EOM-CCSD-EE diagonalises the similarity-transformed Hamiltonian H̄ = e^{-T} H e^T
in the space of 1h1p and 2h2p excitations:

```
H̄ R_k |Φ_0⟩ = ω_k R_k |Φ_0⟩
```

where R_k = r_0 + Σ_{ia} r_i^a a†_a a_i + (1/4) Σ_{ijab} r_{ij}^{ab} ...

The matrix elements of H̄ in the {singles, doubles} basis define a
non-symmetric eigenvalue problem. Use `numpy.linalg.eig` (not `eigh`).

Use the spin-orbital formulation. The Ā intermediates from the CCSD
iteration can be reused.

Reference: Koch & Jørgensen (1990); Stanton & Bartlett (1993) Eq. 14–23.

## Implementation

File: `src/molekul/eom_ccsd.py` (new file)

```python
@dataclass
class EOMCCSDResult:
    excitation_energies: np.ndarray   # shape (n_states,), Ha
    excitation_eV: np.ndarray         # same in eV
    r1: np.ndarray                    # shape (n_states, n_occ*2, n_virt*2)
    r2: np.ndarray                    # shape (n_states, ...)
    n_states: int
    n_occ: int
    n_virt: int

def eom_ccsd_ee(
    mol: Molecule,
    basis_fn,
    rhf_result,
    n_states: int = 5,
    verbose: bool = False,
) -> EOMCCSDResult:
    ...
```

- Call `ccsd_energy()` to get converged T1, T2 amplitudes.
- Build the H̄ matrix in the singles+doubles space.
- Diagonalise with `np.linalg.eig`; take the `n_states` lowest positive
  real eigenvalues.
- Return only real parts (imaginary parts should be negligible; warn if > 1e-6).

## Tests

File: `tests/test_eom_ccsd.py`

PySCF reference: `pyscf.eom_ccsd.EOM_CCSD(ccsd_object).run()`, or use
`pyscf.tddft` for comparison.

| Molecule | Basis | State 1 (Ha) | Tolerance |
|----------|-------|-------------|-----------|
| H₂ r=0.74Å | STO-3G | TBD PySCF | 0.001 |
| H₂O std geom | STO-3G | TBD PySCF | 0.001 |

Required tests:
- `test_h2_eom_converged` — result contains expected fields
- `test_h2_eom_positive_energies` — all excitation energies > 0
- `test_h2o_eom_state1` — first excitation within 1 meV of PySCF EOM-CCSD
- `test_eom_better_than_cis` — EOM-CCSD state 1 energy closer to PySCF CCSD
  reference than CIS state 1 (for H₂O)

## Validation Script

File: `scripts/validate_eom_ccsd.py`

Output:
- `outputs/logs/phase15_eom_ccsd.json`
- `outputs/logs/phase15_eom_ccsd.txt`

## Acceptance Criteria

- `pytest tests/test_eom_ccsd.py -v` all pass
- `pytest tests/ -x` no regressions
- H₂O state 1 within 0.001 Ha of PySCF EOM-CCSD
- Log committed
