# Phase 17: TD-DFT (Casida Linear Response)

## Context

KS-DFT is implemented in `src/molekul/dft.py`:
- `ks_dft(mol, basis_fn, functional='lda', verbose=False) -> KSDFTResult`
- `KSDFTResult`: fields `energy_total, C, eps, P, converged, ...`

CIS is implemented in `src/molekul/cis.py` and provides an excitation-energy
interface; its matrix-build pattern is a template for the Casida A/B matrices.

TD-DFT (Casida) gives excitation energies with DFT-quality ground states,
significantly better than CIS for charge-transfer states and valence excitations.

## Objective

Implement the Casida linear-response TD-DFT equations for singlet excited
states in a new file `src/molekul/tddft.py`.

Support LDA and PBE functionals (same as `dft.py`). Restrict to
Tamm–Dancoff Approximation (TDA, B=0) first; full Casida (A-B)(A+B) is
optional if TDA works.

## Theory

Casida (1995), in Recent Advances in DFT.
Burke et al. (2005), J. Chem. Phys. 123, 062206.

**Tamm–Dancoff Approximation:**
```
A_{ia,jb} = δ_{ij} δ_{ab} (ε_a - ε_i) + (ia|f_xc|jb) + (ia|jb)
```
where:
- `(ia|jb)` = Coulomb integral in MO basis
- `(ia|f_xc|jb)` = exchange-correlation kernel integral

For LDA: `f_xc(r) = d²e_xc/dρ²` evaluated on the ground-state density.
For PBE: includes gradient correction to kernel (harder; do LDA first).

Diagonalise A to get excitation energies ω_k and transition vectors X_k.

Oscillator strengths:
```
f_k = (2/3) ω_k |⟨0|r̂|k⟩|²
```

## Implementation

File: `src/molekul/tddft.py` (new file)

```python
@dataclass
class TDDFTResult:
    excitation_energies: np.ndarray   # Ha, shape (n_states,)
    excitation_eV: np.ndarray
    oscillator_strengths: np.ndarray
    X: np.ndarray                     # transition vectors, shape (n_states, n_occ, n_virt)
    n_states: int
    functional: str
    n_occ: int
    n_virt: int

def tddft_tda(
    mol: Molecule,
    basis_fn,
    functional: str = 'lda',
    n_states: int = 5,
    verbose: bool = False,
) -> TDDFTResult:
    ...
```

- Call `ks_dft()` to get ground-state KS orbitals and density.
- Build the Casida A matrix in the MO basis using numerical integration
  (reuse `dft.py`'s Becke grid).
- Diagonalise with `np.linalg.eigh` (A is symmetric in TDA).
- Return lowest `n_states` positive roots.

## Tests

File: `tests/test_tddft.py`

PySCF reference: `pyscf.tddft.TDA(ks).run()`.

| Molecule | Basis | Functional | State 1 (Ha) | Tolerance |
|----------|-------|-----------|-------------|-----------|
| H₂ r=0.74Å | STO-3G | LDA | TBD PySCF | 0.01 |
| H₂O std geom | STO-3G | LDA | TBD PySCF | 0.01 |

Required tests:
- `test_h2_tda_converged`
- `test_h2_tda_positive_energies`
- `test_h2o_tda_state1_lda` — within 0.01 Ha of PySCF TDA-LDA
- `test_oscillator_strengths_positive`
- `test_tddft_better_than_cis` — for H₂O, TDDFT state 1 closer to EOM-CCSD than CIS

## Validation Script

File: `scripts/validate_tddft.py`

Output:
- `outputs/logs/phase17_tddft.json`
- `outputs/logs/phase17_tddft.txt`

## Acceptance Criteria

- `pytest tests/test_tddft.py -v` all pass
- `pytest tests/ -x` no regressions
- H₂O LDA TDA state 1 within 0.01 Ha of PySCF
- Log committed
