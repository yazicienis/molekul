# Phase 20b: Periodic HF — 1D H Chain (Γ-point + k-mesh)

## Context

Phase 20a implemented `Crystal`, `monkhorst_pack`, `bloch_overlap`, and
`bloch_hcore` in `src/molekul/periodic.py`. These are validated and correct.

This phase adds the SCF loop for periodic HF, validated on a 1D hydrogen chain
using STO-3G and PySCF `pyscf.pbc` as reference.

**No Ewald in this phase.** Nuclear attraction uses the real-space cutoff from
Phase 20a. Ewald summation is deferred to Phase 20c.

## Objective

Add `periodic_hf()` to `src/molekul/periodic.py`. Validate at Γ-point and
with a 4-point k-mesh for a 1D H chain.

## PySCF reference parameters (fix these exactly)

To get reproducible PySCF references:

```python
from pyscf.pbc import gto, scf

cell = gto.Cell()
cell.atom = "H 0 0 0"          # one H per unit cell
cell.a = [[1.8, 0, 0],         # lattice vector in Angstrom? No — see below
          [0, 20, 0],
          [0, 0, 20]]
cell.basis = "sto-3g"
cell.dimension = 1              # 1D periodic
cell.low_dim_ft_type = "inf_vacuum"
cell.verbose = 0
cell.build()

mf = scf.RHF(cell)
mf.kpts = cell.make_kpts([4, 1, 1])
mf.kernel()
```

**Unit convention:** PySCF `cell.a` is in Angstrom. The H chain bond length is
1.8 Bohr = 0.952 Å. The transverse cell dimensions (20 Bohr ≈ 10.58 Å) must be
large enough to decouple periodic images in y/z.

Document the exact PySCF version and these parameters in `outputs/logs/phase20b_*`.
Use PySCF runtime reference values (not hardcoded) in the validation script.

## Theory

### Periodic SCF loop

```
Initialize: P_μν = 0

for iteration:
    F_μν(R) = H_core_μν(R) + J_μν(R) - ½ K_μν(R)   [real-space Fock]
    F_μν(k) = Σ_R e^{ik·R} F_μν(R)                   [Bloch transform]
    solve F(k) C(k) = S(k) C(k) E(k) at each k
    P_μν = (1/N_k) Σ_k Σ_i^{occ} C_μi(k) C*_νi(k)   [density, complex sum → real P]
    check convergence: max |P_new - P_old| < conv_tol
```

The Coulomb and exchange terms in the Fock matrix use the periodic ERI:

```
(μν|λσ)_R = Σ_{R'} ∫∫ φ_μ(r) φ_ν(r−R) / |r−r'| φ_λ(r') φ_σ(r'−R') dr dr'
```

For a minimal basis 1D chain with real-space cutoff, these are computed by
displacing the bra/ket pairs across lattice vectors.

**Occupied orbitals:** for 1D H chain with 1 electron per cell, n_occ = 1 band.
The Fermi level at half-filling must be handled correctly.

### Real-space density matrix

P is real (because P = P* for real orbitals / time-reversal symmetry) and
corresponds to the unit-cell density. It has shape (n_basis, n_basis).

### Energy per unit cell

```
E/cell = (1/N_k) Σ_k Tr[P (H_core(k) + ½ F(k))] + E_nn/cell
```

where `E_nn/cell` is the nuclear repulsion energy per unit cell (computed with
Ewald or cutoff; use cutoff here for consistency with the Fock).

## Implementation

Add to `src/molekul/periodic.py`:

```python
@dataclass
class PeriodicHFResult:
    energy_per_cell: float       # Ha per unit cell
    band_energies: np.ndarray    # shape (n_kpts, n_basis)
    kpoints: np.ndarray          # shape (n_kpts, 3), Cartesian Bohr⁻¹
    mo_coefficients_k: list      # list of C(k), complex, each (n_basis, n_basis)
    density_matrix: np.ndarray   # real-space P, shape (n_basis, n_basis)
    converged: bool
    n_iter: int
    n_occ: int
    n_basis: int

def periodic_hf(
    crystal: Crystal,
    basis_fn: BasisSet,
    kpoints: np.ndarray,         # Cartesian Bohr⁻¹, shape (n_kpts, 3)
    *,
    r_max_factor: float = 4.0,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    verbose: bool = False,
) -> PeriodicHFResult:
```

## Tests

File: `tests/test_periodic_hf_1d.py`

Reference values are obtained at runtime from PySCF with the exact parameters
documented above. Hardcode fallback values in case PySCF is unavailable.

```python
def test_h_chain_gamma_energy():
    # Γ-point energy per cell within 1e-3 Ha of PySCF

def test_h_chain_kgrid_energy():
    # 4-point k-mesh energy per cell within 1e-3 Ha of PySCF

def test_h_chain_band_energies_shape():
    # band_energies.shape == (n_kpts, n_basis)

def test_h_chain_density_matrix_real():
    # max |Im(P)| < 1e-10

def test_h_chain_density_matrix_trace():
    # Tr[P S] ≈ n_electrons_per_cell (= 1 for H chain)

def test_h_chain_converged():
    # result.converged is True
```

## Validation Script

File: `scripts/validate_periodic_hf_1d.py`

Output:
- `outputs/logs/phase20b_periodic_hf_1d.json`
- `outputs/logs/phase20b_periodic_hf_1d.txt`

Log: PySCF parameters, Γ-point energy (MOLEKUL vs PySCF), k-mesh energy,
band energies at each k-point, convergence status.

## Acceptance Criteria

- H chain Γ-point energy per cell within 1e-3 Ha of PySCF `pyscf.pbc`
- H chain 4-point k-mesh energy within 1e-3 Ha of PySCF
- Density matrix is real, Tr[PS] = n_elec/cell
- `pytest tests/ -x` no regressions
- PySCF parameters fully documented in log
- Commit: Phase 20b files only
