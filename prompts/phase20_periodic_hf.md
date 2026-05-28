# Phase 20: Periodic Hartree-Fock

## Context

MOLEKUL implements molecular RHF in `src/molekul/rhf.py` using Gaussian-type
orbitals (GTOs). Periodic HF extends this to crystalline systems by applying
the Bloch theorem: each basis function φ_μ centered on atom A in the unit cell
generates a Bloch sum over all lattice translations R:

```
Φ_μk(r) = (1/√N) Σ_R e^{ik·R} φ_μ(r - A - R)
```

This gives a k-dependent generalised eigenvalue problem at each k-point:

```
H(k) C(k) = S(k) C(k) E(k)
```

where:

```
H_μν(k) = Σ_R e^{ik·R} ⟨φ_μ(r)|H|φ_ν(r-R)⟩
S_μν(k) = Σ_R e^{ik·R} ⟨φ_μ(r)|φ_ν(r-R)⟩
```

The GPU backend (`src/molekul/backend.py`) from Phase 19 is available.

Reference codes: CRYSTAL (Dovesi et al.), PySCF periodic module (`pyscf.pbc`).

## Objective

Implement periodic Hartree-Fock for 1D and 3D systems in a new file
`src/molekul/periodic_hf.py`.

Validated against PySCF `pyscf.pbc.scf.RHF` for:
- 1D H chain (equidistant, STO-3G, Γ-point and k-mesh)
- 3D LiH in rock-salt structure (STO-3G, Γ-point)

## Theory

### Lattice representation

A crystal is described by lattice vectors `a_1, a_2, a_3` (or just `a_1` for 1D).
Atom positions within the unit cell are fractional coordinates.

```python
@dataclass
class Crystal:
    lattice: np.ndarray        # shape (ndim, 3) — lattice vectors in Bohr
    atoms: list[Atom]          # fractional coordinates in the unit cell
    basis_fn: BasisSet
```

### Real-space lattice sum (cutoff)

Integrals over the infinite lattice are truncated at a real-space cutoff R_max:

```
H_μν(k) ≈ Σ_{|R|<R_max} e^{ik·R} ⟨φ_μ(r)|H|φ_ν(r-R)⟩
```

Use R_max = 4 × max(lattice parameter). This is implemented first.

### Ewald summation (nuclear attraction)

The nuclear attraction sum diverges without compensation. Use the standard
Ewald split:

```
V_ne = V_ne^{short}(real space, η) + V_ne^{long}(reciprocal space, η)
```

where η is the Ewald splitting parameter. Implement after the real-space
cutoff version is validated at Γ-point.

### k-point sampling

Monkhorst-Pack grid for 1D: k_n = n/N_k * (2π/a), n = 0, ..., N_k-1.

For 3D: k = (n1/N1)*b1 + (n2/N2)*b2 + (n3/N3)*b3 where b_i are reciprocal
lattice vectors.

### SCF loop

```
for iteration:
    H(k), S(k) for each k
    solve H(k) C(k) = S(k) C(k) E(k) at each k
    density matrix: P_μν = (1/N_k) Σ_k Σ_i^occ C_μi(k) C*_νi(k)
    build new Fock matrix F_μν(R) using P
    check convergence: max|P_new - P_old| < tol
```

Note: the density matrix P is real-space (not k-space), built by summing over
all k-points.

## Implementation

File: `src/molekul/periodic_hf.py` (new file)

```python
@dataclass
class Crystal:
    lattice: np.ndarray        # shape (ndim, 3), Bohr
    atoms: list[Atom]          # positions in Bohr (Cartesian)
    charge: int = 0

@dataclass
class PeriodicHFResult:
    energy_total: float        # energy per unit cell, Ha
    energy_hf: float
    band_energies: np.ndarray  # shape (n_kpts, n_basis), Ha
    kpoints: np.ndarray        # shape (n_kpts, 3)
    mo_coefficients_k: list    # list of C(k), each (n_basis, n_basis)
    density_matrix: np.ndarray # real-space, shape (n_basis, n_basis)
    converged: bool
    n_iter: int
    n_occ: int
    n_basis: int

def periodic_hf(
    crystal: Crystal,
    basis_fn: BasisSet,
    kpoints: np.ndarray,       # shape (n_kpts, 3), fractional coords
    *,
    r_max_factor: float = 4.0, # real-space cutoff = r_max_factor × max(|a_i|)
    use_ewald: bool = True,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    verbose: bool = False,
) -> PeriodicHFResult:
    ...

def monkhorst_pack(lattice: np.ndarray, mesh: tuple[int, ...]) -> np.ndarray:
    """Generate Monkhorst-Pack k-point grid, fractional coords."""
    ...
```

Internal helpers:
```python
def _lattice_sum_overlap(crystal, basis_fn, kpoints, R_vecs) -> np.ndarray
def _lattice_sum_hcore(crystal, basis_fn, kpoints, R_vecs, use_ewald) -> np.ndarray
def _lattice_sum_eri(crystal, basis_fn, R_vecs) -> np.ndarray  # real-space, no k
def _ewald_nuclear(crystal, eta) -> float
```

Use the GPU backend (`from .backend import get_xp`) for all array operations.

## Tests

File: `tests/test_periodic_hf.py`

### 1D H chain (STO-3G, bond length 1.8 Bohr = 0.952 Å)

PySCF reference (1D, k-mesh 4):
- Γ-point total energy per atom: TBD at runtime
- Band gap at Γ (HOMO-LUMO): TBD at runtime

Required tests:
- `test_h_chain_1d_gamma_energy` — Γ-point energy per atom within 1e-4 Ha of PySCF
- `test_h_chain_1d_kgrid_converges` — energy with 4-point k-mesh within 1e-3 Ha of PySCF
- `test_h_chain_band_structure` — band_energies shape is (n_kpts, n_basis)
- `test_monkhorst_pack_1d` — for mesh=(4,), returns 4 k-points in [0, 2π/a)

### 3D LiH rock-salt (STO-3G, a = 7.608 Bohr = 4.026 Å)

PySCF reference (Γ-point):
- Total energy per formula unit: TBD at runtime

Required tests:
- `test_lih_3d_gamma_energy` — Γ-point energy per f.u. within 1e-3 Ha of PySCF
- `test_monkhorst_pack_3d` — for mesh=(2,2,2), returns 8 k-points

## Validation Script

File: `scripts/validate_periodic_hf.py`

Output:
- `outputs/logs/phase20_periodic_hf.json`
- `outputs/logs/phase20_periodic_hf.txt`

Log: system, k-mesh, energy per cell (MOLEKUL vs PySCF), diff, convergence.

## Acceptance Criteria

- H chain Γ-point energy within 1e-4 Ha of PySCF `pyscf.pbc`
- LiH Γ-point energy within 1e-3 Ha of PySCF `pyscf.pbc`
- `pytest tests/ -x` no regressions (628 + new tests)
- Log committed
- Commit: Phase 20 files only
