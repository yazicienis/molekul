# Phase 20c: Periodic HF — 3D + Ewald + k-mesh (LiH)

## Context

Phase 20b implemented and validated 1D periodic HF with real-space cutoff.
The `periodic_hf()` function in `src/molekul/periodic.py` is working for 1D.

This phase extends it to 3D systems and adds Ewald summation for the long-range
Coulomb divergence.

## Objective

1. Extend `periodic_hf()` to 3D.
2. Implement Ewald summation for the nuclear-nuclear, electron-nuclear, and
   electron-electron long-range Coulomb contributions.
3. Validate on LiH in rock-salt structure with Γ-point and 2×2×2 k-mesh.

## Why Ewald is needed in 3D

In 3D, the Coulomb sum `Σ_R 1/|r−R|` diverges conditionally. Real-space
truncation gives incorrect energies for 3D ionic systems. The Ewald method
splits the interaction into short-range (real space) and long-range (reciprocal
space) parts using a Gaussian screening function with splitting parameter η:

```
V_Ewald = V_short(η) + V_long(η)   [result independent of η]

V_short = Σ_{|R|<R_max} erfc(η |r+R|) / |r+R|       (real space)
V_long  = (4π/V) Σ_{0<|G|<G_max} exp(-G²/4η²)/G² exp(iG·r)  (reciprocal)
```

Reference: Ashcroft & Mermin App. B; Tosi (1964) Sol. State Phys. 16, 1.

### What to Ewald-sum

Three contributions need consistent Ewald treatment:
1. **Nuclear-nuclear** `E_nn`: standard ionic Ewald sum.
2. **Electron-nuclear** `V_ne` in H_core: Ewald potential of the nuclei at
   electron position r.
3. **Electron-electron** `J+K`: for a minimal-basis periodic code, the periodic
   Coulomb integrals are built in real space with erfc screening.

A practical simplification: compute the Ewald correction as a difference term
added to the real-space-cutoff result from Phase 20b:
`E_Ewald = E_cutoff + ΔE_Ewald`
where `ΔE_Ewald` is the long-range G-space correction. This keeps the code
modular.

### Ewald parameter η

Choose η = sqrt(π) / a_min where a_min = min(|a_i|). Document in SCIENCE.md.

## Implementation

### Add to `src/molekul/periodic.py`

```python
def ewald_energy(crystal: Crystal, eta: float | None = None) -> float:
    """
    Nuclear-nuclear Ewald energy per unit cell.
    eta: splitting parameter (Bohr⁻¹). Default: sqrt(π)/min(|a_i|).
    """

def ewald_hcore(crystal: Crystal, basis_fn: BasisSet,
                kpoints: np.ndarray,
                eta: float | None = None) -> np.ndarray:
    """
    H_core(k) with Ewald electron-nuclear potential.
    Shape (n_kpts, n_basis, n_basis), complex128.
    Replaces bloch_hcore for 3D systems.
    """
```

Extend `periodic_hf()` with:
```python
def periodic_hf(
    crystal: Crystal,
    basis_fn: BasisSet,
    kpoints: np.ndarray,
    *,
    r_max_factor: float = 4.0,
    use_ewald: bool = True,      # NEW: default True for 3D, ignored for 1D
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    verbose: bool = False,
) -> PeriodicHFResult:
```

For `crystal.lattice.shape[0] == 1` (1D), `use_ewald` is ignored.

## LiH rock-salt structure

```
Space group: Fm-3m (rock-salt)
Lattice parameter: a = 7.608 Bohr = 4.026 Å
Basis: Li at (0,0,0), H at (a/2, a/2, a/2) in Cartesian Bohr
Basis set: STO-3G
```

PySCF reference:
```python
from pyscf.pbc import gto, scf

cell = gto.Cell()
cell.atom = """
Li  0.0000  0.0000  0.0000
H   2.0130  2.0130  2.0130
"""   # Angstrom
cell.a = [[4.026, 0, 0],
          [0, 4.026, 0],
          [0, 0, 4.026]]
cell.basis = "sto-3g"
cell.verbose = 0
cell.build()

mf = scf.RHF(cell)
mf.kpts = cell.make_kpts([2, 2, 2])
mf.kernel()
```

## Tests

File: `tests/test_periodic_hf_3d.py`

```python
def test_lih_gamma_energy():
    # Γ-point energy per f.u. within 1e-2 Ha of PySCF

def test_lih_kgrid_energy():
    # 2×2×2 k-mesh energy per f.u. within 1e-2 Ha of PySCF

def test_ewald_nn_positive():
    # Nuclear-nuclear Ewald energy for LiH is positive (ions repel)

def test_monkhorst_pack_3d():
    # 3D cubic, mesh=(2,2,2) → 8 k-points

def test_lih_converged():
    assert result.converged

def test_lih_density_real():
    # max |Im(P)| < 1e-10
```

Note: LiH tolerance is 1e-2 Ha, looser than 1D H chain. This reflects that
STO-3G + minimal Ewald is approximate for ionic systems.

## SCIENCE.md entry

Add:
```
## Ewald splitting parameter η
- Value: sqrt(π) / min(|a_i|)
- File: src/molekul/periodic.py
- Justification: Balances real-space and reciprocal-space convergence.
  Standard choice (Tosi 1964). At this value both sums converge with
  similar truncation errors.
- Citation: Tosi (1964) Sol. State Phys. 16, 1; Ashcroft & Mermin App. B
```

## Validation Script

File: `scripts/validate_periodic_hf_3d.py`

Output:
- `outputs/logs/phase20c_periodic_hf_3d.json`
- `outputs/logs/phase20c_periodic_hf_3d.txt`

Log: LiH Γ-point energy, 2×2×2 energy, E_nn Ewald, PySCF reference, diffs.

## Acceptance Criteria

- LiH Γ-point energy within 1e-2 Ha of PySCF `pyscf.pbc`
- 2×2×2 k-mesh energy within 1e-2 Ha of PySCF
- Ewald E_nn positive for LiH
- 1D H chain tests from Phase 20b still pass (no regression)
- SCIENCE.md: Ewald η entry with Tosi (1964) citation
- `pytest tests/ -x` no regressions
- Commit: Phase 20c files only
