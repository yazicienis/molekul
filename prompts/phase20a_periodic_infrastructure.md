# Phase 20a: Periodic Infrastructure (Crystal, Lattice, Bloch Sums)

## Context

This is the first of three phases building periodic Hartree-Fock. No SCF, no
Fock matrix, no ERI in this phase. The goal is to get the data model and
single-electron Bloch-sum integrals right before anything else is built on top.

The GPU backend from Phase 19 is available (`from .backend import get_xp`).

## Objective

Implement `src/molekul/periodic.py` with:
1. The `Crystal` dataclass (unambiguous coordinate conventions)
2. Reciprocal lattice and k-point utilities
3. Bloch-summed overlap S(k) and core Hamiltonian H(k) matrices

No SCF loop. No ERI. Validated by comparing S(k=0) and H(k=0) for a
H₂ dimer against the corresponding molecular integrals.

## Coordinate conventions (fixed, do not deviate)

| Quantity | Convention |
|----------|-----------|
| Lattice vectors `a_i` | Cartesian, Bohr |
| Atom positions | Cartesian, Bohr (same as molecular code) |
| Reciprocal vectors `b_i` | Cartesian, Bohr⁻¹: `b_i = 2π (a_j × a_k) / V` |
| k-points (internal) | Cartesian Bohr⁻¹ |
| k-points (Monkhorst-Pack input) | fractional `[0,1)` — converted to Cartesian internally |
| Phase factor | `e^{i k · R}` where R is a Cartesian lattice vector in Bohr |

## Theory

### Crystal dataclass

```python
@dataclass
class Crystal:
    lattice: np.ndarray   # shape (ndim, 3), Cartesian Bohr; ndim = 1, 2, or 3
    atoms: list[Atom]     # Cartesian Bohr (same convention as Molecule)
    charge: int = 0
    multiplicity: int = 1

    @property
    def n_electrons(self) -> int: ...

    @property
    def n_alpha(self) -> int: ...

    @property
    def reciprocal_lattice(self) -> np.ndarray:
        """b vectors, shape (ndim, 3), Cartesian Bohr⁻¹."""
        ...

    def lattice_vectors_in_shell(self, r_max: float) -> np.ndarray:
        """All R with |R| ≤ r_max, shape (N_R, 3), Cartesian Bohr."""
        ...
```

### Bloch sums

```
S_μν(k) = Σ_R e^{ik·R} ⟨φ_μ(r) | φ_ν(r − R)⟩
H_μν(k) = Σ_R e^{ik·R} ⟨φ_μ(r) | T + V_ne | φ_ν(r − R)⟩
```

The sum over R is truncated at `r_max = r_max_factor × max(|a_i|)`.
Default `r_max_factor = 4.0`.

For `R = 0` the integral is the standard molecular integral with the unit-cell
atoms. For `R ≠ 0` the bra function φ_μ is on atom A in the unit cell and the
ket function φ_ν is on atom B + R (atom B translated by lattice vector R).

**Important:** nuclear attraction V_ne in H(k) sums over all nuclei in all
translated cells within the cutoff. This is approximate (not Ewald) but
sufficient for Phase 20a/20b.

### k-point generation

```python
def monkhorst_pack(lattice: np.ndarray, mesh: tuple[int, ...]) -> np.ndarray:
    """
    Generate Monkhorst-Pack k-grid.

    Returns shape (prod(mesh), 3), Cartesian Bohr⁻¹.
    Fractional grid points: n_i / N_i for n_i in [0, N_i).
    Converted to Cartesian: k = Σ_i (n_i/N_i) * b_i.
    """
```

## Implementation

File: `src/molekul/periodic.py` (new file)

Public API for this phase:

```python
@dataclass
class Crystal: ...

def monkhorst_pack(lattice, mesh) -> np.ndarray: ...

def bloch_overlap(crystal, basis_fn, kpoints, r_max_factor=4.0) -> np.ndarray:
    """S(k) for each k. Returns shape (n_kpts, n_basis, n_basis), complex128."""

def bloch_hcore(crystal, basis_fn, kpoints, r_max_factor=4.0) -> np.ndarray:
    """H_core(k) for each k. Returns shape (n_kpts, n_basis, n_basis), complex128."""
```

Use `get_xp()` from `backend.py` for array operations.

Reuse molecular integral functions from `integrals.py` and `eri.py` with shifted
atom positions for the translated-cell contributions.

## Tests

File: `tests/test_periodic_infrastructure.py`

### Validation against molecular integrals

At k=0, S(k=0) for a crystal with a single H₂ molecule per unit cell (large
enough lattice so adjacent cells don't overlap) must equal the molecular S matrix.

```python
def test_bloch_overlap_gamma_matches_molecular():
    # H2 in a large box (a = 20 Bohr), k=Γ=(0,0,0)
    # max |S_periodic(Γ) - S_molecular| < 1e-10

def test_bloch_hcore_gamma_matches_molecular():
    # Same: H_core(Γ) must match molecular h_core within 1e-8

def test_monkhorst_pack_1d_shape():
    # 1D, mesh=(4,) → shape (4, 3)

def test_monkhorst_pack_3d_shape():
    # 3D, mesh=(2,2,2) → shape (8, 3)

def test_monkhorst_pack_gamma_included():
    # mesh=(1,1,1) → single k-point at Γ=(0,0,0)

def test_reciprocal_lattice_orthogonality():
    # a_i · b_j = 2π δ_ij for cubic lattice

def test_crystal_n_electrons():
    # H2 crystal: n_electrons = 2

def test_lattice_vectors_shell_count():
    # For 1D with a=2 Bohr, r_max=5 Bohr → should include R=0,±2,±4 (5 vectors)
```

## Validation Script

File: `scripts/validate_periodic_infrastructure.py`

Output:
- `outputs/logs/phase20a_periodic_infrastructure.json`
- `outputs/logs/phase20a_periodic_infrastructure.txt`

Log: H₂-in-box S(Γ) vs molecular S diff, H_core(Γ) vs molecular H_core diff,
Monkhorst-Pack k-points for 1D mesh=(4,) and 3D mesh=(2,2,2).

## Acceptance Criteria

- S(Γ) matches molecular S within 1e-10 for H₂ in large box
- H_core(Γ) matches molecular H_core within 1e-8 for H₂ in large box
- Monkhorst-Pack generates correct shapes and includes Γ for (1,1,1) mesh
- Reciprocal lattice satisfies `a_i · b_j = 2π δ_ij` for cubic cell
- `pytest tests/ -x` no regressions (628 + new tests)
- Commit: Phase 20a files only
