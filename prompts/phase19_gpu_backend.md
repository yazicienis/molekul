# Phase 19: Optional GPU Backend (CuPy)

## Context

MOLEKUL is pure NumPy. All compute-heavy operations (ERI build, MO transformation,
CCSD amplitude equations, future periodic integrals) use `numpy.einsum`. The goal
is to make these operations optionally run on GPU via CuPy, which provides a
NumPy-compatible API over CUDA.

Design constraint: NumPy must remain the default. GPU is opt-in. All existing tests
must pass unchanged on CPU. GPU tests are skipped if CuPy is not installed.

## Objective

Introduce an array-module backend abstraction (`src/molekul/backend.py`) and
refactor the four most compute-heavy modules to use it:
- `src/molekul/eri.py` (ERI build — O(N^4))
- `src/molekul/ccsd.py` (MO transformation + amplitude einsum — O(N^6))
- `src/molekul/rhf.py` (Fock build — O(N^4))
- `src/molekul/uhf.py` (Fock build — O(N^4))

Downstream modules (mp2, dft, tddft, eom_ccsd, etc.) benefit automatically once
their dependencies use the backend.

## Implementation

### `src/molekul/backend.py` (new file)

```python
import numpy as _np
import warnings

_cupy = None

def set_backend(use_gpu: bool = False):
    """Select NumPy (default) or CuPy array module."""
    global _cupy
    if use_gpu:
        try:
            import cupy as cp
            _cupy = cp
        except ImportError:
            warnings.warn(
                "CuPy not installed — falling back to NumPy.",
                RuntimeWarning, stacklevel=2,
            )
            _cupy = None
    else:
        _cupy = None

def get_xp():
    """Return the active array module (numpy or cupy)."""
    return _cupy if _cupy is not None else _np

def to_cpu(arr):
    """Move array to CPU (no-op for NumPy arrays)."""
    xp = get_xp()
    if xp is not _np:
        return xp.asnumpy(arr)
    return arr

def to_device(arr):
    """Move array to active device."""
    xp = get_xp()
    if xp is not _np:
        return xp.asarray(arr)
    return arr
```

### Refactoring pattern

Replace `import numpy as np` with:
```python
from .backend import get_xp
# inside functions:
xp = get_xp()
result = xp.einsum("ijkl,kl->ij", eri, P)
```

All intermediate arrays created inside a function should use `xp`. Arrays that
must be returned to callers (and may be stored as Python floats or passed to
non-GPU code) should be converted with `to_cpu()` at the function boundary.

### Modules to refactor

1. **`src/molekul/eri.py`** — `build_eri()`: the primitive integral loops are
   pure Python (cannot be GPU-accelerated directly); but the contraction step
   `np.einsum("pqrs,pi,qj,rk,sl->ijkl", ...)` in `transform_mo_full` can use `xp`.

2. **`src/molekul/ccsd.py`** — `transform_mo_full()` and all einsum calls inside
   `_solve_ccsd_from_so_data()`.

3. **`src/molekul/rhf.py`** — `_build_fock()` einsum.

4. **`src/molekul/uhf.py`** — `_build_uhf_focks()` einsum.

Keep all public function signatures identical. The `use_gpu` flag is a
module-level switch via `set_backend(use_gpu=True)`, not per-function.

## Tests

File: `tests/test_backend.py`

```python
def test_default_backend_is_numpy():
    from molekul.backend import get_xp
    import numpy as np
    assert get_xp() is np

def test_set_backend_cpu():
    from molekul import backend
    backend.set_backend(use_gpu=False)
    import numpy as np
    assert backend.get_xp() is np

@pytest.mark.skipif(not cupy_available, reason="CuPy not installed")
def test_gpu_eri_matches_cpu():
    # Run build_eri on H2, compare CPU vs GPU result
    ...

@pytest.mark.skipif(not cupy_available, reason="CuPy not installed")
def test_gpu_rhf_matches_cpu():
    # Run rhf_scf on H2O with use_gpu=True, compare energy
    ...
```

Required CPU-only tests (always run):
- `test_default_backend_is_numpy`
- `test_set_backend_cpu`
- `test_backend_to_cpu_noop_for_numpy`

GPU tests (skipped if CuPy absent):
- `test_gpu_eri_matches_cpu` — H2 ERI max diff < 1e-12
- `test_gpu_rhf_matches_cpu` — H2O RHF energy diff < 1e-10

## Validation Script

File: `scripts/validate_gpu_backend.py`

Output:
- `outputs/logs/phase19_gpu_backend.json`
- `outputs/logs/phase19_gpu_backend.txt`

Log: backend type, H2O RHF energy (CPU), H2O RHF energy (GPU if available),
diff, wall-clock time (CPU vs GPU). If CuPy absent, log "GPU not available".

## Acceptance Criteria

- All 628 + new tests pass on CPU (unchanged behaviour)
- GPU tests pass if CuPy is installed; skip gracefully if not
- H2O RHF energy CPU vs GPU diff < 1e-10 Ha
- No changes to public API signatures
- Commit: Phase 19 files only
