# Phase 19: Optional GPU Backend (CuPy)

## Context

MOLEKUL is pure NumPy. Compute-heavy operations (MO transformation, CCSD amplitude
equations, future periodic integrals) use `numpy.einsum`. The goal is to make these
einsum-heavy contractions optionally run on GPU via CuPy.

**Important constraints:**
- The ERI build (`build_eri`) uses Python loops over basis functions — these cannot
  be GPU-accelerated in this phase. Only numpy array operations and einsum
  contractions are targeted.
- NumPy must remain the default. All existing tests must pass unchanged on CPU.
- GPU is opt-in per call site via context manager, not a global process-level switch.

## Objective

Introduce `src/molekul/backend.py` with a context-manager interface and refactor
the three most einsum-heavy operations to use it:
1. `transform_mo_full()` in `ccsd.py` — the 4-index MO transformation (O(N^5))
2. CCSD amplitude einsum contractions in `_solve_ccsd_from_so_data()`
3. `_build_fock()` in `rhf.py`

This phase does **not** require every module to be GPU-aware. Only the three
operations above; others follow in later phases as needed.

## Implementation

### `src/molekul/backend.py` (new file)

```python
import contextlib
import numpy as _np
import warnings
from typing import Generator

_active = _np  # module-level default: always numpy


@contextlib.contextmanager
def use_gpu() -> Generator[None, None, None]:
    """Context manager: run array operations on GPU if CuPy is available."""
    global _active
    try:
        import cupy as cp
        _active = cp
        yield
    except ImportError:
        warnings.warn(
            "CuPy not installed — running on CPU.",
            RuntimeWarning, stacklevel=3,
        )
        yield
    finally:
        _active = _np


def get_xp():
    """Return the active array module (numpy or cupy)."""
    return _active


def to_cpu(arr):
    """Move array to CPU numpy. No-op for numpy arrays."""
    if _active is not _np:
        return _active.asnumpy(arr)
    return _np.asarray(arr)


def to_device(arr):
    """Move numpy array to active device. No-op when active is numpy."""
    return _active.asarray(arr)
```

Usage pattern at call sites:
```python
from .backend import get_xp, to_cpu, to_device

def transform_mo_full(eri_ao, C):
    xp = get_xp()
    eri = to_device(eri_ao)
    c = to_device(C)
    tmp = xp.einsum("pqrs,pi->iqrs", eri, c, optimize=True)
    # ... remaining contractions with xp ...
    return to_cpu(result)
```

**Boundary rule:** every public function that may return data to non-GPU callers
must call `to_cpu()` on its return values. Internal helpers within a GPU-aware
function may keep arrays on device.

### Modules to refactor (exactly these three)

1. **`src/molekul/ccsd.py`** — `transform_mo_full()` and the einsum calls in
   `_t1_residual_so()`, `_t2_residual_so()`, `_make_intermediates_so()`.

2. **`src/molekul/rhf.py`** — `_build_fock()`.

3. Keep all other modules (mp2, dft, tddft, eom_ccsd, uhf, …) unchanged.
   They benefit when they call `transform_mo_full` or `rhf_scf`, not by
   being refactored themselves.

## Tests

File: `tests/test_backend.py`

```python
import numpy as np
import pytest

cupy_available = False
try:
    import cupy  # noqa: F401
    cupy_available = True
except ImportError:
    pass

def test_default_is_numpy():
    from molekul.backend import get_xp
    assert get_xp() is np

def test_cpu_context_manager():
    from molekul import backend
    with backend.use_gpu():
        pass  # CuPy absent → falls back silently
    assert backend.get_xp() is np

def test_to_cpu_noop():
    from molekul.backend import to_cpu
    a = np.array([1.0, 2.0])
    assert np.array_equal(to_cpu(a), a)

def test_backend_restores_after_exception():
    from molekul import backend
    try:
        with backend.use_gpu():
            raise RuntimeError("test")
    except RuntimeError:
        pass
    assert backend.get_xp() is np

@pytest.mark.skipif(not cupy_available, reason="CuPy not installed")
def test_gpu_transform_mo_matches_cpu():
    # Build H2 ERI, run transform_mo_full on CPU and GPU, compare.
    # Max diff < 1e-12
    ...

@pytest.mark.skipif(not cupy_available, reason="CuPy not installed")
def test_gpu_rhf_energy_matches_cpu():
    # H2O RHF energy CPU vs GPU diff < 1e-10 Ha
    ...
```

Always-run (no GPU required):
- `test_default_is_numpy`
- `test_cpu_context_manager`
- `test_to_cpu_noop`
- `test_backend_restores_after_exception`

GPU tests (skipped if CuPy absent):
- `test_gpu_transform_mo_matches_cpu`
- `test_gpu_rhf_energy_matches_cpu`

## Validation Script

File: `scripts/validate_gpu_backend.py`

Output:
- `outputs/logs/phase19_gpu_backend.json`
- `outputs/logs/phase19_gpu_backend.txt`

Log: backend detected (numpy / cupy), H2O RHF energy CPU, H2O RHF energy GPU
(if available), diff, wall-clock time for `transform_mo_full` (CPU vs GPU).
If CuPy absent: record "GPU not available — CPU only".

## Acceptance Criteria

- All 628 + new tests pass on CPU (behaviour unchanged)
- GPU tests pass if CuPy available; skip gracefully if not
- H2O RHF energy CPU vs GPU diff < 1e-10 Ha (when GPU available)
- Backend restores to numpy after context exit, including on exception
- No CuPy arrays leak into non-GPU callers
- **No performance criterion.** Numerical equivalence + graceful fallback sufficient.
- Commit: Phase 19 files only
