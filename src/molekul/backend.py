"""Optional array backend selection for NumPy/CuPy operations."""

from __future__ import annotations

import contextlib
from typing import Generator
import warnings

import numpy as _np


_active = _np


@contextlib.contextmanager
def use_gpu() -> Generator[None, None, None]:
    """Run backend-aware array operations on CuPy when it is available."""
    global _active
    previous = _active
    try:
        import cupy as cp
    except ImportError:
        warnings.warn(
            "CuPy not installed; running on CPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        try:
            yield
        finally:
            _active = previous
    else:
        _active = cp
        try:
            yield
        finally:
            _active = previous


def get_xp():
    """Return the active array module."""
    return _active


def to_cpu(arr):
    """Return a NumPy array, copying from GPU when CuPy is active."""
    if _active is not _np:
        return _active.asnumpy(arr)
    return _np.asarray(arr)


def to_device(arr):
    """Return an array on the active backend."""
    return _active.asarray(arr)
