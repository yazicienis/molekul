"""Tests for the optional NumPy/CuPy backend."""

import numpy as np
import pytest

from molekul.atoms import Atom
from molekul.backend import get_xp, to_cpu, use_gpu
from molekul.basis_sto3g import STO3G
from molekul.ccsd import transform_mo_full
from molekul.eri import build_eri
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf


try:
    import cupy  # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def _h2():
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
        name="H2",
    )


def _h2o():
    return Molecule(
        atoms=[
            Atom.from_angstrom("O", 0.000, 0.000, 0.000),
            Atom.from_angstrom("H", 0.000, 0.757, -0.586),
            Atom.from_angstrom("H", 0.000, -0.757, -0.586),
        ],
        charge=0,
        multiplicity=1,
        name="H2O",
    )


def test_default_is_numpy():
    assert get_xp() is np


def test_cpu_context_manager():
    with pytest.warns(RuntimeWarning, match="CuPy not installed") if not CUPY_AVAILABLE else use_gpu():
        if not CUPY_AVAILABLE:
            with use_gpu():
                pass
    assert get_xp() is np


def test_to_cpu_noop():
    a = np.array([1.0, 2.0])
    assert np.array_equal(to_cpu(a), a)


def test_backend_restores_after_exception():
    try:
        ctx = use_gpu()
        if CUPY_AVAILABLE:
            with ctx:
                raise RuntimeError("test")
        else:
            with pytest.warns(RuntimeWarning, match="CuPy not installed"):
                with ctx:
                    raise RuntimeError("test")
    except RuntimeError:
        pass
    assert get_xp() is np


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_gpu_transform_mo_matches_cpu():
    mol = _h2()
    rhf = rhf_scf(mol, STO3G)
    eri = build_eri(STO3G, mol)
    cpu = transform_mo_full(eri, rhf.mo_coefficients)
    with use_gpu():
        gpu = transform_mo_full(eri, rhf.mo_coefficients)
    assert isinstance(gpu, np.ndarray)
    assert np.max(np.abs(cpu - gpu)) < 1e-12


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_gpu_rhf_energy_matches_cpu():
    mol = _h2o()
    cpu = rhf_scf(mol, STO3G)
    with use_gpu():
        gpu = rhf_scf(mol, STO3G)
    assert isinstance(gpu.density_matrix, np.ndarray)
    assert abs(cpu.energy_total - gpu.energy_total) < 1e-10
