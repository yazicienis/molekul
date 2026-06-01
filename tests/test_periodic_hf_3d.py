"""Tests for Phase 20c 3D periodic infrastructure: Ewald and Bloch matrices.

periodic_hf() supports 1D only; full 3D SCF requires Ewald-screened J/K
which is outside MOLEKUL's educational scope. This file tests the native
3D infrastructure: Ewald nuclear repulsion, Bloch S/H shapes, and k-mesh.
"""

import numpy as np
import pytest

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.periodic import Crystal, bloch_hcore, bloch_overlap, ewald_energy, monkhorst_pack, periodic_hf


def _lih_crystal():
    a = 7.608
    return Crystal(
        lattice=np.eye(3) * a,
        atoms=[Atom("Li", [0.0, 0.0, 0.0]), Atom("H", [a / 2.0, a / 2.0, a / 2.0])],
        charge=0,
        multiplicity=1,
        name="LiH rock-salt",
    )


def test_ewald_nn_positive():
    """Ewald nuclear repulsion must be positive for LiH."""
    assert ewald_energy(_lih_crystal()) > 0.0


def test_ewald_nn_value():
    """Ewald E_nn for LiH at a=7.608 Bohr — regression against Phase 20c log."""
    assert abs(ewald_energy(_lih_crystal()) - 0.181236255282) < 1e-6


def test_monkhorst_pack_3d_shape():
    """2×2×2 mesh → 8 k-points, each with 3 Cartesian components."""
    kpts = monkhorst_pack(_lih_crystal().lattice, (2, 2, 2))
    assert kpts.shape == (8, 3)


def test_bloch_overlap_3d_shape():
    """S(k) for LiH 2×2×2 mesh → shape (8, n_basis, n_basis), complex."""
    crystal = _lih_crystal()
    kpts = monkhorst_pack(crystal.lattice, (2, 2, 2))
    S = bloch_overlap(crystal, STO3G, kpts)
    n_basis = STO3G.n_basis(crystal)
    assert S.shape == (8, n_basis, n_basis)
    assert S.dtype == np.complex128


def test_bloch_hcore_3d_shape():
    """H_core(k) for LiH Γ → shape (1, n_basis, n_basis), complex."""
    crystal = _lih_crystal()
    kpts = monkhorst_pack(crystal.lattice, (1, 1, 1))
    H = bloch_hcore(crystal, STO3G, kpts)
    n_basis = STO3G.n_basis(crystal)
    assert H.shape == (1, n_basis, n_basis)


def test_periodic_hf_3d_not_implemented():
    """periodic_hf raises NotImplementedError for 3D crystals."""
    crystal = _lih_crystal()
    kpts = monkhorst_pack(crystal.lattice, (1, 1, 1))
    with pytest.raises(NotImplementedError, match="1D"):
        periodic_hf(crystal, STO3G, kpts)
