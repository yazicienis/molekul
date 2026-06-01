"""Tests for Phase 23 full periodic-HF phonons."""

from functools import lru_cache

import numpy as np
import pytest

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.periodic import Crystal, phonon_band_structure, phonon_band_structure_full


def _h_chain():
    a = 1.8
    return Crystal(
        lattice=np.array([[a, 0.0, 0.0]]),
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
        name="H chain",
    )


def _h_special_points():
    a = 1.8
    return {"G": np.zeros(3), "X": np.array([np.pi / a, 0.0, 0.0])}


def _lih_crystal():
    a = 7.608
    return Crystal(
        lattice=np.eye(3) * a,
        atoms=[Atom("Li", [0.0, 0.0, 0.0]), Atom("H", [a / 2.0, a / 2.0, a / 2.0])],
        charge=0,
        multiplicity=1,
        name="LiH rock-salt",
    )


@lru_cache(maxsize=1)
def _full_result():
    return phonon_band_structure_full(_h_chain(), STO3G, _h_special_points(), "G-X", n_points=4)


def test_full_phonon_shape():
    result = _full_result()
    assert result.frequencies.shape == (4, 3)
    assert result.qpoints.shape == (4, 3)


def test_full_phonon_gamma_acoustic():
    result = _full_result()
    assert np.max(np.abs(result.frequencies[0])) < 1e-3


def test_full_phonon_3d_not_impl():
    a = 7.608
    special_points = {"G": np.zeros(3), "X": np.array([np.pi / a, 0.0, 0.0])}
    with pytest.raises(NotImplementedError, match="1D"):
        phonon_band_structure_full(_lih_crystal(), STO3G, special_points, "G-X")


def test_full_vs_nuclear_different():
    full = _full_result()
    nuclear = phonon_band_structure(_h_chain(), STO3G, _h_special_points(), "G-X", n_points=4)
    assert np.max(np.abs(full.frequencies[-1] - nuclear.frequencies[-1])) > 1e-3


def test_full_phonon_finite_freq_at_X():
    result = _full_result()
    assert np.max(result.frequencies[-1]) > 1e-3
