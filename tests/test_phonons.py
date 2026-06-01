"""Tests for Phase 22 nuclear-repulsion-only phonons."""

import numpy as np
import pytest

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.periodic import Crystal, phonon_band_structure


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


def test_phonon_shape():
    result = phonon_band_structure(_h_chain(), STO3G, _h_special_points(), "G-X", n_points=12)
    assert result.frequencies.shape == (12, 3)
    assert result.qpoints.shape == (12, 3)


def test_phonon_gamma_acoustic():
    result = phonon_band_structure(_h_chain(), STO3G, _h_special_points(), "G-X", n_points=12)
    assert np.max(np.abs(result.frequencies[0])) < 1e-3


def test_phonon_3d_not_impl():
    a = 7.608
    special_points = {"G": np.zeros(3), "X": np.array([np.pi / a, 0.0, 0.0])}
    with pytest.raises(NotImplementedError, match="1D"):
        phonon_band_structure(_lih_crystal(), STO3G, special_points, "G-X")


def test_phonon_tick_labels():
    result = phonon_band_structure(_h_chain(), STO3G, _h_special_points(), "G-X", n_points=12)
    assert result.tick_labels == ["Γ", "X"]
    assert len(result.tick_positions) == 2
