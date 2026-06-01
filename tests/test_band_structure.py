"""Tests for Phase 21 native tight-binding band structures."""

import numpy as np

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.periodic import Crystal, band_structure


def _h_chain():
    return Crystal(
        lattice=np.array([[1.8, 0.0, 0.0]]),
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
        name="H chain",
    )


def _lih_crystal():
    a = 7.608
    return Crystal(
        lattice=np.eye(3) * a,
        atoms=[Atom("Li", [0.0, 0.0, 0.0]), Atom("H", [a / 2.0, a / 2.0, a / 2.0])],
        charge=0,
        multiplicity=1,
        name="LiH rock-salt",
    )


def _h_special_points():
    a = 1.8
    return {"G": np.zeros(3), "X": np.array([np.pi / a, 0.0, 0.0])}


def _lih_special_points():
    a = 7.608
    return {
        "G": np.zeros(3),
        "X": np.array([np.pi / a, 0.0, 0.0]),
        "M": np.array([np.pi / a, np.pi / a, 0.0]),
    }


def test_h_chain_band_1d_shape():
    result = band_structure(_h_chain(), STO3G, _h_special_points(), "G-X", n_points=20)
    assert result.band_energies.shape == (20, 1)


def test_h_chain_band_dispersion():
    result = band_structure(_h_chain(), STO3G, _h_special_points(), "G-X", n_points=20)
    band = result.band_energies[:, 0]
    diffs = np.diff(band)
    assert np.all(diffs >= -1e-10) or np.all(diffs <= 1e-10)
    assert np.ptp(band) > 1e-3


def test_h_chain_band_gamma_lt_x():
    result = band_structure(_h_chain(), STO3G, _h_special_points(), "G-X", n_points=20)
    assert result.band_energies[0, 0] < result.band_energies[-1, 0]


def test_lih_band_3d_shape():
    result = band_structure(_lih_crystal(), STO3G, _lih_special_points(), "G-X-M-G", n_points=50)
    assert result.band_energies.shape == (150, 6)


def test_lih_n_occ():
    result = band_structure(_lih_crystal(), STO3G, _lih_special_points(), "G-X-M-G", n_points=10)
    assert result.n_occ == 2


def test_band_structure_tick_labels():
    result = band_structure(_lih_crystal(), STO3G, _lih_special_points(), "G-X-M-G", n_points=10)
    assert result.tick_labels == ["Γ", "X", "M", "Γ"]
    assert len(result.tick_positions) == 4


def test_band_structure_x_coords_monotone():
    result = band_structure(_lih_crystal(), STO3G, _lih_special_points(), "G-X-M-G", n_points=10)
    assert np.all(np.diff(result.x_coords) >= -1e-12)
