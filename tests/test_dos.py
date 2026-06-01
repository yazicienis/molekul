"""Tests for Phase 22 Gaussian density of states."""

import numpy as np

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.periodic import Crystal, band_structure, dos


def _h_chain_band():
    a = 1.8
    crystal = Crystal(
        lattice=np.array([[a, 0.0, 0.0]]),
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
        name="H chain",
    )
    return band_structure(
        crystal,
        STO3G,
        {"G": np.zeros(3), "X": np.array([np.pi / a, 0.0, 0.0])},
        "G-X",
        n_points=20,
    )


def test_dos_shape():
    result = dos(_h_chain_band(), n_grid=300)
    assert result.energies.shape == (300,)
    assert result.dos.shape == (300,)


def test_dos_non_negative():
    result = dos(_h_chain_band())
    assert np.all(result.dos >= 0.0)


def test_dos_normalizable():
    result = dos(_h_chain_band())
    assert np.trapezoid(result.dos, result.energies) > 0.0


def test_dos_fermi_in_range():
    result = dos(_h_chain_band())
    assert result.energies[0] <= result.e_fermi <= result.energies[-1]


def test_dos_h_chain_peak():
    bands = _h_chain_band().band_energies
    result = dos(_h_chain_band())
    peak_energy = result.energies[int(np.argmax(result.dos))]
    assert np.min(bands) <= peak_energy <= np.max(bands)
