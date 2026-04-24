"""Tests for atoms.py"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molekul.atoms import Atom
from molekul.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM


def test_atom_creation_bohr():
    a = Atom(symbol="H", coords=np.array([0.0, 0.0, 1.0]))
    assert a.symbol == "H"
    assert a.Z == 1
    np.testing.assert_array_equal(a.coords, [0.0, 0.0, 1.0])


def test_atom_from_angstrom():
    a = Atom.from_angstrom("O", 0.0, 0.0, 0.5)
    expected = np.array([0.0, 0.0, 0.5 * ANGSTROM_TO_BOHR])
    np.testing.assert_allclose(a.coords, expected)


def test_atom_roundtrip_angstrom():
    a = Atom.from_angstrom("C", 1.2, -0.5, 3.0)
    c_ang = a.coords_angstrom()
    np.testing.assert_allclose(c_ang, [1.2, -0.5, 3.0], atol=1e-10)


def test_atom_unknown_symbol():
    with pytest.raises(ValueError, match="Unknown element"):
        Atom(symbol="Xx", coords=np.array([0.0, 0.0, 0.0]))


def test_atom_bad_coords_shape():
    with pytest.raises(ValueError, match="shape"):
        Atom(symbol="H", coords=np.array([0.0, 0.0]))


def test_atom_z_values():
    assert Atom(symbol="H",  coords=np.zeros(3)).Z == 1
    assert Atom(symbol="He", coords=np.zeros(3)).Z == 2
    assert Atom(symbol="O",  coords=np.zeros(3)).Z == 8
    assert Atom(symbol="N",  coords=np.zeros(3)).Z == 7


def test_atom_repr():
    a = Atom.from_angstrom("H", 0.0, 0.0, 0.0)
    r = repr(a)
    assert "H" in r
    assert "Z=1" in r
