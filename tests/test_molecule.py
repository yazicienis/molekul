"""Tests for molecule.py"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molekul.atoms import Atom
from molekul.molecule import Molecule


def make_h2(bond_bohr=1.4):
    a1 = Atom(symbol="H", coords=np.array([0.0, 0.0, 0.0]))
    a2 = Atom(symbol="H", coords=np.array([0.0, 0.0, bond_bohr]))
    return Molecule(atoms=[a1, a2], charge=0, multiplicity=1, name="H2")


def make_h2o():
    from molekul.constants import ANGSTROM_TO_BOHR
    O  = Atom.from_angstrom("O",  0.000000,  0.000000,  0.117176)
    H1 = Atom.from_angstrom("H",  0.000000,  0.757306, -0.468704)
    H2 = Atom.from_angstrom("H",  0.000000, -0.757306, -0.468704)
    return Molecule(atoms=[O, H1, H2], charge=0, multiplicity=1, name="H2O")


class TestMoleculeBasic:
    def test_n_atoms(self):
        mol = make_h2()
        assert mol.n_atoms == 2

    def test_n_electrons_neutral(self):
        mol = make_h2()
        assert mol.n_electrons == 2

    def test_n_electrons_charged(self):
        # H2+ has 1 electron
        a1 = Atom(symbol="H", coords=np.zeros(3))
        a2 = Atom(symbol="H", coords=np.array([0.0, 0.0, 1.4]))
        mol = Molecule(atoms=[a1, a2], charge=1, multiplicity=2)
        assert mol.n_electrons == 1

    def test_n_alpha_beta_singlet(self):
        mol = make_h2()
        assert mol.n_alpha == 1
        assert mol.n_beta == 1

    def test_inconsistent_charge_multiplicity(self):
        a1 = Atom(symbol="H", coords=np.zeros(3))
        a2 = Atom(symbol="H", coords=np.array([0.0, 0.0, 1.4]))
        # H2 has 2 electrons; multiplicity=2 would need 1 unpaired -> 1+1 odd -> inconsistent
        with pytest.raises(ValueError):
            Molecule(atoms=[a1, a2], charge=0, multiplicity=2)

    def test_coords_bohr_shape(self):
        mol = make_h2()
        assert mol.coords_bohr.shape == (2, 3)

    def test_coords_angstrom_shape(self):
        mol = make_h2()
        assert mol.coords_angstrom.shape == (2, 3)


class TestNuclearRepulsion:
    def test_h2_nuclear_repulsion(self):
        # H2 at 1.4 Bohr: Z_A=Z_B=1, E_nuc = 1/1.4 = 0.714285... Hartree
        mol = make_h2(bond_bohr=1.4)
        e = mol.nuclear_repulsion_energy()
        np.testing.assert_allclose(e, 1.0 / 1.4, rtol=1e-10)

    def test_h2o_nuclear_repulsion_positive(self):
        mol = make_h2o()
        e = mol.nuclear_repulsion_energy()
        assert e > 0

    def test_h2o_nuclear_repulsion_value(self):
        # Known reference: H2O at experimental geometry ~ 8.9 Hartree
        mol = make_h2o()
        e = mol.nuclear_repulsion_energy()
        assert 8.0 < e < 10.0, f"Unexpected E_nuc={e:.4f}"

    def test_single_atom_zero_repulsion(self):
        mol = Molecule(atoms=[Atom(symbol="He", coords=np.zeros(3))], charge=0, multiplicity=1)
        assert mol.nuclear_repulsion_energy() == 0.0


class TestAtomicNumbers:
    def test_atomic_numbers_h2o(self):
        mol = make_h2o()
        assert mol.atomic_numbers == [8, 1, 1]
