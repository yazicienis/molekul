"""Tests for EOM-CCSD excitation energies (Phase 15)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.cis import cis_excitations
from molekul.eom_ccsd import EOMCCSDResult, eom_ccsd_ee
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf


BOHR = 1.0 / 0.529177
H2_EOM_CCSD_STATE1 = 0.96893157
H2O_EOM_CCSD_STATE1 = 0.45711995


@pytest.fixture(scope="module")
def h2_system():
    mol = Molecule(
        atoms=[Atom("H", [0, 0, 0]), Atom("H", [0, 0, 0.74 * BOHR])],
        charge=0,
        multiplicity=1,
    )
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return mol, rhf


@pytest.fixture(scope="module")
def h2o_system():
    mol = Molecule(
        atoms=[
            Atom("O", [0, 0, 0]),
            Atom("H", [0, 0.757 * BOHR, 0.586 * BOHR]),
            Atom("H", [0, -0.757 * BOHR, 0.586 * BOHR]),
        ],
        charge=0,
        multiplicity=1,
    )
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return mol, rhf


@pytest.fixture(scope="module")
def h2_eom(h2_system):
    mol, rhf = h2_system
    return eom_ccsd_ee(mol, STO3G, rhf, n_states=2, verbose=False)


@pytest.fixture(scope="module")
def h2o_eom(h2o_system):
    mol, rhf = h2o_system
    return eom_ccsd_ee(mol, STO3G, rhf, n_states=5, verbose=False)


def test_h2_eom_converged(h2_eom):
    assert isinstance(h2_eom, EOMCCSDResult)
    assert h2_eom.n_states == 2
    assert h2_eom.excitation_energies.shape == (2,)
    assert h2_eom.excitation_eV.shape == (2,)
    assert h2_eom.r1.shape == (2, h2_eom.n_occ * 2, h2_eom.n_virt * 2)


def test_h2_eom_positive_energies(h2_eom):
    assert np.all(h2_eom.excitation_energies > 0.0)
    assert abs(h2_eom.excitation_energies[0] - H2_EOM_CCSD_STATE1) < 0.001


def test_h2o_eom_state1(h2o_eom):
    assert abs(h2o_eom.excitation_energies[0] - H2O_EOM_CCSD_STATE1) < 0.001


def test_eom_better_than_cis(h2o_system, h2o_eom):
    mol, rhf = h2o_system
    cis = cis_excitations(mol, STO3G, rhf, n_states=1, verbose=False)
    eom_err = abs(h2o_eom.excitation_energies[0] - H2O_EOM_CCSD_STATE1)
    cis_err = abs(cis.excitation_energies[0] - H2O_EOM_CCSD_STATE1)
    assert eom_err < cis_err


def test_h2o_eom_amplitude_shapes(h2o_eom):
    assert h2o_eom.r1.shape == (5, h2o_eom.n_occ * 2, h2o_eom.n_virt * 2)
    assert h2o_eom.r2.shape == (
        5,
        h2o_eom.n_occ * 2,
        h2o_eom.n_occ * 2,
        h2o_eom.n_virt * 2,
        h2o_eom.n_virt * 2,
    )
