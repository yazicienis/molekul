"""Tests for CIS excited states (Phase 13).

All reference values from PySCF TDA at identical geometries (STO-3G).
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf import rhf_scf
from molekul.cis import cis_excitations, CISResult

BOHR = 1.0 / 0.529177


@pytest.fixture(scope="module")
def h2_cis():
    mol = Molecule(atoms=[Atom("H", [0, 0, 0]), Atom("H", [0, 0, 0.74 * BOHR])],
                   charge=0, multiplicity=1)
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return cis_excitations(mol, STO3G, rhf, n_states=1, verbose=False)


@pytest.fixture(scope="module")
def h2o_cis():
    mol = Molecule(atoms=[
        Atom("O", [0, 0, 0]),
        Atom("H", [0,  0.757 * BOHR, 0.586 * BOHR]),
        Atom("H", [0, -0.757 * BOHR, 0.586 * BOHR]),
    ], charge=0, multiplicity=1)
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return cis_excitations(mol, STO3G, rhf, n_states=5, verbose=False)


# ---------- H2 ----------

def test_h2_cis_type(h2_cis):
    assert isinstance(h2_cis, CISResult)

def test_h2_cis_state1_energy(h2_cis):
    # PySCF: 0.948407 Ha
    assert abs(h2_cis.excitation_energies[0] - 0.948407) < 1e-5

def test_h2_cis_positive_energies(h2_cis):
    assert np.all(h2_cis.excitation_energies > 0)


# ---------- H2O ----------

def test_h2o_cis_n_states(h2o_cis):
    assert h2o_cis.n_states == 5

def test_h2o_cis_energies(h2o_cis):
    refs = [0.485074, 0.557134, 0.616592, 0.705487, 0.811445]
    for k, ref in enumerate(refs):
        assert abs(h2o_cis.excitation_energies[k] - ref) < 1e-5, \
            f"State {k+1}: {h2o_cis.excitation_energies[k]:.6f} vs {ref:.6f}"

def test_h2o_cis_ev(h2o_cis):
    # State 5 should be ~22.08 eV
    assert abs(h2o_cis.excitation_energies_ev[4] - 22.0806) < 0.01

def test_h2o_cis_oscillator_strengths(h2o_cis):
    refs_f = [0.003548, 0.000000, 0.077169, 0.060090, 1.167016]
    for k, ref in enumerate(refs_f):
        assert abs(h2o_cis.oscillator_strengths[k] - ref) < 1e-4, \
            f"f state {k+1}: {h2o_cis.oscillator_strengths[k]:.6f} vs {ref:.6f}"

def test_h2o_cis_forbidden_state(h2o_cis):
    # State 2 is symmetry-forbidden: f=0
    assert h2o_cis.oscillator_strengths[1] < 1e-10

def test_h2o_cis_ascending(h2o_cis):
    e = h2o_cis.excitation_energies
    assert np.all(np.diff(e) > 0)

def test_h2o_cis_amplitudes_shape(h2o_cis):
    assert h2o_cis.t_amplitudes.shape == (5, h2o_cis.n_occ, h2o_cis.n_virt)

def test_h2o_cis_fields(h2o_cis):
    for attr in ("excitation_energies", "excitation_energies_ev",
                 "oscillator_strengths", "t_amplitudes",
                 "n_occ", "n_virt", "n_basis", "n_states"):
        assert hasattr(h2o_cis, attr)
