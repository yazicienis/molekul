"""Tests for TD-DFT/TDA excited states (Phase 17)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.cis import cis_excitations
from molekul.constants import ANGSTROM_TO_BOHR
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf
from molekul.tddft import TDDFTResult, tddft_tda


H2_TDA_LDA_STATE1 = 0.97176243
H2O_TDA_LDA_STATE1 = 0.42245348


def h2_molecule():
    return Molecule(
        atoms=[
            Atom("H", [0, 0, 0]),
            Atom("H", [0, 0, 0.74 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
    )


def h2o_molecule():
    return Molecule(
        atoms=[
            Atom("O", [0, 0, 0]),
            Atom("H", [0, 0.757 * ANGSTROM_TO_BOHR, 0.586 * ANGSTROM_TO_BOHR]),
            Atom("H", [0, -0.757 * ANGSTROM_TO_BOHR, 0.586 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
    )


def test_h2_tda_converged():
    result = tddft_tda(h2_molecule(), STO3G, functional="lda", n_states=1, verbose=False)
    assert isinstance(result, TDDFTResult)
    assert result.n_states == 1
    assert result.X.shape == (1, result.n_occ, result.n_virt)
    assert abs(result.excitation_energies[0] - H2_TDA_LDA_STATE1) < 0.01


def test_h2_tda_positive_energies():
    result = tddft_tda(h2_molecule(), STO3G, functional="lda", n_states=1, verbose=False)
    assert np.all(result.excitation_energies > 0.0)
    assert np.all(result.excitation_eV > 0.0)


def test_h2o_tda_state1_lda():
    result = tddft_tda(h2o_molecule(), STO3G, functional="lda", n_states=5, verbose=False)
    assert abs(result.excitation_energies[0] - H2O_TDA_LDA_STATE1) < 0.01


def test_oscillator_strengths_nonnegative():
    result = tddft_tda(h2o_molecule(), STO3G, functional="lda", n_states=5, verbose=False)
    assert np.all(result.oscillator_strengths >= -1e-12)
    assert np.any(result.oscillator_strengths > 1e-4)


def test_tddft_closer_than_cis_to_tda_reference():
    mol = h2o_molecule()
    tda = tddft_tda(mol, STO3G, functional="lda", n_states=1, verbose=False)
    rhf = rhf_scf(mol, STO3G, verbose=False)
    cis = cis_excitations(mol, STO3G, rhf, n_states=1, verbose=False)

    tddft_err = abs(tda.excitation_energies[0] - H2O_TDA_LDA_STATE1)
    cis_err = abs(cis.excitation_energies[0] - H2O_TDA_LDA_STATE1)
    assert tddft_err < cis_err
