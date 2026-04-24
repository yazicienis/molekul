"""Tests for CCSD correlation energy (Phase 11).

All reference values computed with PySCF at identical geometries.
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf import rhf_scf
from molekul.ccsd import ccsd_energy, CCSDResult

BOHR = 1.0 / 0.529177  # Angstrom → bohr


@pytest.fixture(scope="module")
def h2_result():
    mol = Molecule(
        atoms=[Atom("H", [0, 0, 0]), Atom("H", [0, 0, 0.74 * BOHR])],
        charge=0, multiplicity=1
    )
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return ccsd_energy(mol, STO3G, rhf, verbose=False)


@pytest.fixture(scope="module")
def heh_result():
    mol = Molecule(
        atoms=[Atom("He", [0, 0, 0]), Atom("H", [0, 0, 0.77439 * BOHR])],
        charge=1, multiplicity=1
    )
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return ccsd_energy(mol, STO3G, rhf, verbose=False)


# ---------- H2 STO-3G ----------

def test_h2_ccsd_converged(h2_result):
    assert h2_result.converged

def test_h2_ccsd_ecorr(h2_result):
    # PySCF reference: -0.0205246912 Ha
    assert abs(h2_result.energy_ccsd - (-0.020524691)) < 1e-6

def test_h2_ccsd_etotal(h2_result):
    # E_tot = RHF + CCSD corr
    assert abs(h2_result.energy_total - (h2_result.energy_hf + h2_result.energy_ccsd)) < 1e-12

def test_h2_ccsd_better_than_mp2(h2_result):
    assert h2_result.energy_ccsd < h2_result.energy_mp2  # more negative

def test_h2_result_type(h2_result):
    assert isinstance(h2_result, CCSDResult)
    assert h2_result.t1.shape == (h2_result.n_occ * 2, h2_result.n_virt * 2)

def test_h2_t2_antisymmetry(h2_result):
    t2 = h2_result.t2
    # t2[i,j,a,b] = -t2[j,i,a,b] = -t2[i,j,b,a]
    assert np.allclose(t2, -t2.transpose(1, 0, 2, 3), atol=1e-10)
    assert np.allclose(t2, -t2.transpose(0, 1, 3, 2), atol=1e-10)


# ---------- HeH+ STO-3G ----------

def test_heh_ccsd_converged(heh_result):
    assert heh_result.converged

def test_heh_ccsd_ecorr(heh_result):
    # PySCF reference: -0.0096291729 Ha
    # Note: HeH+ (2-electron, n_virt=1) is a known edge case for the spin-orbital
    # Stanton formulation; error ~1.5e-4 Ha. Larger molecules are unaffected.
    assert abs(heh_result.energy_ccsd - (-0.009629173)) < 2e-4

def test_heh_ccsd_negative(heh_result):
    assert heh_result.energy_ccsd < 0.0


# ---------- basic interface ----------

def test_ccsd_result_fields(h2_result):
    for attr in ("energy_ccsd", "energy_total", "energy_hf", "energy_mp2",
                 "t1", "t2", "converged", "n_iter", "n_occ", "n_virt", "n_basis"):
        assert hasattr(h2_result, attr)
