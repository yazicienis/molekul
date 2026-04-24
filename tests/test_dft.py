"""Tests for KS-DFT LDA/PBE (Phase 12).

Reference values from PySCF at identical geometries (grids.level=5).
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.dft import ks_scf, KSResult

BOHR = 1.0 / 0.529177


@pytest.fixture(scope="module")
def h2_lda():
    mol = Molecule(atoms=[Atom("H", [0, 0, 0]), Atom("H", [0, 0, 0.74 * BOHR])],
                   charge=0, multiplicity=1)
    return ks_scf(mol, STO3G, xc='lda', verbose=False)


@pytest.fixture(scope="module")
def h2_pbe():
    mol = Molecule(atoms=[Atom("H", [0, 0, 0]), Atom("H", [0, 0, 0.74 * BOHR])],
                   charge=0, multiplicity=1)
    return ks_scf(mol, STO3G, xc='pbe', verbose=False)


@pytest.fixture(scope="module")
def h2o_lda():
    mol = Molecule(atoms=[
        Atom("O", [0, 0, 0]),
        Atom("H", [0,  0.757 * BOHR, 0.586 * BOHR]),
        Atom("H", [0, -0.757 * BOHR, 0.586 * BOHR]),
    ], charge=0, multiplicity=1)
    return ks_scf(mol, STO3G, xc='lda', verbose=False)


@pytest.fixture(scope="module")
def h2o_pbe():
    mol = Molecule(atoms=[
        Atom("O", [0, 0, 0]),
        Atom("H", [0,  0.757 * BOHR, 0.586 * BOHR]),
        Atom("H", [0, -0.757 * BOHR, 0.586 * BOHR]),
    ], charge=0, multiplicity=1)
    return ks_scf(mol, STO3G, xc='pbe', verbose=False)


# ---------- H2 LDA ----------

def test_h2_lda_converged(h2_lda):
    assert h2_lda.converged

def test_h2_lda_energy(h2_lda):
    # PySCF reference: -1.121206 Ha
    assert abs(h2_lda.energy_total - (-1.121206)) < 1e-4

def test_h2_lda_result_type(h2_lda):
    assert isinstance(h2_lda, KSResult)


# ---------- H2O LDA ----------

def test_h2o_lda_converged(h2o_lda):
    assert h2o_lda.converged

def test_h2o_lda_energy(h2o_lda):
    # PySCF reference: -74.731897 Ha
    assert abs(h2o_lda.energy_total - (-74.731897)) < 1e-3

def test_h2o_lda_exc(h2o_lda):
    # PySCF E_xc: -8.876357 Ha
    assert abs(h2o_lda.energy_xc - (-8.876357)) < 5e-2


# ---------- H2 PBE ----------

def test_h2_pbe_converged(h2_pbe):
    assert h2_pbe.converged

def test_h2_pbe_energy(h2_pbe):
    # PySCF reference: -1.152073 Ha (level=5 grid)
    assert abs(h2_pbe.energy_total - (-1.152073)) < 5e-3

def test_h2_pbe_lower_than_lda(h2_pbe, h2_lda):
    assert h2_pbe.energy_total < h2_lda.energy_total


# ---------- H2O PBE ----------

def test_h2o_pbe_converged(h2o_pbe):
    assert h2o_pbe.converged

def test_h2o_pbe_energy(h2o_pbe):
    # PySCF reference: -75.225413 Ha (level=5 grid)
    assert abs(h2o_pbe.energy_total - (-75.225413)) < 1e-2

def test_h2o_pbe_lower_than_lda(h2o_pbe, h2o_lda):
    assert h2o_pbe.energy_total < h2o_lda.energy_total

def test_h2o_pbe_exc(h2o_pbe):
    # PySCF E_xc: -9.365855 Ha
    assert abs(h2o_pbe.energy_xc - (-9.365855)) < 5e-2


# ---------- result fields ----------

def test_ksresult_fields(h2o_lda):
    for attr in ("energy_total", "energy_xc", "mo_energies", "mo_coefficients",
                 "density_matrix", "converged", "n_iter"):
        assert hasattr(h2o_lda, attr)
