"""
tests/test_basis_ccpvdz.py — Unit tests for cc-pVDZ and 6-31G* basis sets.

Coverage
--------
A. Basis function counts (n_basis)
B. RHF convergence and energy regression vs PySCF/Cartesian d
C. MP2 invariants (E_MP2 < 0, E_total < E_HF)
D. Energy ordering: larger basis ≥ lower energy (variational principle)
E. Overlap matrix properties: diagonal ≈ 1, S positive-definite

Reference energies (PySCF, Cartesian d, same geometries)
---------------------------------------------------------
H2  cc-pVDZ : E_HF = -1.12870945  Ha   n_basis = 10
H2  6-31G*  : E_HF = -1.12674270  Ha   n_basis =  4
H2O cc-pVDZ : E_HF = -76.02347668 Ha   n_basis = 25
H2O 6-31G*  : E_HF = -76.00677042 Ha   n_basis = 19
"""

import math
import numpy as np
import pytest

from molekul.atoms         import Atom
from molekul.molecule      import Molecule
from molekul.basis_sto3g   import STO3G
from molekul.basis_ccpvdz  import ccpVDZ
from molekul.basis_631gstar import G631Star
from molekul.rhf           import rhf_scf
from molekul.mp2           import mp2_energy
from molekul.integrals     import build_overlap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2():
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )


@pytest.fixture(scope="module")
def h2o():
    R, th = 1.870, math.radians(50.0)
    return Molecule(
        atoms=[
            Atom("O", [0.0,  0.0,              0.0]),
            Atom("H", [0.0,  R * math.sin(th), -R * math.cos(th)]),
            Atom("H", [0.0, -R * math.sin(th), -R * math.cos(th)]),
        ],
        charge=0, multiplicity=1,
    )


# ---------------------------------------------------------------------------
# A. Basis function counts
# ---------------------------------------------------------------------------

class TestNBasis:
    def test_h2_ccpvdz(self, h2):
        assert ccpVDZ.n_basis(h2) == 10    # 2 × [2s1p]

    def test_h2_631gstar(self, h2):
        assert G631Star.n_basis(h2) == 4   # 2 × [2s]

    def test_h2o_ccpvdz(self, h2o):
        assert ccpVDZ.n_basis(h2o) == 25   # O[3s2p1d]=15 + 2×H[2s1p]=10

    def test_h2o_631gstar(self, h2o):
        assert G631Star.n_basis(h2o) == 19  # O[3s2p1d]=15 + 2×H[2s]=4


# ---------------------------------------------------------------------------
# B. RHF regression vs PySCF (Cartesian d, tol 1e-6 Ha)
# ---------------------------------------------------------------------------

class TestRHFRegression:
    def test_h2_ccpvdz(self, h2):
        r = rhf_scf(h2, ccpVDZ, verbose=False)
        assert r.converged
        assert abs(r.energy_total - (-1.12870945)) < 1e-6

    def test_h2_631gstar(self, h2):
        r = rhf_scf(h2, G631Star, verbose=False)
        assert r.converged
        assert abs(r.energy_total - (-1.12674270)) < 1e-6

    def test_h2o_ccpvdz(self, h2o):
        r = rhf_scf(h2o, ccpVDZ, verbose=False)
        assert r.converged
        assert abs(r.energy_total - (-76.02347668)) < 1e-6

    def test_h2o_631gstar(self, h2o):
        r = rhf_scf(h2o, G631Star, verbose=False)
        assert r.converged
        assert abs(r.energy_total - (-76.00677042)) < 1e-6


# ---------------------------------------------------------------------------
# C. MP2 invariants
# ---------------------------------------------------------------------------

class TestMP2Invariants:
    def test_h2o_ccpvdz_mp2_negative(self, h2o):
        r = rhf_scf(h2o, ccpVDZ, verbose=False)
        mp = mp2_energy(h2o, ccpVDZ, r)
        assert mp.energy_mp2 <= 0.0
        assert mp.energy_total < mp.energy_hf

    def test_h2o_631gstar_mp2_negative(self, h2o):
        r = rhf_scf(h2o, G631Star, verbose=False)
        mp = mp2_energy(h2o, G631Star, r)
        assert mp.energy_mp2 <= 0.0
        assert mp.energy_total < mp.energy_hf


# ---------------------------------------------------------------------------
# D. Variational principle: larger basis → lower or equal HF energy
# ---------------------------------------------------------------------------

class TestVariationalOrdering:
    def test_h2_basis_ordering(self, h2):
        e_sto3g  = rhf_scf(h2, STO3G,   verbose=False).energy_total
        e_631gs  = rhf_scf(h2, G631Star, verbose=False).energy_total
        e_ccpvdz = rhf_scf(h2, ccpVDZ,  verbose=False).energy_total
        assert e_sto3g  >= e_631gs  - 1e-8
        assert e_631gs  >= e_ccpvdz - 1e-8

    def test_h2o_ccpvdz_lower_than_sto3g(self, h2o):
        e_sto3g  = rhf_scf(h2o, STO3G,  verbose=False).energy_total
        e_ccpvdz = rhf_scf(h2o, ccpVDZ, verbose=False).energy_total
        assert e_ccpvdz < e_sto3g


# ---------------------------------------------------------------------------
# E. Overlap matrix properties
# ---------------------------------------------------------------------------

class TestOverlapMatrix:
    def test_h2o_ccpvdz_overlap_symmetric(self, h2o):
        S = build_overlap(ccpVDZ, h2o)
        assert np.allclose(S, S.T, atol=1e-12)

    def test_h2o_ccpvdz_overlap_positive_definite(self, h2o):
        S = build_overlap(ccpVDZ, h2o)
        eigs = np.linalg.eigvalsh(S)
        assert np.all(eigs > 0)

    def test_h2o_631gstar_overlap_symmetric(self, h2o):
        S = build_overlap(G631Star, h2o)
        assert np.allclose(S, S.T, atol=1e-12)
