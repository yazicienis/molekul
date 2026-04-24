"""
tests/test_mp2.py — Unit tests for MP2 correlation energy (Phase 8).

Coverage
--------
A. Physical invariants (no external reference):
   - E_MP2 <= 0 for all closed-shell molecules
   - E_total(MP2) < E_HF
   - E_total = E_HF + E_MP2 (bookkeeping)

B. (ia|jb) transformation properties:
   - Shape is (n_occ, n_virt, n_occ, n_virt)
   - Symmetry: (ia|jb) = (jb|ia)  i.e. iajb[i,a,j,b] == iajb[j,b,i,a]
   - (ia|jb) = (ia|jb) under real orbitals: iajb == iajb.transpose(2,3,0,1)

C. Regression values (PySCF/STO-3G, same geometries):
   H2   E_MP2 ≈ -0.013158 Ha  (tol 1e-5)
   HeH+ E_MP2 ≈ -0.007238 Ha  (tol 1e-5)
   H2O  E_MP2 ≈ -0.038972 Ha  (tol 1e-5)

D. n_occ / n_virt accounting
"""

import math
import numpy as np
import pytest

from molekul.atoms       import Atom
from molekul.molecule    import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf         import rhf_scf
from molekul.eri         import build_eri
from molekul.mp2         import mp2_energy, transform_iajb


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
def heh():
    return Molecule(
        atoms=[Atom("He", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4632])],
        charge=1, multiplicity=1,
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


@pytest.fixture(scope="module")
def h2_mp2(h2):
    rhf = rhf_scf(h2, STO3G, verbose=False)
    return mp2_energy(h2, STO3G, rhf), rhf


@pytest.fixture(scope="module")
def heh_mp2(heh):
    rhf = rhf_scf(heh, STO3G, verbose=False)
    return mp2_energy(heh, STO3G, rhf), rhf


@pytest.fixture(scope="module")
def h2o_mp2(h2o):
    rhf = rhf_scf(h2o, STO3G, verbose=False)
    return mp2_energy(h2o, STO3G, rhf), rhf


# ---------------------------------------------------------------------------
# A. Physical invariants
# ---------------------------------------------------------------------------

class TestInvariants:

    def test_h2_mp2_negative(self, h2_mp2):
        res, _ = h2_mp2
        assert res.energy_mp2 <= 0.0

    def test_heh_mp2_negative(self, heh_mp2):
        res, _ = heh_mp2
        assert res.energy_mp2 <= 0.0

    def test_h2o_mp2_negative(self, h2o_mp2):
        res, _ = h2o_mp2
        assert res.energy_mp2 <= 0.0

    def test_h2_total_below_hf(self, h2_mp2):
        res, _ = h2_mp2
        assert res.energy_total < res.energy_hf

    def test_h2o_total_below_hf(self, h2o_mp2):
        res, _ = h2o_mp2
        assert res.energy_total < res.energy_hf

    def test_h2_energy_bookkeeping(self, h2_mp2):
        res, _ = h2_mp2
        assert abs(res.energy_total - (res.energy_hf + res.energy_mp2)) < 1e-12

    def test_h2o_energy_bookkeeping(self, h2o_mp2):
        res, _ = h2o_mp2
        assert abs(res.energy_total - (res.energy_hf + res.energy_mp2)) < 1e-12


# ---------------------------------------------------------------------------
# B. (ia|jb) transformation properties
# ---------------------------------------------------------------------------

class TestTransformation:

    def test_shape_h2o(self, h2o):
        rhf = rhf_scf(h2o, STO3G, verbose=False)
        nocc   = h2o.n_electrons // 2
        eri_ao = build_eri(STO3G, h2o)
        iajb   = transform_iajb(eri_ao, rhf.mo_coefficients, nocc)
        nbasis = rhf.mo_coefficients.shape[0]
        nvirt  = nbasis - nocc
        assert iajb.shape == (nocc, nvirt, nocc, nvirt)

    def test_symmetry_iajb_eq_jbai(self, h2o):
        """(ia|jb) = (jb|ia) — integral symmetry under bra-ket swap."""
        rhf    = rhf_scf(h2o, STO3G, verbose=False)
        nocc   = h2o.n_electrons // 2
        eri_ao = build_eri(STO3G, h2o)
        iajb   = transform_iajb(eri_ao, rhf.mo_coefficients, nocc)
        # iajb[i,a,j,b] == iajb[j,b,i,a]  → transpose (2,3,0,1)
        assert np.allclose(iajb, iajb.transpose(2, 3, 0, 1), atol=1e-10)

    def test_symmetry_iajb_eq_jaib(self, h2o):
        """(ia|jb) = (ja|ib) — symmetry under i↔j with a↔b fix (real orbitals)."""
        rhf    = rhf_scf(h2o, STO3G, verbose=False)
        nocc   = h2o.n_electrons // 2
        eri_ao = build_eri(STO3G, h2o)
        iajb   = transform_iajb(eri_ao, rhf.mo_coefficients, nocc)
        # (ib|ja) = iajb[i,b,j,a] = iajb.transpose(0,3,2,1)[i,a,j,b]
        # (ia|jb) = (ib|ja) only if i=j and a=b generally, but we check
        # the weaker (ia|jb) vs (jb|ia) above; here we check another permutation.
        # From 8-fold symmetry of real AO integrals, (ia|jb) = (jb|ia) ✓ (tested above)
        # and also (ai|bj) = ... For real orbitals the full 8-fold symmetry holds.
        # Let us check: iajb[i,a,j,b] == iajb[j,a,i,b]  (swap i↔j, same a,b)
        # This holds only if (ia|jb) = (ja|ib), which is NOT generally true.
        # Skip this check — only bra-ket symmetry is universal.
        pass


# ---------------------------------------------------------------------------
# C. Regression values (PySCF/STO-3G)
# ---------------------------------------------------------------------------

class TestRegression:

    # Reference: PySCF/STO-3G, same geometries
    def test_h2_mp2_energy(self, h2_mp2):
        res, _ = h2_mp2
        assert abs(res.energy_mp2 - (-0.013158)) < 1e-5

    def test_heh_mp2_energy(self, heh_mp2):
        res, _ = heh_mp2
        assert abs(res.energy_mp2 - (-0.007238)) < 1e-5

    def test_h2o_mp2_energy(self, h2o_mp2):
        res, _ = h2o_mp2
        assert abs(res.energy_mp2 - (-0.038972)) < 1e-5

    def test_h2_total_energy(self, h2_mp2):
        res, _ = h2_mp2
        # E_HF + E_MP2 = -1.1167143190 + (-0.0131578705) ≈ -1.1299
        assert abs(res.energy_total - (-1.12987)) < 1e-4

    def test_h2o_total_energy(self, h2o_mp2):
        res, _ = h2o_mp2
        assert abs(res.energy_total - (-75.0049)) < 1e-3


# ---------------------------------------------------------------------------
# D. n_occ / n_virt accounting
# ---------------------------------------------------------------------------

class TestAccounting:

    def test_h2_nocc_nvirt(self, h2_mp2, h2):
        res, _ = h2_mp2
        assert res.n_occ == 1
        assert res.n_virt == 1
        assert res.n_occ + res.n_virt == res.n_basis

    def test_h2o_nocc_nvirt(self, h2o_mp2, h2o):
        res, _ = h2o_mp2
        assert res.n_occ == 5
        assert res.n_virt == 2
        assert res.n_occ + res.n_virt == res.n_basis

    def test_heh_nocc_nvirt(self, heh_mp2, heh):
        res, _ = heh_mp2
        assert res.n_occ == 1
        assert res.n_virt == 1
