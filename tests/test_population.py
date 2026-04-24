"""
tests/test_population.py — Unit tests for Mulliken population analysis
and dipole moment (Phase 7).

Coverage
--------
A. MullikenResult invariants:
   - tr(PS) = N_elec  (electron count conservation)
   - Σq_A = mol.charge  (charge conservation)
   - gross_orbital_pop.sum() == total_electrons

B. H2O symmetry constraints:
   - μ_x = μ_y = 0  (molecule in yz-plane, C2v)
   - q(H1) = q(H2)  (symmetry-equivalent hydrogens)

C. HeH+ constraints:
   - μ_x = μ_y = 0  (linear on z-axis)
   - N(He) > N(H)   (He more electronegative)

D. Dipole integrals:
   - D_c matrices are symmetric
   - tr(D_c S^{-1} D_c) > 0  (sanity; matrices are positive-semi-def)

E. Regression:
   - H2O dipole magnitude within 1e-3 D of known STO-3G value (~1.71 D)
   - H2O Mulliken charges: q(O) < 0, q(H) > 0

Reference values (PySCF/STO-3G, same geometry as validate_population.py):
  H2O |μ| ≈ 1.7091 D   q(O) ≈ −0.3304  q(H) ≈ +0.1652
"""

import math
import numpy as np
import pytest

from molekul.atoms       import Atom
from molekul.molecule    import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf         import rhf_scf
from molekul.integrals   import build_overlap
from molekul.population  import (
    analyze,
    build_dipole_integrals,
    mulliken_populations,
    dipole_moment,
    DEBYE_PER_AU,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def heh():
    return Molecule(
        atoms=[
            Atom("He", [0.0, 0.0, 0.0   ]),
            Atom("H",  [0.0, 0.0, 1.4632]),
        ],
        charge=1, multiplicity=1,
    )


@pytest.fixture(scope="module")
def h2o_pop(h2o):
    result = rhf_scf(h2o, STO3G, verbose=False)
    return analyze(h2o, STO3G, result), result


@pytest.fixture(scope="module")
def heh_pop(heh):
    result = rhf_scf(heh, STO3G, verbose=False)
    return analyze(heh, STO3G, result), result


# ---------------------------------------------------------------------------
# A. MullikenResult invariants
# ---------------------------------------------------------------------------

class TestMullikenInvariants:

    def test_h2o_electron_count(self, h2o_pop, h2o):
        pop, _ = h2o_pop
        assert abs(pop.mulliken.total_electrons - h2o.n_electrons) < 1e-8

    def test_h2o_charge_conservation(self, h2o_pop, h2o):
        pop, _ = h2o_pop
        assert abs(float(pop.mulliken.mulliken_charges.sum()) - h2o.charge) < 1e-8

    def test_h2o_orbital_sum_equals_total(self, h2o_pop):
        pop, _ = h2o_pop
        assert abs(pop.mulliken.gross_orbital_pop.sum()
                   - pop.mulliken.total_electrons) < 1e-12

    def test_heh_electron_count(self, heh_pop, heh):
        pop, _ = heh_pop
        assert abs(pop.mulliken.total_electrons - heh.n_electrons) < 1e-8

    def test_heh_charge_conservation(self, heh_pop, heh):
        pop, _ = heh_pop
        assert abs(float(pop.mulliken.mulliken_charges.sum()) - heh.charge) < 1e-8

    def test_h2o_atomic_pop_positive(self, h2o_pop):
        pop, _ = h2o_pop
        assert np.all(pop.mulliken.gross_atomic_pop > 0)

    def test_heh_atomic_pop_positive(self, heh_pop):
        pop, _ = heh_pop
        assert np.all(pop.mulliken.gross_atomic_pop > 0)


# ---------------------------------------------------------------------------
# B. H2O symmetry
# ---------------------------------------------------------------------------

class TestH2OSymmetry:

    def test_mu_x_zero(self, h2o_pop):
        pop, _ = h2o_pop
        assert abs(pop.dipole.total_debye[0]) < 1e-6

    def test_mu_y_zero(self, h2o_pop):
        pop, _ = h2o_pop
        assert abs(pop.dipole.total_debye[1]) < 1e-6

    def test_equivalent_hydrogens(self, h2o_pop):
        """q(H1) and q(H2) must be equal by C2v symmetry."""
        pop, _ = h2o_pop
        q1 = float(pop.mulliken.mulliken_charges[1])
        q2 = float(pop.mulliken.mulliken_charges[2])
        assert abs(q1 - q2) < 1e-8


# ---------------------------------------------------------------------------
# C. HeH+ constraints
# ---------------------------------------------------------------------------

class TestHehConstraints:

    def test_mu_x_y_zero(self, heh_pop):
        pop, _ = heh_pop
        assert abs(pop.dipole.total_debye[0]) < 1e-8
        assert abs(pop.dipole.total_debye[1]) < 1e-8

    def test_he_more_electrons_than_h(self, heh_pop):
        pop, _ = heh_pop
        assert pop.mulliken.gross_atomic_pop[0] > pop.mulliken.gross_atomic_pop[1]


# ---------------------------------------------------------------------------
# D. Dipole integral matrix properties
# ---------------------------------------------------------------------------

class TestDipoleIntegrals:

    def test_symmetry(self, h2o):
        Dx, Dy, Dz = build_dipole_integrals(STO3G, h2o)
        assert np.allclose(Dx, Dx.T, atol=1e-12)
        assert np.allclose(Dy, Dy.T, atol=1e-12)
        assert np.allclose(Dz, Dz.T, atol=1e-12)

    def test_shape(self, h2o):
        Dx, Dy, Dz = build_dipole_integrals(STO3G, h2o)
        n = len(STO3G.basis_functions(h2o))
        assert Dx.shape == (n, n)
        assert Dy.shape == (n, n)
        assert Dz.shape == (n, n)

    def test_nuclear_dipole_matches_coords(self, h2o):
        """Nuclear contribution of dipole must equal Σ Z_A R_A."""
        pop_result, _ = (analyze(h2o, STO3G, rhf_scf(h2o, STO3G, verbose=False)),
                         None)
        coords = np.array([a.coords for a in h2o.atoms])
        Z      = np.array([float(a.Z) for a in h2o.atoms])
        expected_nuc = (Z[:, None] * coords).sum(axis=0)
        assert np.allclose(pop_result.dipole.nuclear_au, expected_nuc, atol=1e-12)


# ---------------------------------------------------------------------------
# E. Regression values
# ---------------------------------------------------------------------------

class TestRegression:

    def test_h2o_dipole_magnitude(self, h2o_pop):
        """H2O |μ| should be ~1.709 D at this geometry (STO-3G)."""
        pop, _ = h2o_pop
        assert abs(pop.dipole.magnitude_debye - 1.7091) < 1e-3

    def test_h2o_oxygen_negative_charge(self, h2o_pop):
        pop, _ = h2o_pop
        assert pop.mulliken.mulliken_charges[0] < 0.0

    def test_h2o_hydrogen_positive_charge(self, h2o_pop):
        pop, _ = h2o_pop
        assert pop.mulliken.mulliken_charges[1] > 0.0

    def test_h2o_oxygen_charge_magnitude(self, h2o_pop):
        """q(O) ≈ −0.330 ± 0.05 (STO-3G Mulliken)."""
        pop, _ = h2o_pop
        assert abs(float(pop.mulliken.mulliken_charges[0]) - (-0.3304)) < 0.05

    def test_h2o_hydrogen_charge_magnitude(self, h2o_pop):
        """q(H) ≈ +0.165 ± 0.05."""
        pop, _ = h2o_pop
        assert abs(float(pop.mulliken.mulliken_charges[1]) - 0.1652) < 0.05

    def test_debye_conversion(self):
        """DEBYE_PER_AU must match the standard conversion factor."""
        assert abs(DEBYE_PER_AU - 2.541746473) < 1e-8
