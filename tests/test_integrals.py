"""
Tests for one-electron integrals (Phase 2) — comprehensive validation.

Coverage:
  A. Primitive-level: normalization, E-coefficient special cases
  B. Mathematical properties: symmetry, positive-definiteness, sign rules
  C. Regression values vs. Szabo & Ostlund and PySCF
  D. Three molecules: H2, HeH+, H2O in STO-3G

References
----------
Szabo & Ostlund, "Modern Quantum Chemistry", Appendix A (H2 STO-3G values).
PySCF 2.x reference values computed at identical geometries and basis.
"""

from math import pi, exp, sqrt

import numpy as np
import pytest

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.integrals import (
    build_overlap, build_kinetic, build_nuclear, build_core_hamiltonian,
    overlap_primitive, kinetic_primitive, nuclear_primitive,
    norm_primitive, _E, _boys,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def h2():
    """H2 at R=1.4 bohr — canonical STO-3G benchmark (Szabo & Ostlund Table A.1)."""
    return Molecule(
        atoms=[Atom("H", [0, 0, 0]), Atom("H", [0, 0, 1.4])],
        charge=0, multiplicity=1, name="H2"
    )


@pytest.fixture
def heh_plus():
    """HeH+ at R=1.4632 bohr."""
    return Molecule(
        atoms=[Atom("He", [0, 0, 0]), Atom("H", [0, 0, 1.4632])],
        charge=1, multiplicity=1, name="HeH+"
    )


@pytest.fixture
def water():
    """H2O at standard C2v geometry (O at origin, H in yz-plane)."""
    return Molecule(
        atoms=[
            Atom.from_angstrom("O",  0.0,  0.000, 0.000),
            Atom.from_angstrom("H",  0.0,  0.757, -0.586),
            Atom.from_angstrom("H",  0.0, -0.757, -0.586),
        ],
        charge=0, multiplicity=1, name="H2O"
    )


@pytest.fixture
def he_atom():
    return Molecule(
        atoms=[Atom("He", [0, 0, 0])],
        charge=0, multiplicity=1, name="He"
    )


# ============================================================================
# A. Primitive-level tests
# ============================================================================

class TestPrimitiveNormalisation:
    """N² <g|g> = 1 for each angular momentum type, computed via overlap_primitive."""

    def _check(self, lx, ly, lz, alpha=1.0, atol=1e-11):
        N = norm_primitive(alpha, lx, ly, lz)
        A = np.zeros(3)
        raw = overlap_primitive(lx, ly, lz, A, alpha, lx, ly, lz, A, alpha)
        assert abs(N**2 * raw - 1.0) < atol, (
            f"N²<g|g> = {N**2 * raw:.15f} for l=({lx},{ly},{lz}), expected 1.0"
        )

    def test_s_type(self):
        self._check(0, 0, 0)

    def test_px_type(self):
        self._check(1, 0, 0)

    def test_py_type(self):
        self._check(0, 1, 0)

    def test_pz_type(self):
        self._check(0, 0, 1)

    def test_dxx_type(self):
        self._check(2, 0, 0)

    def test_dxy_type(self):
        self._check(1, 1, 0)

    def test_dxz_type(self):
        self._check(1, 0, 1)

    def test_dyy_type(self):
        self._check(0, 2, 0)

    def test_multiple_exponents(self):
        for alpha in [0.3, 1.0, 3.5, 10.0]:
            self._check(1, 0, 0, alpha=alpha)
            self._check(0, 1, 1, alpha=alpha)


class TestECoefficients:
    """McMurchie-Davidson E^{ij}_t coefficient identities."""

    def test_e00_zero_separation(self):
        """E^{0,0}_0(Qx=0) = 1."""
        assert abs(_E(0, 0, 0, 0.0, 1.5, 2.3) - 1.0) < 1e-14

    def test_e00_nonzero_separation(self):
        """E^{0,0}_0(Qx) = exp(-mu*Qx^2)."""
        a, b, Qx = 1.2, 0.8, 1.5
        mu = a * b / (a + b)
        expected = exp(-mu * Qx**2)
        assert abs(_E(0, 0, 0, Qx, a, b) - expected) < 1e-14

    def test_e00_t_nonzero_is_zero(self):
        """E^{0,0}_t = 0 for t > 0."""
        assert _E(0, 0, 1, 1.0, 1.0, 1.0) == 0.0
        assert _E(0, 0, 2, 0.5, 2.0, 3.0) == 0.0

    def test_e10_is_xpa_times_e00(self):
        """E^{1,0}_0 = XPA * E^{0,0}_0  (XPA = -b*Qx/p)."""
        a, b, Qx = 1.2, 0.8, 1.5
        p = a + b
        XPA = -b * Qx / p
        e00 = _E(0, 0, 0, Qx, a, b)
        expected = XPA * e00
        assert abs(_E(1, 0, 0, Qx, a, b) - expected) < 1e-13

    def test_e01_is_xpb_times_e00(self):
        """E^{0,1}_0 = XPB * E^{0,0}_0  (XPB = a*Qx/p)."""
        a, b, Qx = 1.2, 0.8, 1.5
        p = a + b
        XPB = a * Qx / p
        e00 = _E(0, 0, 0, Qx, a, b)
        expected = XPB * e00
        assert abs(_E(0, 1, 0, Qx, a, b) - expected) < 1e-13

    def test_e_out_of_range_is_zero(self):
        """E^{ij}_t = 0 for t > i+j or t < 0."""
        assert _E(1, 1, 3, 0.5, 1.0, 1.0) == 0.0
        assert _E(0, 0, 1, 0.5, 1.0, 1.0) == 0.0
        assert _E(1, 0, -1, 0.5, 1.0, 1.0) == 0.0

    def test_e_symmetry_swapped_centers(self):
        """
        E^{i,j}_t(Qx, a, b) and E^{j,i}_t(-Qx, b, a) should be equal
        (swapping which atom is A and which is B).
        """
        i, j, t, Qx, a, b = 2, 1, 1, 0.7, 1.3, 0.9
        assert abs(_E(i, j, t, Qx, a, b) - _E(j, i, t, -Qx, b, a)) < 1e-13


class TestBoysFunction:
    """F_n(x) spot checks."""

    def test_boys_n0_x0(self):
        """F_0(0) = 1."""
        assert abs(_boys(0, 0) - 1.0) < 1e-14

    def test_boys_n0_large_x(self):
        """F_0(x) → sqrt(pi/x)/2 for large x."""
        x = 100.0
        expected = sqrt(pi / x) / 2
        assert abs(_boys(0, x) - expected) < 1e-6

    def test_boys_n1_x0(self):
        """F_1(0) = 1/3."""
        assert abs(_boys(1, 0) - 1.0 / 3.0) < 1e-14

    def test_boys_monotone_decreasing_in_n(self):
        """F_n(x) is decreasing in n for fixed x > 0."""
        x = 2.0
        vals = [_boys(n, x) for n in range(5)]
        assert all(vals[i] > vals[i + 1] for i in range(4))


class TestPrimitiveOverlap:
    """Direct tests of overlap_primitive()."""

    def test_ss_same_center_normalised(self):
        """N² * <s|s> at same center = 1."""
        a = 1.7
        N = norm_primitive(a, 0, 0, 0)
        A = np.array([0.0, 0.0, 0.0])
        raw = overlap_primitive(0, 0, 0, A, a, 0, 0, 0, A, a)
        assert abs(N**2 * raw - 1.0) < 1e-12

    def test_pp_same_center_normalised(self):
        """N² * <px|px> at same center = 1."""
        a = 1.7
        N = norm_primitive(a, 1, 0, 0)
        A = np.array([0.0, 0.0, 0.0])
        raw = overlap_primitive(1, 0, 0, A, a, 1, 0, 0, A, a)
        assert abs(N**2 * raw - 1.0) < 1e-12

    def test_sp_same_center_zero(self):
        """<s|px> at same center = 0 by parity."""
        a = 1.7
        A = np.array([0.0, 0.0, 0.0])
        raw = overlap_primitive(0, 0, 0, A, a, 1, 0, 0, A, a)
        assert abs(raw) < 1e-14

    def test_px_py_same_center_zero(self):
        """<px|py> at same center = 0 by angular momentum."""
        a = 1.7
        A = np.array([0.0, 0.0, 0.0])
        raw = overlap_primitive(1, 0, 0, A, a, 0, 1, 0, A, a)
        assert abs(raw) < 1e-14

    def test_ss_decreases_with_distance(self):
        """<s_A|s_B> decreases as A,B separate."""
        a, b = 1.0, 1.0
        A = np.array([0.0, 0.0, 0.0])
        B1 = np.array([0.0, 0.0, 0.5])
        B2 = np.array([0.0, 0.0, 2.0])
        s1 = overlap_primitive(0, 0, 0, A, a, 0, 0, 0, B1, b)
        s2 = overlap_primitive(0, 0, 0, A, a, 0, 0, 0, B2, b)
        assert s1 > s2 > 0

    def test_overlap_symmetry(self):
        """<g1|g2> = <g2|g1>."""
        a, b = 1.3, 0.9
        A = np.array([0.1, 0.2, 0.3])
        B = np.array([0.5, 0.0, 1.1])
        fwd = overlap_primitive(1, 0, 0, A, a, 0, 1, 0, B, b)
        rev = overlap_primitive(0, 1, 0, B, b, 1, 0, 0, A, a)
        assert abs(fwd - rev) < 1e-13


# ============================================================================
# B. Mathematical properties of contracted integral matrices
# ============================================================================

def _assert_symmetric(M, name, atol=1e-10):
    np.testing.assert_allclose(M, M.T, atol=atol,
                               err_msg=f"{name} must be symmetric")


def _assert_positive_definite(M, name):
    eigs = np.linalg.eigvalsh(M)
    assert np.all(eigs > 0), f"{name} must be positive definite; min eigval = {eigs.min():.4e}"


def _assert_diagonal_unity(M, name, atol=1e-5):
    diag = np.diag(M)
    np.testing.assert_allclose(diag, np.ones_like(diag), atol=atol,
                               err_msg=f"{name} diagonal must be 1")


class TestMathProperties:
    """Matrix-level mathematical invariants — molecule-agnostic."""

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_overlap_symmetric(self, mol_fixture, request):
        mol = request.getfixturevalue(mol_fixture)
        _assert_symmetric(build_overlap(STO3G, mol), "S")

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_overlap_diagonal_unity(self, mol_fixture, request):
        mol = request.getfixturevalue(mol_fixture)
        _assert_diagonal_unity(build_overlap(STO3G, mol), "S")

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_overlap_positive_definite(self, mol_fixture, request):
        mol = request.getfixturevalue(mol_fixture)
        _assert_positive_definite(build_overlap(STO3G, mol), "S")

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_overlap_cauchy_schwarz(self, mol_fixture, request):
        """
        Cauchy-Schwarz: |S_ij|^2 ≤ S_ii * S_jj.
        Diagonal is numerically ~1 ± 1e-5, so we check against actual diagonal.
        """
        mol = request.getfixturevalue(mol_fixture)
        S = build_overlap(STO3G, mol)
        diag = np.diag(S)
        n = len(diag)
        for i in range(n):
            for j in range(n):
                bound = diag[i] * diag[j]
                assert S[i, j]**2 <= bound + 1e-8, (
                    f"Cauchy-Schwarz violated at [{i},{j}]: "
                    f"S²={S[i,j]**2:.6f} > S_ii*S_jj={bound:.6f}"
                )

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_overlap_trace(self, mol_fixture, request):
        """Tr(S) = n_basis (from diagonal = 1)."""
        mol = request.getfixturevalue(mol_fixture)
        S = build_overlap(STO3G, mol)
        n = STO3G.n_basis(mol)
        assert abs(np.trace(S) - n) < 1e-4

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_kinetic_symmetric(self, mol_fixture, request):
        mol = request.getfixturevalue(mol_fixture)
        _assert_symmetric(build_kinetic(STO3G, mol), "T")

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_kinetic_diagonal_positive(self, mol_fixture, request):
        """T_ii = <phi_i|-½∇²|phi_i> > 0 for any normalised basis function."""
        mol = request.getfixturevalue(mol_fixture)
        T = build_kinetic(STO3G, mol)
        diag = np.diag(T)
        assert np.all(diag > 0), f"T diagonal must be positive; got min={diag.min():.4e}"

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_kinetic_positive_definite(self, mol_fixture, request):
        """T must be positive definite (½<∇φ|∇ψ> Gram matrix)."""
        mol = request.getfixturevalue(mol_fixture)
        _assert_positive_definite(build_kinetic(STO3G, mol), "T")

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_nuclear_symmetric(self, mol_fixture, request):
        mol = request.getfixturevalue(mol_fixture)
        _assert_symmetric(build_nuclear(STO3G, mol), "V")

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_nuclear_diagonal_negative(self, mol_fixture, request):
        """V_ii = <phi_i|Σ -Z/r|phi_i> < 0 (always attractive)."""
        mol = request.getfixturevalue(mol_fixture)
        V = build_nuclear(STO3G, mol)
        diag = np.diag(V)
        assert np.all(diag < 0), f"V diagonal must be negative; got max={diag.max():.4e}"

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_hcore_symmetric(self, mol_fixture, request):
        mol = request.getfixturevalue(mol_fixture)
        _assert_symmetric(build_core_hamiltonian(STO3G, mol), "H_core")

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_hcore_diagonal_negative(self, mol_fixture, request):
        """H_core diagonal should be negative (V dominates T for atom-centred basis)."""
        mol = request.getfixturevalue(mol_fixture)
        H = build_core_hamiltonian(STO3G, mol)
        diag = np.diag(H)
        assert np.all(diag < 0), f"H_core diagonal not all negative; got max={diag.max():.4e}"

    @pytest.mark.parametrize("mol_fixture", ["h2", "heh_plus", "water"])
    def test_matrix_shape(self, mol_fixture, request):
        mol = request.getfixturevalue(mol_fixture)
        n = STO3G.n_basis(mol)
        for M, name in [
            (build_overlap(STO3G, mol), "S"),
            (build_kinetic(STO3G, mol), "T"),
            (build_nuclear(STO3G, mol), "V"),
        ]:
            assert M.shape == (n, n), f"{name} shape mismatch"


# ============================================================================
# C. Regression tests — values from Szabo & Ostlund and PySCF
# ============================================================================

class TestH2Regression:
    """
    H2 STO-3G at R=1.4 bohr.
    Primary reference: Szabo & Ostlund, Appendix A, Table A.1.
    Confirmed by PySCF.
    """
    TOL = 5e-5   # tolerance vs. reference

    def test_n_basis(self, h2):
        assert STO3G.n_basis(h2) == 2

    def test_nuclear_repulsion(self, h2):
        """E_nuc = Z_H * Z_H / R = 1/1.4."""
        assert abs(h2.nuclear_repulsion_energy() - 1.0 / 1.4) < 1e-8

    def test_S12(self, h2):
        """S12 = 0.6593 (S&O)."""
        S = build_overlap(STO3G, h2)
        assert abs(S[0, 1] - 0.6593182) < self.TOL

    def test_T11(self, h2):
        """T11 = 0.7600 (S&O)."""
        T = build_kinetic(STO3G, h2)
        assert abs(T[0, 0] - 0.7600319) < self.TOL

    def test_T12(self, h2):
        """T12 = 0.2365 (S&O)."""
        T = build_kinetic(STO3G, h2)
        assert abs(T[0, 1] - 0.2364547) < self.TOL

    def test_V11(self, h2):
        """V11 = -1.8804 (PySCF reference)."""
        V = build_nuclear(STO3G, h2)
        assert abs(V[0, 0] - (-1.8804409)) < self.TOL

    def test_V12(self, h2):
        """V12 = -1.1948 (PySCF reference)."""
        V = build_nuclear(STO3G, h2)
        assert abs(V[0, 1] - (-1.1948346)) < self.TOL

    def test_H11(self, h2):
        """H_core(1,1) = -1.1204 (S&O)."""
        H = build_core_hamiltonian(STO3G, h2)
        assert abs(H[0, 0] - (-1.1204090)) < self.TOL

    def test_H12(self, h2):
        """H_core(1,2) = -0.9577 (S&O)."""
        H = build_core_hamiltonian(STO3G, h2)
        assert abs(H[0, 1] - (-0.9583800)) < self.TOL

    def test_symmetry_of_all_matrices(self, h2):
        for build, name in [
            (build_overlap, "S"),
            (build_kinetic, "T"),
            (build_nuclear, "V"),
            (build_core_hamiltonian, "H"),
        ]:
            M = build(STO3G, h2)
            np.testing.assert_allclose(M, M.T, atol=1e-12,
                                       err_msg=f"H2 {name} not symmetric")


class TestHeHPlusRegression:
    """
    HeH+ STO-3G at R=1.4632 bohr.
    Reference: PySCF 2.x (no Szabo & Ostlund entry).
    """
    TOL = 5e-5

    def test_n_basis(self, heh_plus):
        assert STO3G.n_basis(heh_plus) == 2

    def test_nuclear_repulsion(self, heh_plus):
        """E_nuc = Z_He * Z_H / R = 2 / 1.4632."""
        assert abs(heh_plus.nuclear_repulsion_energy() - 2.0 / 1.4632) < 1e-6

    def test_S_diagonal(self, heh_plus):
        S = build_overlap(STO3G, heh_plus)
        assert abs(S[0, 0] - 1.0) < 1e-4
        assert abs(S[1, 1] - 1.0) < 1e-4

    def test_S12(self, heh_plus):
        """S12 = 0.53682 (PySCF)."""
        S = build_overlap(STO3G, heh_plus)
        assert abs(S[0, 1] - 0.5368194) < self.TOL

    def test_T11(self, heh_plus):
        """T11 = 1.41176 (PySCF) — He 1s has higher kinetic energy."""
        T = build_kinetic(STO3G, heh_plus)
        assert abs(T[0, 0] - 1.4117632) < self.TOL

    def test_T22(self, heh_plus):
        """T22 = 0.76003 (PySCF) — H 1s matches H2 diagonal."""
        T = build_kinetic(STO3G, heh_plus)
        assert abs(T[1, 1] - 0.7600319) < self.TOL

    def test_T12(self, heh_plus):
        """T12 = 0.19744 (PySCF)."""
        T = build_kinetic(STO3G, heh_plus)
        assert abs(T[0, 1] - 0.1974432) < self.TOL

    def test_V11(self, heh_plus):
        """V11 = -4.01005 (PySCF)."""
        V = build_nuclear(STO3G, heh_plus)
        assert abs(V[0, 0] - (-4.0100462)) < self.TOL

    def test_V22(self, heh_plus):
        """V22 = -2.49186 (PySCF)."""
        V = build_nuclear(STO3G, heh_plus)
        assert abs(V[1, 1] - (-2.4918576)) < self.TOL

    def test_V12(self, heh_plus):
        """V12 = -1.62927 (PySCF)."""
        V = build_nuclear(STO3G, heh_plus)
        assert abs(V[0, 1] - (-1.6292717)) < self.TOL

    def test_H11(self, heh_plus):
        """H_core(1,1) = -2.59828 (PySCF)."""
        H = build_core_hamiltonian(STO3G, heh_plus)
        assert abs(H[0, 0] - (-2.5982830)) < self.TOL

    def test_H22(self, heh_plus):
        """H_core(2,2) = -1.73183 (PySCF)."""
        H = build_core_hamiltonian(STO3G, heh_plus)
        assert abs(H[1, 1] - (-1.7318257)) < self.TOL


class TestWaterRegression:
    """
    H2O STO-3G at O=(0,0,0), H=(0,±0.757,−0.586) Å.
    Reference: PySCF 2.x.
    Basis function ordering: 1s_O, 2s_O, 2px_O, 2py_O, 2pz_O, 1s_H1, 1s_H2.
    """
    TOL = 1e-4

    def test_n_basis(self, water):
        assert STO3G.n_basis(water) == 7

    def test_S_diagonal(self, water):
        S = build_overlap(STO3G, water)
        np.testing.assert_allclose(np.diag(S), np.ones(7), atol=1e-4)

    def test_S_positive_definite(self, water):
        S = build_overlap(STO3G, water)
        assert np.all(np.linalg.eigvalsh(S) > 0)

    # Specific off-diagonal elements (PySCF reference)
    def test_S_1s_2s_oxygen(self, water):
        """S[0,1] (1s_O, 2s_O) = 0.2367 (PySCF)."""
        S = build_overlap(STO3G, water)
        assert abs(S[0, 1] - 0.23670394) < self.TOL

    def test_S_px_is_zero_with_H(self, water):
        """
        2px_O has zero overlap with both H atoms: px is perpendicular to
        the molecular plane (yz), so <px|1s_H> = 0 by parity.
        """
        S = build_overlap(STO3G, water)
        assert abs(S[2, 5]) < 1e-8
        assert abs(S[2, 6]) < 1e-8

    def test_S_py_antisymmetric_with_H(self, water):
        """
        2py_O: H1 is at +y, H2 at -y → S[3,5] = -S[3,6].
        """
        S = build_overlap(STO3G, water)
        assert abs(S[3, 5] + S[3, 6]) < self.TOL
        assert abs(S[3, 5] - 0.31109395) < self.TOL

    def test_S_pz_symmetric_with_H(self, water):
        """
        2pz_O: both H are at same z → S[4,5] = S[4,6].
        """
        S = build_overlap(STO3G, water)
        assert abs(S[4, 5] - S[4, 6]) < 1e-8
        assert abs(S[4, 5] - (-0.24082042)) < self.TOL

    def test_T_1s_O_diagonal(self, water):
        """T[0,0] (1s_O core) = 29.0032 (PySCF)."""
        T = build_kinetic(STO3G, water)
        assert abs(T[0, 0] - 29.00319995) < self.TOL * 10

    def test_T_2s_O_diagonal(self, water):
        """T[1,1] = 0.80813 (PySCF)."""
        T = build_kinetic(STO3G, water)
        assert abs(T[1, 1] - 0.80812795) < self.TOL

    def test_T_px_diagonal(self, water):
        """T[2,2] = T[3,3] = T[4,4] = 2.52873 (PySCF, 2p functions on O)."""
        T = build_kinetic(STO3G, water)
        for idx in [2, 3, 4]:
            assert abs(T[idx, idx] - 2.52873120) < self.TOL

    def test_T_H_diagonal(self, water):
        """T[5,5] = T[6,6] = 0.76003 (PySCF, H 1s)."""
        T = build_kinetic(STO3G, water)
        for idx in [5, 6]:
            assert abs(T[idx, idx] - 0.76003188) < self.TOL

    def test_V_1s_O_diagonal(self, water):
        """V[0,0] = -61.724 (PySCF)."""
        V = build_nuclear(STO3G, water)
        assert abs(V[0, 0] - (-61.72400373)) < self.TOL * 100

    def test_H_diagonal_values(self, water):
        """H_core diagonal matches PySCF within tolerance."""
        H = build_core_hamiltonian(STO3G, water)
        ref = np.array([-32.72080378, -9.33471026, -7.45756728,
                        -7.61384064, -7.55121310, -5.07626089, -5.07626089])
        np.testing.assert_allclose(np.diag(H), ref, atol=self.TOL * 10,
                                   err_msg="H_core diagonal mismatch vs PySCF")

    def test_H_positive_definite_S(self, water):
        """Full matrix-level: S > 0 (all eigenvalues positive)."""
        S = build_overlap(STO3G, water)
        eigs = np.linalg.eigvalsh(S)
        assert eigs.min() > 0

    def test_T_positive_definite(self, water):
        T = build_kinetic(STO3G, water)
        eigs = np.linalg.eigvalsh(T)
        assert eigs.min() > 0


# ============================================================================
# D. Basis set data structure tests
# ============================================================================

class TestBasisDataStructures:
    def test_sto3g_elements_present(self):
        for el in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]:
            assert el in STO3G.shells_by_element

    def test_hydrogen_one_shell(self):
        shells = STO3G.shells_by_element["H"]
        assert len(shells) == 1
        assert shells[0].l == 0
        assert shells[0].n_primitives == 3

    def test_oxygen_three_shells(self):
        shells = STO3G.shells_by_element["O"]
        assert len(shells) == 3
        assert shells[0].l == 0  # 1s
        assert shells[1].l == 0  # 2s
        assert shells[2].l == 1  # 2p

    def test_oxygen_five_basis_functions(self, he_atom):
        """O contributes 5 basis functions (1s+2s+2px+2py+2pz)."""
        o_mol = Molecule(
            atoms=[Atom("O", [0, 0, 0])],
            charge=0, multiplicity=1, name="O"
        )
        assert STO3G.n_basis(o_mol) == 5

    def test_h2_two_basis_functions(self, h2):
        assert STO3G.n_basis(h2) == 2

    def test_heh_plus_two_basis_functions(self, heh_plus):
        assert STO3G.n_basis(heh_plus) == 2

    def test_water_seven_basis_functions(self, water):
        assert STO3G.n_basis(water) == 7
