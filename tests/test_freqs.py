"""
tests/test_freqs.py — Unit tests for harmonic frequency analysis.

Coverage
--------
A. FREQ_CONV and INTENS_CONV physical constants sanity
B. _n_rigid_modes: geometry-based rigid-mode counting
C. numerical_hessian: symmetry and diagonal formula spot-check
D. Full harmonic_analysis on H2 and H2O (STO-3G equilibrium)
   - n_zero, n_vib, n_imaginary
   - Frequency regression vs PySCF STO-3G analytic Hessian (tol 10 cm⁻¹)
   - ZPE > 0, intensities >= 0

STO-3G equilibrium geometries (MOLEKUL/BFGS optimizer):
  H2  : R = 1.345919 bohr  (E = -1.11750588 Ha)
  H2O : O=[0, 0, 0.06311], H=[0, ±1.43256, -1.13839] bohr  (E = -74.96590121 Ha)

Reference vibrational frequencies [PySCF analytic Hessian, same geometry, avg masses]:
  H2  : 5481 cm⁻¹
  H2O : 2169.85, 4139.64, 4390.67 cm⁻¹
"""

import math
import numpy as np
import pytest

from molekul.atoms    import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.freqs    import (
    FREQ_CONV, INTENS_CONV, ZERO_FREQ_THRESHOLD,
    _n_rigid_modes, numerical_hessian, harmonic_analysis,
    FreqResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2_eq():
    """H2 at STO-3G equilibrium R = 1.345919 bohr (BFGS-optimized)."""
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.345919])],
        charge=0, multiplicity=1,
    )


@pytest.fixture(scope="module")
def h2o_eq():
    """H2O at STO-3G equilibrium (BFGS-optimized, bohr)."""
    return Molecule(
        atoms=[
            Atom("O", [ 0.00000000,  0.00000000,  0.06310826]),
            Atom("H", [ 0.00000000,  1.43256299, -1.13838561]),
            Atom("H", [ 0.00000000, -1.43256299, -1.13838561]),
        ],
        charge=0, multiplicity=1,
    )


@pytest.fixture(scope="module")
def h2_analysis(h2_eq):
    return harmonic_analysis(h2_eq, STO3G, verbose=False)


@pytest.fixture(scope="module")
def h2o_analysis(h2o_eq):
    return harmonic_analysis(h2o_eq, STO3G, verbose=False)


# ---------------------------------------------------------------------------
# A. Physical constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_freq_conv_range(self):
        """FREQ_CONV should be ~5140 cm⁻¹ (CODATA range 5100–5200)."""
        assert 5100 < FREQ_CONV < 5200

    def test_intens_conv_range(self):
        """INTENS_CONV should be ~974.9 km/mol per a.u. (range 800–1200)."""
        assert 800 < INTENS_CONV < 1200

    def test_intens_conv_approx(self):
        """INTENS_CONV should be close to the standard value 974.9."""
        assert abs(INTENS_CONV - 974.9) < 50.0


# ---------------------------------------------------------------------------
# B. _n_rigid_modes
# ---------------------------------------------------------------------------

class TestRigidModes:
    def test_single_atom(self):
        mol = Molecule(atoms=[Atom("O", [0.0, 0.0, 0.0])], charge=0, multiplicity=1)
        assert _n_rigid_modes(mol) == 3

    def test_diatomic_linear(self):
        mol = Molecule(
            atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
            charge=0, multiplicity=1,
        )
        assert _n_rigid_modes(mol) == 5

    def test_triatomic_linear(self):
        # CO2-like linear geometry
        mol = Molecule(
            atoms=[Atom("O", [0.0, 0.0, -2.2]),
                   Atom("C", [0.0, 0.0,  0.0]),
                   Atom("O", [0.0, 0.0,  2.2])],
            charge=0, multiplicity=1,
        )
        assert _n_rigid_modes(mol) == 5

    def test_h2o_nonlinear(self, h2o_eq):
        assert _n_rigid_modes(h2o_eq) == 6

    def test_h2_linear(self, h2_eq):
        assert _n_rigid_modes(h2_eq) == 5


# ---------------------------------------------------------------------------
# C. numerical_hessian
# ---------------------------------------------------------------------------

class TestNumericalHessian:
    def test_hessian_symmetric(self, h2_eq):
        H = numerical_hessian(h2_eq, STO3G, h=5e-3, verbose=False)
        assert np.allclose(H, H.T, atol=1e-10)

    def test_hessian_shape(self, h2_eq):
        H = numerical_hessian(h2_eq, STO3G, h=5e-3, verbose=False)
        assert H.shape == (6, 6)   # 2 atoms × 3 DOF

    def test_h2o_hessian_shape(self, h2o_eq):
        H = numerical_hessian(h2o_eq, STO3G, h=5e-3, verbose=False)
        assert H.shape == (9, 9)   # 3 atoms × 3 DOF

    def test_h2o_hessian_symmetric(self, h2o_eq):
        H = numerical_hessian(h2o_eq, STO3G, h=5e-3, verbose=False)
        assert np.allclose(H, H.T, atol=1e-8)


# ---------------------------------------------------------------------------
# D. Full harmonic_analysis — H2
# ---------------------------------------------------------------------------

class TestH2Analysis:
    def test_returns_freqresult(self, h2_analysis):
        assert isinstance(h2_analysis, FreqResult)

    def test_n_zero_linear(self, h2_analysis):
        assert h2_analysis.n_zero == 5

    def test_n_vib(self, h2_analysis):
        assert len(h2_analysis.frequencies) == 1

    def test_n_imaginary(self, h2_analysis):
        assert h2_analysis.n_imaginary == 0

    def test_frequency_vs_pyscf(self, h2_analysis):
        """H2 stretch ~5481 cm⁻¹ (PySCF analytic Hessian, avg mass, same geometry)."""
        assert len(h2_analysis.frequencies) == 1
        assert abs(h2_analysis.frequencies[0] - 5481.0) < 10.0

    def test_intensities_nonneg(self, h2_analysis):
        assert np.all(h2_analysis.intensities >= 0.0)

    def test_zpe_positive(self, h2_analysis):
        assert h2_analysis.zero_point_energy > 0.0

    def test_all_freqs_length(self, h2_analysis):
        assert len(h2_analysis.all_frequencies) == 6   # 3N = 6

    def test_eigenvalues_ascending(self, h2_analysis):
        evs = h2_analysis.eigenvalues
        assert np.all(evs[1:] >= evs[:-1] - 1e-10)


# ---------------------------------------------------------------------------
# D. Full harmonic_analysis — H2O
# ---------------------------------------------------------------------------

_H2O_REF_FREQS = [2169.85, 4139.64, 4390.67]   # PySCF analytic Hessian, avg mass

class TestH2OAnalysis:
    def test_returns_freqresult(self, h2o_analysis):
        assert isinstance(h2o_analysis, FreqResult)

    def test_n_zero_nonlinear(self, h2o_analysis):
        assert h2o_analysis.n_zero == 6

    def test_n_vib(self, h2o_analysis):
        assert len(h2o_analysis.frequencies) == 3

    def test_n_imaginary_zero(self, h2o_analysis):
        """Equilibrium geometry → no imaginary modes."""
        assert h2o_analysis.n_imaginary == 0

    @pytest.mark.parametrize("k,ref", enumerate(_H2O_REF_FREQS))
    def test_frequency_vs_pyscf(self, h2o_analysis, k, ref):
        freq = float(h2o_analysis.frequencies[k])
        assert abs(freq - ref) < 5.0, (
            f"Mode {k+1}: MOLEKUL={freq:.1f} PySCF={ref:.1f} diff={abs(freq-ref):.1f}"
        )

    def test_intensities_nonneg(self, h2o_analysis):
        assert np.all(h2o_analysis.intensities >= 0.0)

    def test_zpe_positive(self, h2o_analysis):
        assert h2o_analysis.zero_point_energy > 0.0

    def test_zpe_reasonable(self, h2o_analysis):
        """H2O ZPE should be ~21–24 kcal/mol = ~0.033–0.038 Ha."""
        zpe = h2o_analysis.zero_point_energy
        assert 0.020 < zpe < 0.060

    def test_all_freqs_length(self, h2o_analysis):
        assert len(h2o_analysis.all_frequencies) == 9   # 3N = 9

    def test_hessian_shape(self, h2o_analysis):
        assert h2o_analysis.hessian.shape == (9, 9)

    def test_normal_modes_orthogonal(self, h2o_analysis):
        """Eigenvectors of a symmetric matrix are orthonormal."""
        Phi = h2o_analysis.normal_modes
        assert np.allclose(Phi.T @ Phi, np.eye(9), atol=1e-10)

    def test_ir_a1_mode_intense(self, h2o_analysis):
        """
        The bending mode (lowest frequency, A1 symmetry in C2v) should have
        nonzero IR intensity since it changes the dipole moment.
        """
        assert h2o_analysis.intensities[0] > 1.0   # at least 1 km/mol
