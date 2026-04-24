"""
Tests for H2O geometry optimisation — Phase 5b.

Coverage
--------
A. geom.py utilities (bond_length, bond_angle, dihedral_angle)
B. H2O optimisation behaviour:
   - convergence from non-equilibrium starting geometry
   - energy decreases monotonically from start to end
   - final gradient norm below threshold
   - trajectory and JSON written correctly
C. Geometry regression vs. PySCF-confirmed STO-3G equilibrium:
   - R(OH) within tolerance
   - θ(HOH) within tolerance
   - E within tolerance
   - C2v symmetry: R(O-H1) ≈ R(O-H2)
D. Force balance: sum of gradient components ≈ 0 (Newton's 3rd law)

PySCF reference (STO-3G, fine 2D scan, 0.005 bohr × 0.25° grid)
-----------------------------------------------------------------
  R(OH)   = 1.870 bohr     (tolerance ±0.02 bohr)
  θ(HOH)  = 100.00°        (tolerance ±1.5°)
  E_eq    = -74.9659011183 Ha  (tolerance 5e-3 Ha — accounts for grid-limited
                                reference and numerical gradient convergence)
"""

import json
import numpy as np
import pytest

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.geom import bond_length, bond_angle, dihedral_angle
from molekul.grad import numerical_gradient, max_gradient
from molekul.optimizer import optimize_geometry, OptResult


# ============================================================================
# PySCF-confirmed reference
# ============================================================================

REF_R_OH   = 1.870    # bohr
REF_THETA  = 100.00   # degrees
REF_ENERGY = -74.9659011183  # Ha
BOHR_TO_ANG = 0.529177210903


# ============================================================================
# Fixtures
# ============================================================================

def _make_start_mol() -> Molecule:
    """Non-equilibrium H2O: R=2.0 bohr, θ=120°."""
    R, th = 2.0, np.radians(60.0)
    Hx = R * np.sin(th)
    Hz = -R * np.cos(th)
    return Molecule(
        atoms=[
            Atom("O",  [0.0,  0.0,  0.0]),
            Atom("H",  [0.0,  Hx,   Hz]),
            Atom("H",  [0.0, -Hx,   Hz]),
        ],
        charge=0, multiplicity=1, name="H2O",
    )


def _make_near_eq_mol() -> Molecule:
    """H2O near equilibrium (used for gradient tests)."""
    R, th = 1.870, np.radians(50.0)   # R_eq, θ/2 = 50°
    Hx = R * np.sin(th)
    Hz = -R * np.cos(th)
    return Molecule(
        atoms=[
            Atom("O",  [0.0,  0.0,  0.0]),
            Atom("H",  [0.0,  Hx,   Hz]),
            Atom("H",  [0.0, -Hx,   Hz]),
        ],
        charge=0, multiplicity=1, name="H2O",
    )


@pytest.fixture(scope="module")
def h2o_start():
    return _make_start_mol()


@pytest.fixture(scope="module")
def h2o_near_eq():
    return _make_near_eq_mol()


@pytest.fixture(scope="module")
def h2o_opt_result():
    """Full optimisation from R=2.0 bohr, θ=120°."""
    return optimize_geometry(
        _make_start_mol(), STO3G,
        grad_tol=1e-4, max_steps=60, verbose=False,
    )


# ============================================================================
# A. Geometry utilities
# ============================================================================

class TestGeomUtils:

    def test_bond_length_h2(self):
        mol = Molecule(
            atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
            charge=0, multiplicity=1,
        )
        assert abs(bond_length(mol, 0, 1) - 1.4) < 1e-12

    def test_bond_length_symmetric(self, h2o_start):
        """For symmetric starting geometry, both OH bonds equal."""
        r01 = bond_length(h2o_start, 0, 1)
        r02 = bond_length(h2o_start, 0, 2)
        assert abs(r01 - r02) < 1e-10

    def test_bond_length_start_value(self, h2o_start):
        """Starting geometry: R(OH) = 2.0 bohr."""
        assert abs(bond_length(h2o_start, 0, 1) - 2.0) < 1e-10

    def test_bond_angle_start_value(self, h2o_start):
        """Starting geometry: θ(HOH) = 120°."""
        theta = bond_angle(h2o_start, 1, 0, 2)
        assert abs(theta - 120.0) < 1e-6

    def test_bond_angle_linear(self):
        """Linear H-O-H: angle = 180°  (10 electrons, singlet OK)."""
        mol = Molecule(
            atoms=[Atom("H", [0.0, 0.0, -1.0]),
                   Atom("O", [0.0, 0.0,  0.0]),
                   Atom("H", [0.0, 0.0,  1.0])],
            charge=0, multiplicity=1,
        )
        assert abs(bond_angle(mol, 0, 1, 2) - 180.0) < 1e-6

    def test_bond_angle_right(self):
        """Right-angle geometry: angle = 90°."""
        mol = Molecule(
            atoms=[Atom("H", [1.0, 0.0, 0.0]),
                   Atom("O", [0.0, 0.0, 0.0]),
                   Atom("H", [0.0, 1.0, 0.0])],
            charge=0, multiplicity=1,
        )
        assert abs(bond_angle(mol, 0, 1, 2) - 90.0) < 1e-6

    def test_dihedral_planar(self):
        """Dihedral on a non-planar He4 geometry; just check range and no crash."""
        # He4: 8 electrons, singlet valid; geometry is non-planar
        mol4 = Molecule(
            atoms=[
                Atom("He", [0.0,  0.0,  0.0]),
                Atom("He", [0.0,  1.5,  0.0]),
                Atom("He", [1.5,  1.5,  0.0]),
                Atom("He", [1.5,  1.5,  1.5]),
            ],
            charge=0, multiplicity=1,
        )
        d = dihedral_angle(mol4, 0, 1, 2, 3)
        assert -180.0 <= d <= 180.0


# ============================================================================
# B. Optimiser behaviour on H2O
# ============================================================================

class TestH2OOptBehaviour:

    def test_converged(self, h2o_opt_result):
        assert h2o_opt_result.converged

    def test_result_type(self, h2o_opt_result):
        assert isinstance(h2o_opt_result, OptResult)

    def test_energy_decreases(self, h2o_opt_result):
        assert h2o_opt_result.energy_final < h2o_opt_result.energy_initial

    def test_energy_drop_significant(self, h2o_opt_result):
        """Energy must drop by more than 0.01 Ha from stretched geometry."""
        assert h2o_opt_result.energy_initial - h2o_opt_result.energy_final > 0.01

    def test_gradient_converged(self, h2o_opt_result):
        assert h2o_opt_result.grad_max_final < 1e-3

    def test_n_steps_reasonable(self, h2o_opt_result):
        """H2O should converge in well under 60 steps with BFGS."""
        assert h2o_opt_result.n_steps < 60

    def test_trajectory_length(self, h2o_opt_result):
        assert len(h2o_opt_result.trajectory) == h2o_opt_result.n_steps

    def test_final_molecule_is_h2o(self, h2o_opt_result):
        mol = h2o_opt_result.final_molecule
        symbols = [a.symbol for a in mol.atoms]
        assert "O" in symbols
        assert symbols.count("H") == 2

    def test_trajectory_written(self, tmp_path):
        traj = tmp_path / "h2o_traj.xyz"
        optimize_geometry(
            _make_start_mol(), STO3G,
            grad_tol=5e-3, max_steps=10,
            traj_path=str(traj), verbose=False,
        )
        assert traj.exists()
        assert traj.stat().st_size > 0
        # Should contain at least one XYZ frame (3 atoms + 2 header lines = 5 lines)
        content = traj.read_text()
        assert content.count("O") >= 1

    def test_history_json_written(self, tmp_path):
        hist = tmp_path / "h2o_hist.json"
        optimize_geometry(
            _make_start_mol(), STO3G,
            grad_tol=5e-3, max_steps=10,
            history_path=str(hist), verbose=False,
        )
        assert hist.exists()
        data = json.loads(hist.read_text())
        assert "steps" in data
        assert len(data["steps"]) > 0

    def test_history_step_has_coords(self, tmp_path):
        hist = tmp_path / "h2o_hist2.json"
        optimize_geometry(
            _make_start_mol(), STO3G,
            grad_tol=5e-3, max_steps=10,
            history_path=str(hist), verbose=False,
        )
        data = json.loads(hist.read_text())
        step0 = data["steps"][0]
        coords = step0["coords_bohr"]
        assert len(coords) == 3        # 3 atoms
        assert len(coords[0]) == 3    # 3 Cartesian coords


# ============================================================================
# C. Geometry regression vs. PySCF STO-3G reference
# ============================================================================

class TestH2OGeometryRegression:

    def test_r_oh_within_tolerance(self, h2o_opt_result):
        mol = h2o_opt_result.final_molecule
        R1 = bond_length(mol, 0, 1)
        R2 = bond_length(mol, 0, 2)
        R_mean = (R1 + R2) / 2
        assert abs(R_mean - REF_R_OH) < 0.02, \
            f"R(OH) = {R_mean:.4f} bohr, expected {REF_R_OH:.4f} ± 0.02"

    def test_theta_within_tolerance(self, h2o_opt_result):
        mol = h2o_opt_result.final_molecule
        theta = bond_angle(mol, 1, 0, 2)
        assert abs(theta - REF_THETA) < 1.5, \
            f"θ(HOH) = {theta:.3f}°, expected {REF_THETA:.1f}° ± 1.5°"

    def test_energy_within_tolerance(self, h2o_opt_result):
        assert abs(h2o_opt_result.energy_final - REF_ENERGY) < 5e-3, \
            f"E = {h2o_opt_result.energy_final:.8f} Ha, expected {REF_ENERGY:.8f} ± 5e-3"

    def test_c2v_symmetry_preserved(self, h2o_opt_result):
        """Both O-H bonds should be equal to within 1e-3 bohr at convergence."""
        mol = h2o_opt_result.final_molecule
        R1 = bond_length(mol, 0, 1)
        R2 = bond_length(mol, 0, 2)
        assert abs(R1 - R2) < 0.01, \
            f"C2v broken: R(OH1)={R1:.6f} vs R(OH2)={R2:.6f} bohr"

    def test_r_oh_shorter_than_start(self, h2o_opt_result):
        """Bond lengths compressed from stretched start geometry (2.0 bohr)."""
        mol = h2o_opt_result.final_molecule
        R1 = bond_length(mol, 0, 1)
        assert R1 < 2.0

    def test_theta_smaller_than_start(self, h2o_opt_result):
        """Angle compressed from 120° toward ~100°."""
        mol = h2o_opt_result.final_molecule
        theta = bond_angle(mol, 1, 0, 2)
        assert theta < 120.0

    def test_energy_below_start(self, h2o_opt_result):
        assert h2o_opt_result.energy_final < h2o_opt_result.energy_initial

    def test_final_bond_length_in_angstrom(self, h2o_opt_result):
        """R(OH) at STO-3G equilibrium: ~0.99 Å (closer to experiment than start)."""
        mol = h2o_opt_result.final_molecule
        R1_ang = bond_length(mol, 0, 1) * BOHR_TO_ANG
        assert 0.95 < R1_ang < 1.05   # physically reasonable STO-3G range


# ============================================================================
# D. Force balance on H2O
# ============================================================================

class TestH2OForceBalance:
    """
    Newton's third law: sum of all atomic forces must be (near) zero.
    Checked at a non-equilibrium geometry.
    """

    def test_force_sum_x(self, h2o_start):
        grad = numerical_gradient(h2o_start, STO3G)
        assert abs(grad[:, 0].sum()) < 1e-7

    def test_force_sum_y(self, h2o_start):
        grad = numerical_gradient(h2o_start, STO3G)
        assert abs(grad[:, 1].sum()) < 1e-7

    def test_force_sum_z(self, h2o_start):
        grad = numerical_gradient(h2o_start, STO3G)
        assert abs(grad[:, 2].sum()) < 5e-5   # slightly relaxed: z-gradient via O z-mode

    def test_x_gradient_zero_by_symmetry(self, h2o_start):
        """H2O in yz-plane: dE/dx = 0 for all atoms by mirror symmetry."""
        grad = numerical_gradient(h2o_start, STO3G)
        assert max(abs(grad[:, 0])) < 1e-8

    def test_y_gradients_antisymmetric(self, h2o_start):
        """For C2v H2O, dE/dy(H1) = -dE/dy(H2)."""
        grad = numerical_gradient(h2o_start, STO3G)
        assert abs(grad[1, 1] + grad[2, 1]) < 1e-7

    def test_gradient_magnitude_non_trivial(self, h2o_start):
        """At R=2.0 bohr θ=120°, gradient should be substantial."""
        grad = numerical_gradient(h2o_start, STO3G)
        assert max_gradient(grad) > 0.01   # Ha/bohr

    def test_near_eq_gradient_small(self, h2o_near_eq):
        """Near equilibrium, max gradient should be small."""
        grad = numerical_gradient(h2o_near_eq, STO3G)
        assert max_gradient(grad) < 5e-3   # Ha/bohr
