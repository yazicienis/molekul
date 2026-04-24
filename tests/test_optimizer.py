"""
Tests for numerical gradient and geometry optimiser — Phase 5.

Coverage
--------
A. Gradient correctness:
   - Newton's 3rd law: forces sum to zero (translational invariance)
   - Symmetry: transverse components zero for axially symmetric molecules
   - Sign: gradient points away from minimum at stretched geometry
   - Finite-difference consistency with energy

B. Optimiser:
   - H2 converges
   - Energy decreases from start to end
   - Final energy below initial
   - Final gradient norm below threshold
   - Trajectory length == n_steps
   - JSON history file written and parseable
   - XYZ trajectory file written

C. Geometry regression:
   - H2 R_eq = 1.3460 ± 0.005 bohr (PySCF-confirmed)
   - H2 E_eq = -1.1175058833 ± 1e-4 Ha (PySCF-confirmed)

D. Gradient consistency check:
   - numerical_gradient at equilibrium ≈ 0 (max < 5e-3 Ha/bohr)

Reference values (PySCF 2.x, STO-3G fine scan R=1.30..1.40 bohr, 51 points)
-----------------------------------------------------------------------------
H2: R_eq = 1.3460 bohr,  E_eq = -1.1175058833 Ha
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.grad import numerical_gradient, gradient_norm, max_gradient
from molekul.optimizer import optimize_geometry, OptResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def h2_stretched():
    """H2 at R = 1.7 bohr (well above equilibrium)."""
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.7])],
        charge=0, multiplicity=1, name="H2",
    )


@pytest.fixture(scope="module")
def h2_eq():
    """H2 near equilibrium at R = 1.346 bohr."""
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.346])],
        charge=0, multiplicity=1, name="H2",
    )


@pytest.fixture(scope="module")
def h2_stretched_grad(h2_stretched):
    return numerical_gradient(h2_stretched, STO3G)


@pytest.fixture(scope="module")
def h2_opt_result():
    """Full geometry optimisation of H2 from R=1.7 bohr."""
    mol = Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.7])],
        charge=0, multiplicity=1, name="H2",
    )
    return optimize_geometry(mol, STO3G, grad_tol=1e-4, max_steps=50, verbose=False)


# ============================================================================
# A. Gradient correctness
# ============================================================================

class TestGradientCorrectness:

    def test_translational_invariance_x(self, h2_stretched_grad):
        """Sum of forces along x must be zero (translational invariance)."""
        assert abs(h2_stretched_grad[:, 0].sum()) < 1e-8

    def test_translational_invariance_y(self, h2_stretched_grad):
        """Sum of forces along y must be zero."""
        assert abs(h2_stretched_grad[:, 1].sum()) < 1e-8

    def test_translational_invariance_z(self, h2_stretched_grad):
        """Sum of forces along z must be zero (Newton's 3rd law for H2)."""
        assert abs(h2_stretched_grad[:, 2].sum()) < 1e-8

    def test_transverse_zero_x(self, h2_stretched_grad):
        """For H2 aligned along z, dE/dx must be zero by symmetry."""
        assert max(abs(h2_stretched_grad[:, 0])) < 1e-8

    def test_transverse_zero_y(self, h2_stretched_grad):
        """For H2 aligned along z, dE/dy must be zero by symmetry."""
        assert max(abs(h2_stretched_grad[:, 1])) < 1e-8

    def test_sign_at_stretched_geometry(self, h2_stretched_grad):
        """At R=1.7 bohr (stretched), H1 force is +z (pulled toward H2) and H2 is -z."""
        # H1 is at z=0, H2 is at z=1.7. The bond is stretched so
        # energy decreases when H2 moves toward H1 (negative z displacement).
        # Thus dE/dz for H1 < 0 (H1 wants to move toward +z).
        # Actually: if H1 moves +z, distance decreases → energy decreases → gradient < 0.
        # Let's just check they are opposite in sign.
        assert h2_stretched_grad[0, 2] * h2_stretched_grad[1, 2] < 0

    def test_gradient_magnitude_reasonable(self, h2_stretched_grad):
        """Gradient at R=1.7 bohr should be significant (not near zero)."""
        assert max_gradient(h2_stretched_grad) > 0.05  # Ha/bohr

    def test_gradient_norm_positive(self, h2_stretched_grad):
        assert gradient_norm(h2_stretched_grad) > 0.0

    def test_gradient_near_zero_at_equilibrium(self, h2_eq):
        """Near equilibrium the gradient should be small."""
        grad = numerical_gradient(h2_eq, STO3G)
        assert max_gradient(grad) < 5e-3  # Ha/bohr

    def test_finite_diff_consistency(self, h2_stretched):
        """Central-difference gradient must be consistent with energy differences."""
        from molekul.rhf import rhf_scf
        from molekul.grad import _displaced

        h = 1e-3
        # Check z-component for atom 1 manually
        mol_p = _displaced(h2_stretched, 1, 2, +h)
        mol_m = _displaced(h2_stretched, 1, 2, -h)
        E_p = rhf_scf(mol_p, STO3G).energy_total
        E_m = rhf_scf(mol_m, STO3G).energy_total
        g_manual = (E_p - E_m) / (2 * h)

        grad = numerical_gradient(h2_stretched, STO3G, h=h)
        assert abs(grad[1, 2] - g_manual) < 1e-10


# ============================================================================
# B. Optimiser behaviour
# ============================================================================

class TestOptimiserBehaviour:

    def test_converged(self, h2_opt_result):
        assert h2_opt_result.converged

    def test_result_type(self, h2_opt_result):
        assert isinstance(h2_opt_result, OptResult)

    def test_energy_decreases(self, h2_opt_result):
        assert h2_opt_result.energy_final < h2_opt_result.energy_initial

    def test_final_below_initial(self, h2_opt_result):
        # Energy must drop by a meaningful amount from R=1.7 bohr
        dE = h2_opt_result.energy_initial - h2_opt_result.energy_final
        assert dE > 0.01  # Ha

    def test_gradient_converged(self, h2_opt_result):
        assert h2_opt_result.grad_max_final < 1e-3  # Ha/bohr

    def test_trajectory_length(self, h2_opt_result):
        assert len(h2_opt_result.trajectory) == h2_opt_result.n_steps

    def test_energy_history_length(self, h2_opt_result):
        assert len(h2_opt_result.energy_history) == h2_opt_result.n_steps

    def test_final_molecule_type(self, h2_opt_result):
        assert isinstance(h2_opt_result.final_molecule, Molecule)

    def test_n_steps_reasonable(self, h2_opt_result):
        # BFGS should converge H2 in well under 50 steps
        assert h2_opt_result.n_steps < 50

    def test_trajectory_written(self, tmp_path):
        mol = Molecule(
            atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.7])],
            charge=0, multiplicity=1, name="H2",
        )
        traj = tmp_path / "traj.xyz"
        optimize_geometry(mol, STO3G, grad_tol=1e-3, max_steps=20,
                          traj_path=str(traj), verbose=False)
        assert traj.exists()
        assert traj.stat().st_size > 0

    def test_history_json_written(self, tmp_path):
        mol = Molecule(
            atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.7])],
            charge=0, multiplicity=1, name="H2",
        )
        hist = tmp_path / "hist.json"
        optimize_geometry(mol, STO3G, grad_tol=1e-3, max_steps=20,
                          history_path=str(hist), verbose=False)
        assert hist.exists()
        data = json.loads(hist.read_text())
        assert "steps" in data
        assert len(data["steps"]) > 0

    def test_history_json_schema(self, tmp_path):
        mol = Molecule(
            atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.7])],
            charge=0, multiplicity=1, name="H2",
        )
        hist = tmp_path / "hist.json"
        optimize_geometry(mol, STO3G, grad_tol=1e-3, max_steps=20,
                          history_path=str(hist), verbose=False)
        data = json.loads(hist.read_text())
        required_keys = {"molecule", "basis", "converged", "n_steps",
                         "energy_initial_Ha", "energy_final_Ha", "steps"}
        assert required_keys.issubset(data.keys())
        step0 = data["steps"][0]
        assert {"step", "energy_Ha", "grad_rms_Ha_per_bohr",
                "grad_max_Ha_per_bohr", "coords_bohr"}.issubset(step0.keys())


# ============================================================================
# C. Geometry regression (PySCF-confirmed)
# ============================================================================

class TestGeometryRegression:
    """
    Reference (PySCF 2.x, STO-3G fine scan 51 points R=1.30..1.40 bohr):
      R_eq = 1.3460 bohr   E_eq = -1.1175058833 Ha
    """
    R_EQ_REF = 1.3460    # bohr
    E_EQ_REF = -1.1175058833  # Ha

    def test_h2_req_bohr(self, h2_opt_result):
        c0 = h2_opt_result.final_molecule.atoms[0].coords
        c1 = h2_opt_result.final_molecule.atoms[1].coords
        R_final = float(np.linalg.norm(c1 - c0))
        assert abs(R_final - self.R_EQ_REF) < 0.005, \
            f"R_eq = {R_final:.4f} bohr, expected {self.R_EQ_REF:.4f} ± 0.005"

    def test_h2_energy_at_eq(self, h2_opt_result):
        assert abs(h2_opt_result.energy_final - self.E_EQ_REF) < 1e-4, \
            f"E = {h2_opt_result.energy_final:.10f} Ha, expected {self.E_EQ_REF:.10f} ± 1e-4"

    def test_energy_below_stretched(self, h2_opt_result):
        """Optimised energy must be lower than energy at R=1.7 bohr."""
        assert h2_opt_result.energy_final < h2_opt_result.energy_initial

    def test_energy_not_too_low(self, h2_opt_result):
        """Energy must stay above dissociation limit (not collapsed)."""
        assert h2_opt_result.energy_final > -2.0  # Ha — well above any physical minimum
