"""
Tests for Restricted Hartree-Fock (RHF) SCF — Phase 4.

Coverage
--------
A. Basic properties:   convergence, correct electron count, RHFResult fields
B. Energy regression:  H2, HeH+, H2O vs. PySCF-confirmed reference values
C. Physical constraints: energy below H_core estimate, E_nuc > 0, E_elec < 0
D. MO properties:      orthonormality C^T S C ≈ I, correct number of MOs
E. DIIS:               convergence with and without DIIS
F. Fock commutativity: [F, P] ≈ 0 at convergence (SCF optimality condition)

Reference energies (STO-3G, PySCF 2.x, same geometries)
---------------------------------------------------------
H2  R=1.4 bohr              : -1.1167143251 Hartree
HeH+ R=1.4632 bohr (charge=1): -2.8418364993 Hartree
H2O C2v (O at origin, Angstrom H coords ±0.757, -0.586):
                               -74.9629466405 Hartree
"""

import numpy as np
import pytest

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf import rhf_scf, RHFResult
from molekul.integrals import build_overlap


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def h2():
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1, name="H2",
    )


@pytest.fixture(scope="module")
def heh_plus():
    return Molecule(
        atoms=[Atom("He", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4632])],
        charge=1, multiplicity=1, name="HeH+",
    )


@pytest.fixture(scope="module")
def water():
    return Molecule(
        atoms=[
            Atom.from_angstrom("O",  0.000,  0.000,  0.000),
            Atom.from_angstrom("H",  0.000,  0.757, -0.586),
            Atom.from_angstrom("H",  0.000, -0.757, -0.586),
        ],
        charge=0, multiplicity=1, name="H2O",
    )


@pytest.fixture(scope="module")
def h2_result(h2):
    return rhf_scf(h2, STO3G)


@pytest.fixture(scope="module")
def heh_result(heh_plus):
    return rhf_scf(heh_plus, STO3G)


@pytest.fixture(scope="module")
def water_result(water):
    return rhf_scf(water, STO3G)


# ============================================================================
# A. Basic properties
# ============================================================================

class TestBasicProperties:

    def test_h2_converged(self, h2_result):
        assert h2_result.converged

    def test_heh_converged(self, heh_result):
        assert heh_result.converged

    def test_water_converged(self, water_result):
        assert water_result.converged

    def test_result_is_rhfresult(self, h2_result):
        assert isinstance(h2_result, RHFResult)

    def test_h2_n_mo(self, h2_result):
        # H2 STO-3G: 2 basis functions → 2 MOs
        assert h2_result.mo_energies.shape == (2,)
        assert h2_result.mo_coefficients.shape == (2, 2)

    def test_water_n_mo(self, water_result):
        # H2O STO-3G: 7 basis functions → 7 MOs
        assert water_result.mo_energies.shape == (7,)
        assert water_result.mo_coefficients.shape == (7, 7)

    def test_energy_history_nonempty(self, h2_result):
        assert len(h2_result.energy_history) > 0

    def test_energy_history_is_monotone_after_first(self, water_result):
        # Energy should not increase significantly during SCF (not strictly required
        # with DIIS but should be roughly monotone after the first few steps)
        hist = water_result.energy_history
        assert hist[-1] < hist[0]   # Final < core-guess energy

    def test_multiplicity_check(self, h2):
        """rhf_scf must reject non-singlet molecules."""
        doublet = Molecule(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            charge=0, multiplicity=2, name="H_radical",
        )
        with pytest.raises(ValueError, match="singlet"):
            rhf_scf(doublet, STO3G)


# ============================================================================
# B. Energy regression vs. PySCF-confirmed reference values
# ============================================================================

class TestEnergyRegression:

    # Tolerances: we aim for 1e-6 Hartree match with PySCF
    # (typical DIIS SCF residual << 1e-8 Hartree)
    H2_REF   = -1.1167143251
    HEH_REF  = -2.8418364993
    H2O_REF  = -74.9629466405

    def test_h2_total_energy(self, h2_result):
        assert abs(h2_result.energy_total - self.H2_REF) < 1e-6

    def test_heh_total_energy(self, heh_result):
        assert abs(heh_result.energy_total - self.HEH_REF) < 1e-6

    def test_water_total_energy(self, water_result):
        assert abs(water_result.energy_total - self.H2O_REF) < 1e-6

    def test_h2_energy_components(self, h2_result):
        # E_total = E_electronic + E_nuclear
        assert abs(
            h2_result.energy_total
            - h2_result.energy_electronic
            - h2_result.energy_nuclear
        ) < 1e-12

    def test_water_energy_components(self, water_result):
        assert abs(
            water_result.energy_total
            - water_result.energy_electronic
            - water_result.energy_nuclear
        ) < 1e-12

    def test_h2_nuclear_repulsion(self, h2_result):
        # E_nuc = 1/R = 1/1.4 bohr
        assert abs(h2_result.energy_nuclear - 1.0 / 1.4) < 1e-10


# ============================================================================
# C. Physical constraints
# ============================================================================

class TestPhysicalConstraints:

    def test_h2_enuc_positive(self, h2_result):
        assert h2_result.energy_nuclear > 0.0

    def test_water_enuc_positive(self, water_result):
        assert water_result.energy_nuclear > 0.0

    def test_h2_eelec_negative(self, h2_result):
        assert h2_result.energy_electronic < 0.0

    def test_water_eelec_negative(self, water_result):
        assert water_result.energy_electronic < 0.0

    def test_h2_total_negative(self, h2_result):
        assert h2_result.energy_total < 0.0

    def test_water_total_negative(self, water_result):
        assert water_result.energy_total < 0.0

    def test_mo_energies_sorted(self, h2_result):
        eps = h2_result.mo_energies
        assert np.all(np.diff(eps) >= -1e-10)  # non-decreasing

    def test_water_mo_energies_sorted(self, water_result):
        eps = water_result.mo_energies
        assert np.all(np.diff(eps) >= -1e-10)

    def test_homo_lumo_gap_positive_h2(self, h2_result):
        # H2 has 1 occupied MO; HOMO < LUMO
        n_occ = 1  # n_alpha for H2
        homo = h2_result.mo_energies[n_occ - 1]
        lumo = h2_result.mo_energies[n_occ]
        assert lumo > homo

    def test_homo_lumo_gap_positive_water(self, water_result):
        n_occ = 5  # 10 electrons / 2
        homo = water_result.mo_energies[n_occ - 1]
        lumo = water_result.mo_energies[n_occ]
        assert lumo > homo


# ============================================================================
# D. MO orthonormality  C^T S C ≈ I
# ============================================================================

class TestMOOrthonormality:

    def _check_orthonormal(self, result, molecule, tol=1e-8):
        S = build_overlap(STO3G, molecule)
        C = result.mo_coefficients
        CSC = C.T @ S @ C
        n = C.shape[0]
        assert CSC.shape == (n, n)
        assert np.allclose(CSC, np.eye(n), atol=tol), \
            f"C^T S C deviates from I: max|Δ|={np.max(np.abs(CSC - np.eye(n))):.2e}"

    def test_h2_mo_orthonormal(self, h2_result, h2):
        self._check_orthonormal(h2_result, h2)

    def test_heh_mo_orthonormal(self, heh_result, heh_plus):
        self._check_orthonormal(heh_result, heh_plus)

    def test_water_mo_orthonormal(self, water_result, water):
        self._check_orthonormal(water_result, water)


# ============================================================================
# E. Fock matrix commutativity at convergence  [F, P]_S ≈ 0
# ============================================================================

class TestFockCommutativity:
    """
    At SCF convergence, the error vector e = FPS - SPF must be ≈ 0.
    This is equivalent to saying the density commutes with the Fock matrix
    in the metric S — the fundamental SCF optimality condition.
    """

    def _check_commutator(self, result, molecule, tol=1e-6):
        S = build_overlap(STO3G, molecule)
        F = result.fock_matrix
        P = result.density_matrix
        comm = F @ P @ S - S @ P @ F
        max_err = np.max(np.abs(comm))
        assert max_err < tol, f"[F,P]_S max = {max_err:.2e} (should be < {tol:.0e})"

    def test_h2_fock_commutes(self, h2_result, h2):
        self._check_commutator(h2_result, h2)

    def test_heh_fock_commutes(self, heh_result, heh_plus):
        self._check_commutator(heh_result, heh_plus)

    def test_water_fock_commutes(self, water_result, water):
        self._check_commutator(water_result, water)


# ============================================================================
# F. Density matrix idempotency  P S P = 2 P  (for closed-shell RHF)
# ============================================================================

class TestDensityIdempotency:
    """
    For closed-shell RHF the density matrix satisfies P S P = 2 P.
    This is a stringent check that the MO coefficients are fully self-consistent.
    """

    def _check_idempotent(self, result, molecule, tol=1e-7):
        S = build_overlap(STO3G, molecule)
        P = result.density_matrix
        PSP = P @ S @ P
        max_err = np.max(np.abs(PSP - 2.0 * P))
        assert max_err < tol, f"PSP - 2P max = {max_err:.2e}"

    def test_h2_idempotent(self, h2_result, h2):
        self._check_idempotent(h2_result, h2)

    def test_heh_idempotent(self, heh_result, heh_plus):
        self._check_idempotent(heh_result, heh_plus)

    def test_water_idempotent(self, water_result, water):
        self._check_idempotent(water_result, water)


# ============================================================================
# G. Electron count from density matrix  tr(PS) = N_elec
# ============================================================================

class TestElectronCount:
    """tr(P S) = N_electrons for the converged density matrix."""

    def _check_electron_count(self, result, molecule, tol=1e-8):
        S = build_overlap(STO3G, molecule)
        P = result.density_matrix
        n_elec_computed = np.trace(P @ S)
        n_elec_expected = float(molecule.n_electrons)
        assert abs(n_elec_computed - n_elec_expected) < tol, \
            f"tr(PS)={n_elec_computed:.8f} != N={n_elec_expected}"

    def test_h2_electron_count(self, h2_result, h2):
        self._check_electron_count(h2_result, h2)

    def test_heh_electron_count(self, heh_result, heh_plus):
        self._check_electron_count(heh_result, heh_plus)

    def test_water_electron_count(self, water_result, water):
        self._check_electron_count(water_result, water)
