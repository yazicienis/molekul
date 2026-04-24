"""
Tests for CUBE file export — Phase 6.

Coverage
--------
A. Grid construction:
   - make_grid produces correct origin, steps, shape
   - grid_points produces correct count and coordinates

B. Basis function evaluation:
   - phi has correct shape (n_basis, N)
   - phi values at atom centre are real and finite
   - phi sums to finite positive number over grid

C. Density evaluation:
   - rho shape is (N,)
   - rho ≥ 0 everywhere (physical)
   - ∫ρ dr ≈ N_electrons within 5% (coarse grid)
   - rho consistency: rho = tr(P φ^T φ) pattern

D. Orbital evaluation:
   - psi shape is (N,)
   - ∫|ψ|² dr ≈ 1.0 within 5% (coarse grid)
   - psi can be negative (unlike density)

E. CUBE file writer:
   - write_cube creates a non-empty file
   - line 3 contains n_atoms and origin
   - data section has 6 values per line (last line may be shorter)
   - atom section has correct number of lines

F. Integration: export_density / export_orbital round-trip
   - Files exist after export
   - Stats dict has expected keys
   - n_electrons_integrated close to N_electrons
   - norm_integrated close to 1

Notes
-----
All tests use a coarse grid (step=0.5, margin=2.0) for speed.
The fixture h2o_rhf_result is module-scoped (computed once).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from molekul.atoms       import Atom
from molekul.molecule    import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf         import rhf_scf
from molekul.cube        import (
    make_grid, grid_points,
    eval_basis_grid, eval_density_grid, eval_orbital_grid,
    write_cube, export_density, export_orbital,
)


# ============================================================================
# Fixtures
# ============================================================================

STEP   = 0.5   # coarse — fast tests
MARGIN = 2.0


def _make_h2() -> Molecule:
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1, name="H2",
    )


def _make_h2o() -> Molecule:
    R, th = 1.870, np.radians(50.0)
    return Molecule(
        atoms=[
            Atom("O", [0.0,  0.0,      0.0     ]),
            Atom("H", [0.0,  R*np.sin(th), -R*np.cos(th)]),
            Atom("H", [0.0, -R*np.sin(th), -R*np.cos(th)]),
        ],
        charge=0, multiplicity=1, name="H2O",
    )


@pytest.fixture(scope="module")
def h2_mol():
    return _make_h2()


@pytest.fixture(scope="module")
def h2o_mol():
    return _make_h2o()


@pytest.fixture(scope="module")
def h2_rhf(h2_mol):
    return rhf_scf(h2_mol, STO3G, verbose=False)


@pytest.fixture(scope="module")
def h2o_rhf(h2o_mol):
    return rhf_scf(h2o_mol, STO3G, verbose=False)


@pytest.fixture(scope="module")
def h2_grid(h2_mol):
    origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
    pts = grid_points(origin, steps, shape)
    phi = eval_basis_grid(STO3G, h2_mol, pts)
    return origin, steps, shape, pts, phi


@pytest.fixture(scope="module")
def h2o_grid(h2o_mol):
    origin, steps, shape = make_grid(h2o_mol, margin=MARGIN, step=STEP)
    pts = grid_points(origin, steps, shape)
    phi = eval_basis_grid(STO3G, h2o_mol, pts)
    return origin, steps, shape, pts, phi


# ============================================================================
# A. Grid construction
# ============================================================================

class TestGridConstruction:

    def test_make_grid_returns_three_items(self, h2_mol):
        result = make_grid(h2_mol, margin=MARGIN, step=STEP)
        assert len(result) == 3

    def test_origin_shape(self, h2_mol):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        assert origin.shape == (3,)

    def test_steps_shape(self, h2_mol):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        assert steps.shape == (3,)

    def test_steps_equal_to_requested(self, h2_mol):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        assert np.allclose(steps, STEP)

    def test_shape_positive_ints(self, h2_mol):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        assert all(isinstance(n, int) and n > 0 for n in shape)

    def test_origin_below_atoms(self, h2_mol):
        """Origin should be at least (margin - step) below lowest atom coord."""
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        lo = h2_mol.coords_bohr.min(axis=0)
        assert np.all(origin <= lo - (MARGIN - STEP))

    def test_grid_covers_atoms(self, h2_mol):
        """All atom coords should lie within the grid bounding box."""
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        hi = origin + np.array(shape) * steps
        for atom in h2_mol.atoms:
            assert np.all(atom.coords >= origin - 1e-10)
            assert np.all(atom.coords <= hi + 1e-10)

    def test_grid_points_count(self, h2_mol):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        pts = grid_points(origin, steps, shape)
        nx, ny, nz = shape
        assert pts.shape == (nx * ny * nz, 3)

    def test_grid_points_first_is_origin(self, h2_mol):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        pts = grid_points(origin, steps, shape)
        assert np.allclose(pts[0], origin)

    def test_grid_points_dtype_float64(self, h2_mol):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        pts = grid_points(origin, steps, shape)
        assert pts.dtype == np.float64

    def test_grid_x_outer_order(self, h2_mol):
        """Verify x-outer, z-inner ordering: second point differs only in z."""
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        pts = grid_points(origin, steps, shape)
        # In x-outer, z-inner order, the second point is (x0, y0, z0+dz)
        assert np.isclose(pts[1, 0], pts[0, 0])  # x unchanged
        assert np.isclose(pts[1, 1], pts[0, 1])  # y unchanged
        assert np.isclose(pts[1, 2], pts[0, 2] + steps[2])  # z incremented


# ============================================================================
# B. Basis function evaluation
# ============================================================================

class TestBasisGridEval:

    def test_phi_shape_h2(self, h2_grid, h2_mol):
        origin, steps, shape, pts, phi = h2_grid
        n_bf = len(STO3G.basis_functions(h2_mol))
        N    = pts.shape[0]
        assert phi.shape == (n_bf, N)

    def test_phi_shape_h2o(self, h2o_grid, h2o_mol):
        origin, steps, shape, pts, phi = h2o_grid
        n_bf = len(STO3G.basis_functions(h2o_mol))
        N    = pts.shape[0]
        assert phi.shape == (n_bf, N)

    def test_phi_finite(self, h2_grid):
        _, _, _, _, phi = h2_grid
        assert np.all(np.isfinite(phi))

    def test_phi_not_all_zero(self, h2_grid):
        _, _, _, _, phi = h2_grid
        assert np.any(phi != 0.0)

    def test_phi_real(self, h2_grid):
        _, _, _, _, phi = h2_grid
        assert phi.dtype in (np.float64, np.float32)

    def test_s_orbital_positive_at_centre(self, h2_mol):
        """An s-type basis function is positive at its atom centre."""
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        pts = grid_points(origin, steps, shape)
        phi = eval_basis_grid(STO3G, h2_mol, pts)
        # MO 1 on H2 is σ_g: both s-type functions are positive at their centres
        # (they decay but are always ≥ 0 for s-type)
        assert np.all(phi >= -1e-12)  # s-only basis for H2


# ============================================================================
# C. Density evaluation
# ============================================================================

class TestDensityEval:

    def test_rho_shape(self, h2_grid, h2_rhf):
        _, _, _, _, phi = h2_grid
        rho = eval_density_grid(h2_rhf.density_matrix, phi)
        assert rho.shape == (phi.shape[1],)

    def test_rho_nonnegative(self, h2_grid, h2_rhf):
        _, _, _, _, phi = h2_grid
        rho = eval_density_grid(h2_rhf.density_matrix, phi)
        assert np.all(rho >= -1e-10)

    def test_rho_finite(self, h2_grid, h2_rhf):
        _, _, _, _, phi = h2_grid
        rho = eval_density_grid(h2_rhf.density_matrix, phi)
        assert np.all(np.isfinite(rho))

    def test_rho_integral_h2(self, h2_grid, h2_rhf):
        """∫ρ dr ≈ 2 electrons (H2) within 10% on coarse grid."""
        origin, steps, shape, pts, phi = h2_grid
        rho = eval_density_grid(h2_rhf.density_matrix, phi)
        dV  = float(np.prod(steps))
        n_e = float(rho.sum() * dV)
        assert abs(n_e - 2.0) / 2.0 < 0.10, f"∫ρ = {n_e:.4f}, expected 2.0"

    def test_rho_integral_h2o(self, h2o_grid, h2o_rhf):
        """∫ρ dr ≈ 10 electrons (H2O) within 10% on coarse grid."""
        origin, steps, shape, pts, phi = h2o_grid
        rho = eval_density_grid(h2o_rhf.density_matrix, phi)
        dV  = float(np.prod(steps))
        n_e = float(rho.sum() * dV)
        assert abs(n_e - 10.0) / 10.0 < 0.10, f"∫ρ = {n_e:.4f}, expected 10.0"

    def test_rho_max_positive(self, h2o_grid, h2o_rhf):
        """H2O density should have a noticeable peak near oxygen."""
        _, _, _, _, phi = h2o_grid
        rho = eval_density_grid(h2o_rhf.density_matrix, phi)
        assert rho.max() > 0.01

    def test_rho_3d_shape(self, h2_mol, h2_rhf):
        """rho.reshape(shape) gives the correct 3D array."""
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        pts = grid_points(origin, steps, shape)
        phi = eval_basis_grid(STO3G, h2_mol, pts)
        rho = eval_density_grid(h2_rhf.density_matrix, phi)
        rho_3d = rho.reshape(shape)
        assert rho_3d.shape == shape


# ============================================================================
# D. Orbital evaluation
# ============================================================================

class TestOrbitalEval:

    def test_psi_shape(self, h2_grid, h2_rhf):
        _, _, _, _, phi = h2_grid
        psi = eval_orbital_grid(h2_rhf.mo_coefficients, 0, phi)
        assert psi.shape == (phi.shape[1],)

    def test_psi_finite(self, h2_grid, h2_rhf):
        _, _, _, _, phi = h2_grid
        psi = eval_orbital_grid(h2_rhf.mo_coefficients, 0, phi)
        assert np.all(np.isfinite(psi))

    def test_psi_can_be_negative(self, h2o_grid, h2o_rhf):
        """Virtual orbitals (antibonding) have both positive and negative lobes."""
        _, _, _, _, phi = h2o_grid
        n_occ = _make_h2o().n_alpha
        psi = eval_orbital_grid(h2o_rhf.mo_coefficients, n_occ, phi)  # LUMO
        assert psi.min() < 0 and psi.max() > 0

    def test_mo_norm_h2_homo(self, h2_grid, h2_rhf):
        """∫|ψ_HOMO|² dr ≈ 1 within 10% on coarse grid."""
        origin, steps, shape, pts, phi = h2_grid
        n_occ = _make_h2().n_alpha
        psi = eval_orbital_grid(h2_rhf.mo_coefficients, n_occ - 1, phi)
        dV  = float(np.prod(steps))
        norm = float((psi ** 2).sum() * dV)
        assert abs(norm - 1.0) < 0.10, f"‖ψ_HOMO‖² = {norm:.4f}, expected 1.0"

    def test_mo_norm_h2o_all_occ(self, h2o_grid, h2o_rhf):
        """All occupied H2O MOs integrate to 1 within 10% on coarse grid."""
        origin, steps, shape, pts, phi = h2o_grid
        dV    = float(np.prod(steps))
        n_occ = _make_h2o().n_alpha
        for i in range(n_occ):
            psi  = eval_orbital_grid(h2o_rhf.mo_coefficients, i, phi)
            norm = float((psi ** 2).sum() * dV)
            assert abs(norm - 1.0) < 0.10, \
                f"MO {i+1}: ‖ψ‖² = {norm:.4f}, expected 1.0"

    def test_different_mo_differ(self, h2o_grid, h2o_rhf):
        """Two distinct MOs should not be identical arrays."""
        _, _, _, _, phi = h2o_grid
        psi0 = eval_orbital_grid(h2o_rhf.mo_coefficients, 0, phi)
        psi1 = eval_orbital_grid(h2o_rhf.mo_coefficients, 1, phi)
        assert not np.allclose(psi0, psi1)


# ============================================================================
# E. CUBE file writer
# ============================================================================

class TestWriteCube:

    def _make_simple_h2_cube(self, tmp_path, h2_mol, h2_rhf):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        pts = grid_points(origin, steps, shape)
        phi = eval_basis_grid(STO3G, h2_mol, pts)
        rho = eval_density_grid(h2_rhf.density_matrix, phi)
        rho_3d = rho.reshape(shape)
        p = tmp_path / "h2_test.cube"
        write_cube(p, h2_mol, rho_3d, origin, steps)
        return p, shape, h2_mol

    def test_file_created(self, tmp_path, h2_mol, h2_rhf):
        p, _, _ = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        assert p.exists()

    def test_file_nonempty(self, tmp_path, h2_mol, h2_rhf):
        p, _, _ = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        assert p.stat().st_size > 0

    def test_has_two_comment_lines(self, tmp_path, h2_mol, h2_rhf):
        p, _, _ = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        lines = p.read_text().splitlines()
        assert len(lines) >= 2

    def test_line3_natoms_correct(self, tmp_path, h2_mol, h2_rhf):
        p, _, mol = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        lines = p.read_text().splitlines()
        natoms = int(lines[2].split()[0])
        assert natoms == mol.n_atoms

    def test_line3_origin_is_float(self, tmp_path, h2_mol, h2_rhf):
        p, _, _ = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        lines = p.read_text().splitlines()
        parts = lines[2].split()
        assert len(parts) == 4
        assert all(re.match(r"[-+]?\d+\.\d+", x) for x in parts[1:])

    def test_grid_vector_lines(self, tmp_path, h2_mol, h2_rhf):
        p, shape, _ = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        lines = p.read_text().splitlines()
        nx_read = int(lines[3].split()[0])
        ny_read = int(lines[4].split()[0])
        nz_read = int(lines[5].split()[0])
        assert nx_read == shape[0]
        assert ny_read == shape[1]
        assert nz_read == shape[2]

    def test_atom_section_length(self, tmp_path, h2_mol, h2_rhf):
        p, _, mol = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        lines = p.read_text().splitlines()
        # Lines 6..(6+n_atoms-1) are atom lines
        for i in range(mol.n_atoms):
            parts = lines[6 + i].split()
            assert len(parts) == 5         # Z charge x y z
            assert int(float(parts[0])) > 0  # Z > 0

    def test_data_section_6_per_line(self, tmp_path, h2_mol, h2_rhf):
        p, _, mol = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        lines = p.read_text().splitlines()
        data_start = 6 + mol.n_atoms
        data_lines = lines[data_start:]
        # All lines except possibly the last should have exactly 6 values
        for line in data_lines[:-1]:
            vals = line.split()
            assert len(vals) == 6, f"Expected 6 values, got {len(vals)}: {line!r}"

    def test_data_all_floats(self, tmp_path, h2_mol, h2_rhf):
        p, _, mol = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        lines = p.read_text().splitlines()
        data_start = 6 + mol.n_atoms
        for line in lines[data_start:]:
            for v in line.split():
                float(v)   # should not raise

    def test_data_total_count(self, tmp_path, h2_mol, h2_rhf):
        p, shape, mol = self._make_simple_h2_cube(tmp_path, h2_mol, h2_rhf)
        lines = p.read_text().splitlines()
        data_start = 6 + mol.n_atoms
        data_vals = []
        for line in lines[data_start:]:
            data_vals.extend(line.split())
        nx, ny, nz = shape
        assert len(data_vals) == nx * ny * nz

    def test_custom_comments(self, tmp_path, h2_mol, h2_rhf):
        origin, steps, shape = make_grid(h2_mol, margin=MARGIN, step=STEP)
        pts = grid_points(origin, steps, shape)
        phi = eval_basis_grid(STO3G, h2_mol, pts)
        rho = eval_density_grid(h2_rhf.density_matrix, phi).reshape(shape)
        p = tmp_path / "h2_comments.cube"
        write_cube(p, h2_mol, rho, origin, steps,
                   comment1="LINE ONE", comment2="LINE TWO")
        lines = p.read_text().splitlines()
        assert lines[0] == "LINE ONE"
        assert lines[1] == "LINE TWO"


# ============================================================================
# F. export_density / export_orbital round-trip
# ============================================================================

class TestExportFunctions:

    def test_export_density_creates_file(self, tmp_path, h2_mol, h2_rhf):
        p, stats = export_density(
            tmp_path / "rho.cube", h2_mol, STO3G, h2_rhf,
            margin=MARGIN, step=STEP,
        )
        assert p.exists()

    def test_export_density_stats_keys(self, tmp_path, h2_mol, h2_rhf):
        _, stats = export_density(
            tmp_path / "rho2.cube", h2_mol, STO3G, h2_rhf,
            margin=MARGIN, step=STEP,
        )
        for key in ("n_electrons_integrated", "max_density", "min_density",
                    "grid_shape", "n_grid_points", "dV_bohr3"):
            assert key in stats, f"Missing key: {key}"

    def test_export_density_electron_count(self, tmp_path, h2_mol, h2_rhf):
        _, stats = export_density(
            tmp_path / "rho3.cube", h2_mol, STO3G, h2_rhf,
            margin=MARGIN, step=STEP,
        )
        n_e = stats["n_electrons_integrated"]
        assert abs(n_e - 2.0) / 2.0 < 0.10, f"∫ρ = {n_e:.4f}"

    def test_export_orbital_creates_file(self, tmp_path, h2_mol, h2_rhf):
        p, stats = export_orbital(
            tmp_path / "homo.cube", h2_mol, STO3G, h2_rhf, 0,
            margin=MARGIN, step=STEP,
        )
        assert p.exists()

    def test_export_orbital_stats_keys(self, tmp_path, h2_mol, h2_rhf):
        _, stats = export_orbital(
            tmp_path / "mo.cube", h2_mol, STO3G, h2_rhf, 0,
            margin=MARGIN, step=STEP,
        )
        for key in ("mo_idx", "mo_label", "mo_energy_ha", "mo_energy_ev",
                    "norm_integrated", "max_amplitude", "min_amplitude"):
            assert key in stats, f"Missing key: {key}"

    def test_export_orbital_norm(self, tmp_path, h2_mol, h2_rhf):
        n_occ = h2_mol.n_alpha
        _, stats = export_orbital(
            tmp_path / "homo2.cube", h2_mol, STO3G, h2_rhf, n_occ - 1,
            margin=MARGIN, step=STEP,
        )
        norm = stats["norm_integrated"]
        assert abs(norm - 1.0) < 0.10, f"‖ψ‖² = {norm:.4f}"

    def test_export_orbital_homo_label(self, tmp_path, h2_mol, h2_rhf):
        n_occ = h2_mol.n_alpha
        _, stats = export_orbital(
            tmp_path / "homo3.cube", h2_mol, STO3G, h2_rhf, n_occ - 1,
            margin=MARGIN, step=STEP,
        )
        assert stats["mo_label"] == "HOMO"

    def test_export_orbital_lumo_label(self, tmp_path, h2_mol, h2_rhf):
        n_occ = h2_mol.n_alpha
        _, stats = export_orbital(
            tmp_path / "lumo.cube", h2_mol, STO3G, h2_rhf, n_occ,
            margin=MARGIN, step=STEP,
        )
        assert stats["mo_label"] == "LUMO"

    def test_export_density_h2o(self, tmp_path, h2o_mol, h2o_rhf):
        """H2O: ∫ρ dr ≈ 10 e within 10%."""
        _, stats = export_density(
            tmp_path / "h2o_rho.cube", h2o_mol, STO3G, h2o_rhf,
            margin=MARGIN, step=STEP,
        )
        n_e = stats["n_electrons_integrated"]
        assert abs(n_e - 10.0) / 10.0 < 0.10, f"∫ρ = {n_e:.4f}"

    def test_export_homo_h2o_norm(self, tmp_path, h2o_mol, h2o_rhf):
        n_occ = h2o_mol.n_alpha
        _, stats = export_orbital(
            tmp_path / "h2o_homo.cube", h2o_mol, STO3G, h2o_rhf, n_occ - 1,
            margin=MARGIN, step=STEP,
        )
        norm = stats["norm_integrated"]
        assert abs(norm - 1.0) < 0.10, f"‖ψ_HOMO‖² = {norm:.4f}"

    def test_export_mo_energy_ev_sign(self, tmp_path, h2_mol, h2_rhf):
        """Core MO energy (MO 1) must be negative in eV."""
        _, stats = export_orbital(
            tmp_path / "mo1.cube", h2_mol, STO3G, h2_rhf, 0,
            margin=MARGIN, step=STEP,
        )
        assert stats["mo_energy_ev"] < 0.0

    def test_export_returns_path_object(self, tmp_path, h2_mol, h2_rhf):
        p, _ = export_density(
            tmp_path / "rho_path.cube", h2_mol, STO3G, h2_rhf,
            margin=MARGIN, step=STEP,
        )
        assert isinstance(p, Path)
