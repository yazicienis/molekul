"""
Harmonic vibrational frequency analysis and IR intensities.

Theory
------
Starting from a converged RHF geometry (energy minimum), the harmonic
approximation expresses the potential energy surface as a quadratic function
of nuclear displacements.  Normal modes and frequencies are obtained by
diagonalising the mass-weighted Hessian matrix.

Hessian (numerical, central differences)
-----------------------------------------
  H_{αβ} = ∂²E / ∂q_α ∂q_β

Diagonal:
  H_{αα} = [E(+α) − 2E₀ + E(−α)] / h²

Off-diagonal  (4-point cross difference):
  H_{αβ} = [E(+α+β) − E(+α−β) − E(−α+β) + E(−α−β)] / (4h²)

Total SCF calls: 1 + 2·(3N) + 4·C(3N,2)  (163 for H₂O, 9 atoms)

Mass-weighted Hessian and normal modes
--------------------------------------
  H̃_{αβ} = H_{αβ} / √(m_α m_β)   [m in amu]
  H̃ Φ = Φ Λ  (eigenvector decomposition)
  ω_k = √λ_k × FREQ_CONV    [cm⁻¹]

  Negative eigenvalue → imaginary frequency (saddle point / transition state).

IR intensities (double-harmonic approximation)
----------------------------------------------
  ∂μ_c / ∂q_α  ≈  [μ_c(+α) − μ_c(−α)] / (2h)   (3N central-difference dipoles)

  dμ/dQ_k = Σ_α Φ_{αk} / √m_α × (∂μ/∂q_α)        [ea₀ / (√amu · a₀)]

  A_k = INTENS_CONV × Σ_c |dμ_c/dQ_k|²              [km / mol]

Physical constants (CODATA 2018)
---------------------------------
  FREQ_CONV  = √(Eₕ / (a₀² · amu)) / (2π c)  ≈ 5140.5 cm⁻¹
  INTENS_CONV = N_A π (e a₀)² / (3000 c² ε₀ amu) ≈ 974.9 km/mol per a.u.

References
----------
Wilson, Decius, Cross, "Molecular Vibrations", McGraw-Hill (1955).
Bauernschmitt & Ahlrichs, Chem. Phys. Lett. 256, 454 (1996).
Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure Theory", Ch. 15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .basis import BasisSet
from .constants import ATOMIC_MASS
from .grad import _displaced
from .molecule import Molecule
from .population import build_dipole_integrals, dipole_moment
from .rhf import rhf_scf, RHFResult


# ---------------------------------------------------------------------------
# Physical constants and derived conversion factors  (CODATA 2018)
# ---------------------------------------------------------------------------

_EH    = 4.3597447222071e-18   # Hartree in J
_A0    = 5.29177210903e-11     # Bohr radius in m
_AMU   = 1.66053906660e-27     # atomic mass unit in kg
_C_CM  = 2.99792458e10         # speed of light in cm / s

#: sqrt(E_h / (a_0² · amu)) / (2π c)  — converts Hessian eigenvalues to cm⁻¹
FREQ_CONV = math.sqrt(_EH / (_AMU * _A0 ** 2)) / (2.0 * math.pi * _C_CM)

# IR intensity conversion:  |dμ/dQ|² in (e·a₀)²/amu  →  km/mol
#
# The standard formula (Wilson, Decius, Cross; Neugebauer & Hess) is:
#   A_k [km/mol] = (N_A π) / (3 · 1000) × (e·a₀)² / (ε₀ · c² · amu) × |dμ/dQ_k|²
#
# In SI: (e·a₀)² = (1.602176634e-19 × 5.29177210903e-11)² C²·m²
#        N_A/(3000 c² ε₀ amu) has units mol⁻¹/(m²/s² · C²/(N·m²) · kg)
#                             = mol⁻¹ · N·s²/(m²·kg) × (1/s²) = mol⁻¹ · N/(m·kg)
#                             = mol⁻¹ · 1/m²  (N = kg·m/s²)
# Full: (e·a₀)² × N_A·π/(3000·c²·ε₀·amu) [C²·m² · mol⁻¹·m⁻²] = [C²·mol⁻¹/N·m²·m²/N·m²]
#
# Numerically this equals 42.2561 × (2.541746 / 0.529177)² ≈ 974.9 km/mol per (e·a₀)²/amu
# where 2.541746 D/au is the dipole conversion and 0.529177 Å/bohr is the Bohr radius.
#
# Simplest correct expression (validated against PySCF/ORCA convention):
_DEBYE_PER_AU   = 2.541746473      # 1 ea₀ in Debye
_ANGST_PER_BOHR = 0.529177210903   # 1 bohr in Ångström
INTENS_CONV = 42.2561 * (_DEBYE_PER_AU / _ANGST_PER_BOHR) ** 2  # ≈ 974.9 km/mol per a.u.

# Hartree to cm⁻¹
CM_INV_PER_EH = 219474.6313632  # 1 Hartree = 219474.63 cm⁻¹

#: Modes with |ω| below this threshold are treated as translation / rotation.
ZERO_FREQ_THRESHOLD = 100.0   # cm⁻¹  (fallback only)


def _n_rigid_modes(molecule) -> int:
    """
    Return the number of rigid-body (translation + rotation) degrees of freedom.

    Rules (standard):
      1 atom  : 3 (translations only, no rotations)
      Linear  : 5 (3 translations + 2 rotations)
      Nonlinear: 6 (3 translations + 3 rotations)
    """
    n = molecule.n_atoms
    if n == 1:
        return 3
    coords = np.array([a.coords for a in molecule.atoms])   # (N, 3)
    # Check collinearity: all atoms lie on the same line through atoms[0]–atoms[1]
    v0 = coords[1] - coords[0]
    v0 /= np.linalg.norm(v0)
    linear = True
    for i in range(2, n):
        vi = coords[i] - coords[0]
        norm_vi = np.linalg.norm(vi)
        if norm_vi < 1e-10:
            continue   # coincident atom — skip
        vi /= norm_vi
        cross = np.cross(v0, vi)
        if np.linalg.norm(cross) > 1e-4:
            linear = False
            break
    return 5 if linear else 6


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FreqResult:
    """
    Harmonic vibrational analysis output.

    Attributes
    ----------
    frequencies     : (n_vib,) harmonic frequencies in cm⁻¹ for vibrational modes
                      Sorted ascending.  Negative values = imaginary modes.
    intensities     : (n_vib,) IR integrated intensities in km/mol
    zero_point_energy : ZPE = ½ Σ_k ω_k  in Hartree  (sum over real vibrations)
    n_imaginary     : number of vibrational modes with imaginary frequency
    n_zero          : number of near-zero modes (translation + rotation removed)

    all_frequencies : (3N,) all 3N frequencies, including zero modes
    all_intensities : (3N,) all 3N IR intensities
    hessian         : (3N, 3N) raw Cartesian Hessian in Eh / bohr²
    normal_modes    : (3N, 3N) eigenvector matrix (column k = mode k, ascending eigenvalue)
    eigenvalues     : (3N,) mass-weighted Hessian eigenvalues in Eh / (bohr² · amu)
    """
    frequencies:      np.ndarray
    intensities:      np.ndarray
    zero_point_energy: float
    n_imaginary:      int
    n_zero:           int
    all_frequencies:  np.ndarray
    all_intensities:  np.ndarray
    hessian:          np.ndarray
    normal_modes:     np.ndarray
    eigenvalues:      np.ndarray


# ---------------------------------------------------------------------------
# Numerical Hessian
# ---------------------------------------------------------------------------

def numerical_hessian(
        molecule: Molecule,
        basis: BasisSet,
        *,
        h: float = 5e-3,
        verbose: bool = True,
) -> np.ndarray:
    """
    Compute the Cartesian nuclear Hessian by central finite differences.

    Parameters
    ----------
    molecule : Molecule at the reference geometry (should be a minimum)
    basis    : BasisSet
    h        : finite-difference step size in bohr (default 5e-3)
    verbose  : print progress

    Returns
    -------
    H : (3N, 3N) float64 array, Cartesian Hessian in Eh / bohr²
        Symmetric by construction.
    """
    n_atoms = molecule.n_atoms
    ndof    = 3 * n_atoms

    def _e(mol):
        return rhf_scf(mol, basis, verbose=False).energy_total

    if verbose:
        total = 1 + 2 * ndof + 4 * (ndof * (ndof - 1) // 2)
        print(f"  Hessian: {ndof} DOF, {total} SCF evaluations  (h = {h:.0e} bohr)")

    E0 = _e(molecule)

    # ── Singly-displaced energies ──────────────────────────────────────────
    E_plus  = np.empty(ndof)
    E_minus = np.empty(ndof)
    for a in range(n_atoms):
        for i in range(3):
            alpha = 3 * a + i
            E_plus[alpha]  = _e(_displaced(molecule, a, i, +h))
            E_minus[alpha] = _e(_displaced(molecule, a, i, -h))
            if verbose:
                print(f"    grad [{alpha+1}/{ndof}]", end="\r", flush=True)

    # ── Hessian matrix ────────────────────────────────────────────────────
    H = np.zeros((ndof, ndof))

    for alpha in range(ndof):
        H[alpha, alpha] = (E_plus[alpha] - 2.0 * E0 + E_minus[alpha]) / h ** 2

    n_off = ndof * (ndof - 1) // 2
    done  = 0
    for alpha in range(ndof):
        a1, i1 = alpha // 3, alpha % 3
        for beta in range(alpha + 1, ndof):
            a2, i2 = beta // 3, beta % 3
            # Two-step displacement: apply i first, then j
            mol_pp = _displaced(_displaced(molecule, a1, i1, +h), a2, i2, +h)
            mol_pm = _displaced(_displaced(molecule, a1, i1, +h), a2, i2, -h)
            mol_mp = _displaced(_displaced(molecule, a1, i1, -h), a2, i2, +h)
            mol_mm = _displaced(_displaced(molecule, a1, i1, -h), a2, i2, -h)
            val = (_e(mol_pp) - _e(mol_pm) - _e(mol_mp) + _e(mol_mm)) / (4.0 * h ** 2)
            H[alpha, beta] = H[beta, alpha] = val
            done += 1
            if verbose:
                print(f"    H[{alpha},{beta}]  {done}/{n_off}", end="\r", flush=True)

    if verbose:
        print()
    return H


# ---------------------------------------------------------------------------
# Dipole derivatives  ∂μ/∂q_α
# ---------------------------------------------------------------------------

def _dipole_derivatives(
        molecule: Molecule,
        basis: BasisSet,
        *,
        h: float = 1e-2,
) -> np.ndarray:
    """
    Compute (3N × 3) matrix of dipole derivatives ∂μ_c/∂q_α in atomic units.

    Result[α, c] = ∂μ_c / ∂q_α  in ea₀ / bohr.
    """
    n_atoms = molecule.n_atoms
    ndof    = 3 * n_atoms
    dmu     = np.zeros((ndof, 3))

    for a in range(n_atoms):
        for i in range(3):
            alpha = 3 * a + i
            mol_p = _displaced(molecule, a, i, +h)
            mol_m = _displaced(molecule, a, i, -h)

            rhf_p = rhf_scf(mol_p, basis, verbose=False)
            rhf_m = rhf_scf(mol_m, basis, verbose=False)

            Dx_p, Dy_p, Dz_p = build_dipole_integrals(basis, mol_p)
            Dx_m, Dy_m, Dz_m = build_dipole_integrals(basis, mol_m)

            mu_p = dipole_moment(rhf_p.density_matrix, basis, mol_p,
                                  Dx=Dx_p, Dy=Dy_p, Dz=Dz_p).total_au
            mu_m = dipole_moment(rhf_m.density_matrix, basis, mol_m,
                                  Dx=Dx_m, Dy=Dy_m, Dz=Dz_m).total_au

            dmu[alpha] = (mu_p - mu_m) / (2.0 * h)

    return dmu


# ---------------------------------------------------------------------------
# Full harmonic analysis
# ---------------------------------------------------------------------------

def harmonic_analysis(
        molecule: Molecule,
        basis: BasisSet,
        rhf_result: RHFResult | None = None,
        *,
        h_hess: float = 5e-3,
        h_dip:  float = 1e-2,
        verbose: bool = True,
) -> FreqResult:
    """
    Perform harmonic vibrational analysis: Hessian → frequencies + IR intensities.

    Parameters
    ----------
    molecule    : Molecule at the equilibrium geometry
    basis       : BasisSet
    rhf_result  : (unused, reserved for future analytic Hessian)
    h_hess      : step size for numerical Hessian (bohr, default 5e-3)
    h_dip       : step size for dipole derivatives (bohr, default 1e-2)
    verbose     : print progress information

    Returns
    -------
    FreqResult
    """
    n_atoms = molecule.n_atoms
    ndof    = 3 * n_atoms

    # ── 1. Hessian ──────────────────────────────────────────────────────────
    if verbose:
        print("  Step 1/3: Numerical Hessian")
    H = numerical_hessian(molecule, basis, h=h_hess, verbose=verbose)

    # ── 2. Mass-weighted Hessian and diagonalization ─────────────────────
    if verbose:
        print("  Step 2/3: Normal mode analysis")

    masses   = np.array([ATOMIC_MASS[a.Z] for a in molecule.atoms])  # (n_atoms,)
    mass_vec = np.repeat(masses, 3)                                    # (3N,)

    M_isqrt = 1.0 / np.sqrt(mass_vec)                                 # 1/sqrt(m_α)
    H_mw    = M_isqrt[:, None] * H * M_isqrt[None, :]                 # (3N, 3N)

    eigenvalues, Phi = np.linalg.eigh(H_mw)     # ascending, Φ columns = eigenvectors

    # Frequencies: ω [cm⁻¹] = sqrt(|λ|) × FREQ_CONV, negative if λ < 0
    all_freqs = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * FREQ_CONV

    # ── 3. Dipole derivatives and IR intensities ─────────────────────────
    if verbose:
        print("  Step 3/3: Dipole derivatives for IR intensities")

    dmu_dq   = _dipole_derivatives(molecule, basis, h=h_dip)   # (3N, 3)

    # dμ/dQ_k [ea₀/(√amu·a₀)] = Σ_α Φ_{αk}/√m_α × ∂μ/∂q_α
    dmu_mw   = dmu_dq * M_isqrt[:, None]    # (3N, 3) — divide each row by √m_α
    dmu_dQ   = Phi.T @ dmu_mw               # (3N, 3) — normal-mode dipole derivatives

    all_ints = np.sum(dmu_dQ ** 2, axis=1) * INTENS_CONV   # (3N,) in km/mol

    # ── 4. Separate zero modes from vibrations ───────────────────────────
    # Use geometry to determine the exact number of rigid-body modes (trans+rot).
    # The lowest n_zero eigenvalues correspond to translations/rotations.
    n_zero   = _n_rigid_modes(molecule)
    # Sort indices by ascending |frequency|, take the n_zero smallest as zero modes
    # (the Hessian diagonalization already returns ascending eigenvalues, so the
    # zero modes are simply the first n_zero entries in all_freqs)
    zero_mask = np.zeros(ndof, dtype=bool)
    zero_mask[:n_zero] = True
    vib_mask  = ~zero_mask

    vib_freqs = all_freqs[vib_mask]
    vib_ints  = all_ints[vib_mask]

    n_imaginary = int((vib_freqs < 0).sum())

    # ── 5. Zero-point energy (real vibrations only) ───────────────────────
    real_vib = vib_freqs[vib_freqs > 0]
    zpe = 0.5 * float(np.sum(real_vib)) / CM_INV_PER_EH   # Hartree

    if verbose:
        print(f"\n  {n_atoms}-atom molecule, {ndof} DOF, "
              f"{n_zero} zero modes, {len(vib_freqs)} vibrations")
        print(f"  Imaginary frequencies: {n_imaginary}")
        print(f"  ZPE = {zpe:.6f} Ha = {zpe * CM_INV_PER_EH:.1f} cm⁻¹")

    return FreqResult(
        frequencies=vib_freqs,
        intensities=vib_ints,
        zero_point_energy=zpe,
        n_imaginary=n_imaginary,
        n_zero=n_zero,
        all_frequencies=all_freqs,
        all_intensities=all_ints,
        hessian=H,
        normal_modes=Phi,
        eigenvalues=eigenvalues,
    )
