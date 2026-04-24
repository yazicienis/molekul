"""
Kohn-Sham DFT with LDA (Slater+VWN) and GGA (PBE) functionals.

Uses Becke atom-centered numerical integration grid.
Restricted closed-shell KS-DFT only.

References
----------
Becke, J. Chem. Phys. 88, 2547 (1988)  [grid partitioning]
Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980)  [VWN5 correlation]
Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)  [PBE]
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List


from .integrals import build_overlap, build_core_hamiltonian
from .eri import build_eri
from .rhf import _symmetric_orthogonalizer, _diis_extrapolate


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KSResult:
    """Container for KS-DFT output."""
    energy_total: float
    energy_electronic: float
    energy_nuclear: float
    energy_xc: float
    mo_energies: np.ndarray
    mo_coefficients: np.ndarray
    density_matrix: np.ndarray
    fock_matrix: np.ndarray
    n_iter: int
    converged: bool
    xc: str
    energy_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Atom-centered numerical grid (Becke, 1988)
# ---------------------------------------------------------------------------

# Bragg-Slater radii in Bohr (used for radial grid scaling)
_BRAGG_SLATER = {
    "H": 0.661, "He": 0.661,
    "Li": 2.268, "Be": 1.984, "B": 1.701, "C": 1.417, "N": 1.134,
    "O": 1.134, "F": 1.134, "Ne": 1.134,
    "Na": 3.402, "Mg": 3.023, "Al": 2.740, "Si": 2.457, "P": 2.173,
    "S": 2.173, "Cl": 2.079, "Ar": 2.079,
}

def _gauss_legendre(n):
    """Gauss-Legendre nodes and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def _atom_grid(symbol: str, n_rad: int = 75, n_ang: int = 302):
    """
    Return (coords, weights) for a single atom centered at origin.

    Radial: Gauss-Chebyshev type 2 with Mura-Knowles r-mapping.
    Angular: product Gauss-Legendre (theta) × uniform phi.
    """
    R_atom = _BRAGG_SLATER.get(symbol, 1.5)  # radial scale in Bohr

    # --- Radial quadrature (Mura-Knowles mapping) ---
    # Gauss-Chebyshev type 2 on (-1, 1)
    k = np.arange(1, n_rad + 1)
    x = np.cos(np.pi * k / (n_rad + 1))   # nodes
    # GC2 integrates ∫f sqrt(1-x²)dx; for plain ∫g(x)dx use w = π/(n+1)*sin(θ)
    w_gc = np.pi / (n_rad + 1) * np.sin(np.pi * k / (n_rad + 1))  # weights

    # Mura-Knowles: r = R * x^3/(1-x)^3, x in (0,1)
    # Map x from (-1,1) to (0,1): u = (x+1)/2
    u = (x + 1.0) * 0.5
    r = R_atom * u**3 / (1.0 - u)**3
    # Jacobian dr/du * du/dx
    drdu = R_atom * 3 * u**2 * (1.0 - u)**(-3) + R_atom * u**3 * 3 * (1.0 - u)**(-4)
    w_rad = w_gc * 0.5 * drdu * r**2  # includes r^2 from spherical volume element

    # --- Angular quadrature (product grid) ---
    n_theta = max(int(np.round(np.sqrt(n_ang / 2))), 8)
    n_phi   = 2 * n_theta
    cos_theta, w_theta = _gauss_legendre(n_theta)
    theta = np.arccos(cos_theta)
    phi   = 2 * np.pi * np.arange(n_phi) / n_phi
    w_phi = 2 * np.pi / n_phi * np.ones(n_phi)

    # Combine angular
    sin_theta = np.sin(theta)
    # grid points on unit sphere
    ang_coords = []
    ang_weights = []
    for i in range(n_theta):
        for j in range(n_phi):
            x_ = sin_theta[i] * np.cos(phi[j])
            y_ = sin_theta[i] * np.sin(phi[j])
            z_ = cos_theta[i]
            ang_coords.append([x_, y_, z_])
            ang_weights.append(w_theta[i] * w_phi[j])
    ang_coords  = np.array(ang_coords)   # (n_ang, 3)
    ang_weights = np.array(ang_weights)  # (n_ang,)

    # --- Combine radial × angular ---
    n_pts = n_rad * len(ang_coords)
    coords  = np.empty((n_pts, 3))
    weights = np.empty(n_pts)
    idx = 0
    for ir in range(n_rad):
        n_a = len(ang_coords)
        coords[idx:idx+n_a]  = r[ir] * ang_coords
        weights[idx:idx+n_a] = w_rad[ir] * ang_weights
        idx += n_a

    return coords, weights


def _becke_partitioning(coords_atom: np.ndarray, atom_idx: int,
                         atom_positions: np.ndarray) -> np.ndarray:
    """
    Becke atomic partitioning weights for grid centered on atom_idx.

    Returns weight array of shape (n_pts,) in [0, 1].
    """
    n_pts   = len(coords_atom)
    n_atoms = len(atom_positions)
    if n_atoms == 1:
        return np.ones(n_pts)

    # Distances from each grid point to each atom
    # coords_atom: (n_pts, 3), atom_positions: (n_atoms, 3)
    diff = coords_atom[:, None, :] - atom_positions[None, :, :]  # (n_pts, n_atoms, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n_pts, n_atoms)
    dist = np.where(dist < 1e-15, 1e-15, dist)

    # mu_{AB} = (r_A - r_B) / R_{AB}
    R_AB = np.linalg.norm(atom_positions[:, None, :] - atom_positions[None, :, :], axis=2)
    np.fill_diagonal(R_AB, 1.0)

    # Compute cell functions P(r) for each atom
    P = np.ones((n_pts, n_atoms))
    for IA in range(n_atoms):
        for IB in range(n_atoms):
            if IA == IB:
                continue
            mu = (dist[:, IA] - dist[:, IB]) / R_AB[IA, IB]  # (n_pts,)
            # Becke cutoff profile: 3 iterations of step function
            s = 0.5 * (1.0 - _becke_k3(mu))
            P[:, IA] *= s

    P_sum = P.sum(axis=1)
    P_sum = np.where(P_sum < 1e-30, 1e-30, P_sum)
    return P[:, atom_idx] / P_sum


def _becke_k3(mu: np.ndarray) -> np.ndarray:
    """Becke's polynomial cutoff: f3(f3(f3(mu))) where f(x) = 1.5x - 0.5x^3."""
    for _ in range(3):
        mu = 1.5 * mu - 0.5 * mu**3
    return mu


def build_grid(molecule, n_rad: int = 75, n_ang: int = 302):
    """
    Build Becke atom-centered integration grid for the molecule.

    Returns
    -------
    coords  : (N, 3) array of grid point coordinates (Bohr)
    weights : (N,)  array of integration weights
    """
    atom_positions = np.array([a.coords for a in molecule.atoms])
    all_coords  = []
    all_weights = []

    for i, atom in enumerate(molecule.atoms):
        center = atom_positions[i]
        c_local, w_local = _atom_grid(atom.symbol, n_rad, n_ang)
        c_global = c_local + center

        # Becke partitioning weights
        p_weights = _becke_partitioning(c_global, i, atom_positions)

        all_coords.append(c_global)
        all_weights.append(w_local * p_weights)

    return np.vstack(all_coords), np.concatenate(all_weights)


# ---------------------------------------------------------------------------
# Basis function evaluation on grid
# ---------------------------------------------------------------------------

def eval_basis_on_grid(basis, molecule, coords: np.ndarray) -> np.ndarray:
    """Evaluate all AO basis functions at grid points. Returns phi: (n_pts, n_basis)."""
    from .basis import norm_primitive
    n_pts   = len(coords)
    bfs     = basis.basis_functions(molecule)
    n_basis = len(bfs)
    phi     = np.zeros((n_pts, n_basis))

    for mu, (iatom, lx, ly, lz, shell) in enumerate(bfs):
        center = molecule.atoms[iatom].coords
        dr = coords - center          # (n_pts, 3)
        r2 = np.sum(dr**2, axis=1)
        val = np.zeros(n_pts)
        for exp_, c_ in zip(shell.exponents, shell.coefficients):
            norm = norm_primitive(exp_, lx, ly, lz)
            gf   = c_ * norm * np.exp(-exp_ * r2)
            poly = dr[:, 0]**lx * dr[:, 1]**ly * dr[:, 2]**lz
            val += gf * poly
        phi[:, mu] = val
    return phi


def eval_density(P: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute electron density on grid.

    rho(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)

    Parameters
    ----------
    P   : (n_basis, n_basis) density matrix
    phi : (n_pts, n_basis) basis values

    Returns rho : (n_pts,)
    """
    # rho[i] = phi[i,:] @ P @ phi[i,:] = einsum("im,mn,in->i", phi, P, phi)
    Pphi = phi @ P                  # (n_pts, n_basis)
    return np.einsum("in,in->i", Pphi, phi)


def eval_density_gradient(P: np.ndarray, phi: np.ndarray,
                           dphi: np.ndarray) -> np.ndarray:
    """
    Compute |∇ρ|^2 on grid (needed for GGA).

    ∇ρ(r) = 2 Σ_μν P_μν φ_μ(r) ∇φ_ν(r)
    sigma = |∇ρ|^2

    Parameters
    ----------
    phi  : (n_pts, n_basis)
    dphi : (n_pts, n_basis, 3) — basis function gradients

    Returns sigma : (n_pts,)
    """
    # grad_rho[i, x] = 2 * phi[i,:] @ P @ dphi[i,:,x]
    Pphi = phi @ P   # (n_pts, n_basis)
    grad_rho = 2.0 * np.einsum("im,imx->ix", Pphi, dphi)  # (n_pts, 3)
    return np.einsum("ix,ix->i", grad_rho, grad_rho)


# ---------------------------------------------------------------------------
# XC functionals
# ---------------------------------------------------------------------------

def _lda_x(rho: np.ndarray):
    """Slater exchange. Returns (eps_x, v_x)."""
    C_x = -(3.0/4.0) * (3.0/np.pi)**(1.0/3.0)
    rho_safe = np.where(rho < 1e-15, 1e-15, rho)
    eps_x = C_x * rho_safe**(1.0/3.0)
    v_x   = (4.0/3.0) * eps_x
    return eps_x, v_x


def _lda_c_vwn(rho: np.ndarray):
    """
    VWN5 correlation functional (Vosko-Wilk-Nusair, 1980, Eq. V).

    Returns (eps_c, v_c).
    """
    # VWN5 parameters (Table V of VWN 1980, Monte Carlo fit)
    A  =  0.0310907
    x0 = -0.10498
    b  =  3.72744
    c  =  12.9352

    rho_safe = np.where(rho < 1e-15, 1e-15, rho)
    rs = (3.0 / (4.0 * np.pi * rho_safe))**(1.0/3.0)
    x  = np.sqrt(rs)
    Q  = np.sqrt(4*c - b**2)
    X  = x**2 + b*x + c
    X0 = x0**2 + b*x0 + c

    eps_c = A * (np.log(x**2 / X)
                 + (2*b/Q) * np.arctan(Q / (2*x + b))
                 - (b*x0/X0) * (np.log((x - x0)**2 / X)
                                 + (2*(2*x0 + b)/Q) * np.arctan(Q / (2*x + b))))

    # v_c = eps_c + rs/3 * deps_c/drs
    # Numerical derivative is simpler and accurate enough
    h   = 1e-5
    rs_p = rs + h; x_p = np.sqrt(rs_p); X_p = x_p**2 + b*x_p + c
    ec_p = A * (np.log(x_p**2/X_p)
                + (2*b/Q)*np.arctan(Q/(2*x_p+b))
                - (b*x0/X0)*(np.log((x_p-x0)**2/X_p)
                              + (2*(2*x0+b)/Q)*np.arctan(Q/(2*x_p+b))))
    deps_drs = (ec_p - eps_c) / h
    v_c = eps_c - (rs / 3.0) * deps_drs

    return eps_c, v_c


def _pbe_x(rho: np.ndarray, sigma: np.ndarray):
    """PBE exchange. Returns (eps_x_pbe, v_x_rho, v_x_sigma)."""
    kappa = 0.804
    mu    = 0.2195149727645171   # pi^2/3 * kappa_x in PBE96

    rho_safe  = np.where(rho < 1e-15, 1e-15, rho)
    rho43     = rho_safe**(4.0/3.0)
    rho13     = rho_safe**(1.0/3.0)

    # LDA exchange energy density
    C_x   = -(3.0/4.0)*(3.0/np.pi)**(1.0/3.0)
    eps_x0 = C_x * rho13

    # Reduced gradient s = |∇ρ|/(2 k_F ρ)  where k_F = (3π²ρ)^(1/3)
    kF   = (3.0 * np.pi**2 * rho_safe)**(1.0/3.0)
    denom = 2.0 * kF * rho_safe
    sigma_safe = np.where(sigma < 0, 0.0, sigma)
    norm_grad  = np.sqrt(sigma_safe)
    s  = norm_grad / np.where(denom < 1e-15, 1e-15, denom)
    s2 = s**2

    # Enhancement factor F_x(s) = 1 + kappa - kappa/(1 + mu*s^2/kappa)
    Fx  = 1.0 + kappa - kappa / (1.0 + mu * s2 / kappa)
    dFx = 2.0 * mu * s / (1.0 + mu * s2 / kappa)**2

    eps_x = eps_x0 * Fx

    # Potentials: v_x_rho = d(rho*eps_x)/d(rho) = (4/3)*eps_x0*(Fx - s*dFx)
    v_x_rho   = (4.0/3.0) * eps_x0 * (Fx - s * dFx)
    # v with respect to sigma: use combined formula to avoid 1/sqrt(sigma) singularity
    # v_x_sigma = rho * eps_x0 * dFx/ds * ds/dsigma
    #           = rho * eps_x0 * mu / (denom^2 * (1 + mu*s^2/kappa)^2)
    denom_safe = np.where(denom < 1e-15, 1e-15, denom)
    v_x_sigma = rho_safe * eps_x0 * mu / (denom_safe**2 * (1.0 + mu * s2 / kappa)**2)

    return eps_x, v_x_rho, v_x_sigma


def _pbe_c(rho: np.ndarray, sigma: np.ndarray):
    """PBE correlation. Returns (eps_c_pbe, v_c_rho, v_c_sigma)."""
    # PBE96 parameters
    gamma_c = 0.031090690869655; beta_c = 0.06672455060314922

    rho_safe  = np.where(rho < 1e-15, 1e-15, rho)
    rho13     = rho_safe**(1.0/3.0)

    # LDA correlation (VWN)
    eps_c0, v_c0 = _lda_c_vwn(rho_safe)

    # rs and kF
    rs = (3.0 / (4.0 * np.pi * rho_safe))**(1.0/3.0)
    kF = (3.0 * np.pi**2 * rho_safe)**(1.0/3.0)
    ks = np.sqrt(4.0 * kF / np.pi)   # Thomas-Fermi screening k_s

    sigma_safe = np.where(sigma < 0, 0.0, sigma)
    norm_grad  = np.sqrt(sigma_safe)
    t = norm_grad / np.where(2.0 * ks * rho_safe < 1e-15, 1e-15, 2.0 * ks * rho_safe)
    t2 = t**2

    A   = beta_c / (gamma_c * (np.exp(-eps_c0 / gamma_c) - 1.0))
    phi = 1.0   # spin-scaling factor = 1 for closed-shell

    H   = gamma_c * phi**3 * np.log(
              1.0 + (beta_c / gamma_c) * t2 * (1.0 + A * t2) / (1.0 + A * t2 + A**2 * t2**2)
          )

    eps_c = eps_c0 + H

    # Potentials (numerical derivative for simplicity)
    h  = rho_safe * 1e-5
    rp = rho_safe + h
    ec0p, _ = _lda_c_vwn(rp)
    rsp = (3.0/(4.0*np.pi*rp))**(1/3); kFp = (3*np.pi**2*rp)**(1/3); ksp = np.sqrt(4*kFp/np.pi)
    tp  = norm_grad / np.where(2*ksp*rp < 1e-15, 1e-15, 2*ksp*rp); t2p = tp**2
    Ap  = beta_c/(gamma_c*(np.exp(-ec0p/gamma_c)-1.0))
    Hp  = gamma_c*np.log(1+(beta_c/gamma_c)*t2p*(1+Ap*t2p)/(1+Ap*t2p+Ap**2*t2p**2))
    ecp = ec0p + Hp
    v_c_rho = eps_c + rho_safe * (ecp - eps_c) / h

    # v_c_sigma: numerical d(eps_c)/d(sigma)
    h_sig = sigma_safe * 1e-5 + 1e-30
    sigma_p = sigma_safe + h_sig
    t2_p = sigma_p / (4.0 * ks**2 * rho_safe**2)
    A_p  = beta_c / (gamma_c * (np.exp(-eps_c0 / gamma_c) - 1.0))
    H_p  = gamma_c * np.log(1.0 + (beta_c / gamma_c) * t2_p * (1.0 + A_p * t2_p)
                            / (1.0 + A_p * t2_p + A_p**2 * t2_p**2))
    eps_c_p = eps_c0 + H_p
    v_c_sigma = rho_safe * (eps_c_p - eps_c) / h_sig

    return eps_c, v_c_rho, v_c_sigma


# ---------------------------------------------------------------------------
# XC evaluation dispatch
# ---------------------------------------------------------------------------

def eval_xc(xc: str, rho: np.ndarray, sigma: np.ndarray = None):
    """
    Evaluate XC energy density and potentials.

    Parameters
    ----------
    xc    : 'lda' or 'pbe'
    rho   : (n_pts,) electron density
    sigma : (n_pts,) |∇ρ|^2, required for GGA

    Returns
    -------
    eps_xc  : (n_pts,) XC energy per particle
    v_xc    : (n_pts,) XC potential dE/dρ
    v_sigma : (n_pts,) dE/d|∇ρ|^2, or None for LDA
    """
    xc = xc.lower().strip()
    if xc in ('lda', 'lsda', 'svwn', 'svwn5'):
        eps_x, v_x = _lda_x(rho)
        eps_c, v_c = _lda_c_vwn(rho)
        return eps_x + eps_c, v_x + v_c, None

    elif xc in ('pbe', 'gga_pbe'):
        if sigma is None:
            raise ValueError("PBE requires sigma = |∇ρ|^2")
        eps_x, v_x_r, v_x_s = _pbe_x(rho, sigma)
        eps_c, v_c_r, v_c_s = _pbe_c(rho, sigma)
        return eps_x + eps_c, v_x_r + v_c_r, v_x_s + v_c_s

    else:
        raise ValueError(f"Unknown XC functional: '{xc}'. Use 'lda' or 'pbe'.")


# ---------------------------------------------------------------------------
# XC matrix construction
# ---------------------------------------------------------------------------

def build_vxc_matrix(phi: np.ndarray, weights: np.ndarray,
                     v_xc: np.ndarray,
                     dphi: np.ndarray = None,
                     v_sigma: np.ndarray = None) -> np.ndarray:
    """
    Build V_xc matrix.

    (V_xc)_μν = ∫ φ_μ(r) v_xc(r) φ_ν(r) dr       [LDA]
               + 2 ∫ ∇φ_μ · (v_sigma ∇ρ) φ_ν dr    [GGA correction]

    Parameters
    ----------
    phi     : (n_pts, n_basis)
    weights : (n_pts,)
    v_xc    : (n_pts,)
    dphi    : (n_pts, n_basis, 3) — needed for GGA
    v_sigma : (n_pts,) — needed for GGA
    """
    w_vxc = weights * v_xc   # (n_pts,)
    Vxc = np.einsum("i,im,in->mn", w_vxc, phi, phi)

    if v_sigma is not None and dphi is not None:
        # GGA correction: 2 ∫ v_sigma * (∇ρ · ∇φ_μ) φ_ν
        # ∇ρ = 2 Σ_kl P_kl φ_k ∇φ_l — precomputed in dphi contractions
        # Simpler: (V_xc^GGA)_μν = 2 Σ_i w_i v_sigma_i (∇φ_μ · ∇ρ) φ_ν + h.c.
        # We store v_sigma * ∇ρ as a (n_pts, 3) vector computed externally
        # Here we use dphi and P to build ∇ρ on-the-fly — but P is not passed.
        # Instead we pass the GGA contribution as a pre-computed (n_pts,3) term.
        # For simplicity, skip the GGA term — caller should pass grad_rho.
        pass  # handled in ks_scf via _build_fock_ks

    return Vxc


def _build_fock_ks(H_core: np.ndarray, P: np.ndarray, eri: np.ndarray,
                   phi: np.ndarray, weights: np.ndarray, xc: str,
                   dphi: np.ndarray = None) -> tuple:
    """
    Build KS Fock matrix and E_xc.

    F_μν = H_μν + J_μν + V_xc_μν  (no exchange for DFT)

    Returns (F, E_xc)
    """
    # Coulomb J
    J = np.einsum("ls,mnls->mn", P, eri)

    # Density on grid
    rho = eval_density(P, phi)
    rho = np.where(rho < 0, 0.0, rho)

    # GGA: density gradient
    sigma = None
    if xc.lower() in ('pbe', 'gga_pbe') and dphi is not None:
        sigma = eval_density_gradient(P, phi, dphi)

    # XC
    eps_xc, v_xc, v_sigma = eval_xc(xc, rho, sigma)

    # E_xc = ∫ rho * eps_xc dr
    E_xc = float(np.dot(weights, rho * eps_xc))

    # V_xc matrix (LDA part)
    Vxc = np.einsum("i,im,in->mn", weights * v_xc, phi, phi)

    # GGA correction to V_xc
    if v_sigma is not None and dphi is not None:
        # ∇ρ = 2 Σ_kl P_kl φ_k ∇φ_l
        Pphi = phi @ P                              # (n_pts, n_basis)
        grad_rho = 2.0 * np.einsum("im,imx->ix", Pphi, dphi)  # (n_pts, 3)
        # Zero out v_sigma at low-density points (standard density cutoff)
        rho_mask = (rho > 1e-10).astype(float)
        v_sigma_safe = v_sigma * rho_mask
        # V_xc^GGA_μν = 2 Σ_i w_i v_sigma_i (∇ρ·∇φ_μ) φ_ν + μ↔ν
        h_mu = np.einsum("i,ix,imx->im", weights * v_sigma_safe, grad_rho, dphi)  # (n_pts, n_basis)
        gga = h_mu.T @ phi   # (n_basis, n_basis), asymmetric
        Vxc += 2.0 * (gga + gga.T)

    F = H_core + J + Vxc
    return F, E_xc


# ---------------------------------------------------------------------------
# Basis gradient evaluation (for GGA)
# ---------------------------------------------------------------------------

def eval_basis_gradient_on_grid(basis, molecule, coords: np.ndarray) -> np.ndarray:
    """Evaluate gradient ∇φ_μ(r) on grid. Returns dphi: (n_pts, n_basis, 3)."""
    from .basis import norm_primitive
    n_pts   = len(coords)
    bfs     = basis.basis_functions(molecule)
    n_basis = len(bfs)
    dphi    = np.zeros((n_pts, n_basis, 3))

    for mu, (iatom, lx, ly, lz, shell) in enumerate(bfs):
        center = molecule.atoms[iatom].coords
        dr = coords - center
        r2 = np.sum(dr**2, axis=1)
        dval = np.zeros((n_pts, 3))
        for exp_, c_ in zip(shell.exponents, shell.coefficients):
            norm = norm_primitive(exp_, lx, ly, lz)
            g    = c_ * norm * np.exp(-exp_ * r2)  # Gaussian factor
            # Cartesian polynomial and its gradient
            px = dr[:, 0]**lx * dr[:, 1]**ly * dr[:, 2]**lz  # φ polynomial
            # ∂φ/∂x = lx*x^(lx-1)*y^ly*z^lz * g + x^lx*y^ly*z^lz * (-2α x) * g
            for cart in range(3):
                powers = [lx, ly, lz]
                p = powers[cart]
                if p > 0:
                    powers[cart] -= 1
                    dpoly = p * dr[:, 0]**powers[0] * dr[:, 1]**powers[1] * dr[:, 2]**powers[2]
                    powers[cart] += 1
                else:
                    dpoly = np.zeros(n_pts)
                dval[:, cart] += g * (dpoly - 2.0 * exp_ * dr[:, cart] * px)
        dphi[:, mu, :] = dval
    return dphi


# ---------------------------------------------------------------------------
# Main KS-DFT SCF driver
# ---------------------------------------------------------------------------

def ks_scf(
        molecule,
        basis: BasisSet,
        xc: str = 'lda',
        *,
        n_rad: int = 75,
        n_ang: int = 302,
        max_iter: int = 100,
        e_conv: float = 1e-8,
        d_conv: float = 1e-6,
        diis_start: int = 2,
        diis_size: int = 8,
        verbose: bool = False,
) -> KSResult:
    """
    Run a Kohn-Sham DFT SCF calculation.

    Parameters
    ----------
    molecule   : closed-shell Molecule
    basis      : BasisSet
    xc         : exchange-correlation functional ('lda' or 'pbe')
    n_rad      : radial grid points per atom (default 75)
    n_ang      : approximate angular grid points per atom (default 302)
    verbose    : print iteration table

    Returns
    -------
    KSResult
    """
    if molecule.multiplicity != 1:
        raise ValueError("ks_scf requires a singlet (multiplicity=1).")

    n_occ = molecule.n_alpha

    # --- Integrals ---
    S      = build_overlap(basis, molecule)
    H_core = build_core_hamiltonian(basis, molecule)
    eri    = build_eri(basis, molecule)
    E_nuc  = molecule.nuclear_repulsion_energy()
    X      = _symmetric_orthogonalizer(S)

    # --- Grid ---
    if verbose:
        print(f"  KS-DFT ({xc.upper()}): building integration grid …")
    grid_coords, grid_weights = build_grid(molecule, n_rad=n_rad, n_ang=n_ang)
    if verbose:
        print(f"  Grid: {len(grid_coords)} points")

    # --- Basis on grid ---
    phi  = eval_basis_on_grid(basis, molecule, grid_coords)   # (n_pts, n_basis)
    dphi = None
    if xc.lower() in ('pbe', 'gga_pbe'):
        dphi = eval_basis_gradient_on_grid(basis, molecule, grid_coords)

    # --- Initial guess: core Hamiltonian ---
    Fp = X.T @ H_core @ X
    _, Cp = np.linalg.eigh(Fp)
    C  = X @ Cp
    P  = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    energy  = 0.0
    energy_history = []
    converged = False
    n_iter = 0

    diis_focks  = []
    diis_errors = []

    if verbose:
        print(f"\n{'Iter':>4}  {'E_total (Ha)':>20}  {'ΔE':>13}  {'ΔP_max':>11}")
        print("-" * 56)

    for it in range(1, max_iter + 1):
        n_iter = it

        F, E_xc = _build_fock_ks(H_core, P, eri, phi, grid_weights, xc, dphi)

        # Electronic energy: E = ½ tr[P(H + F)] - E_xc + E_xc  (KS form)
        # Correct: E_elec = ½ tr[P(H+F)] + E_xc - ½ tr[P * V_xc]
        # Simplest correct form:
        #   E_elec = Σ_i ε_i - ½ tr[P*J] - ½(tr[P*Vxc] - E_xc)?
        # Standard: E_elec = ½ tr[P*(H+F_KS)] - E_xc + E_xc ... let me be precise.
        #
        # KS energy: E = Σ_i ε_i - J/2 + E_xc - ∫ρ v_xc dr
        # Equivalently: E_elec = ½ tr[P*(H_core + F_KS)] + E_xc - tr[P*Vxc]/2
        # But simpler: tr[P*H]/2 + tr[P*J]/2 + E_xc  (since J = Coulomb only, no K)
        J_mat = np.einsum("ls,mnls->mn", P, eri)
        E_elec = 0.5 * np.einsum("mn,mn->", P, H_core + H_core + J_mat) + E_xc
        # = tr[P*H_core] + tr[P*J]/2 + E_xc
        E_elec = np.einsum("mn,mn->", P, H_core) + 0.5 * np.einsum("mn,mn->", P, J_mat) + E_xc
        E_total = E_elec + E_nuc

        dE = E_total - energy
        energy = E_total
        energy_history.append(E_total)

        # DIIS error: e = F P S - S P F
        FPS = F @ P @ S
        SPF = S @ P @ F
        err_mat = FPS - SPF

        if it >= diis_start:
            diis_focks.append(F.copy())
            diis_errors.append(err_mat.copy())
            if len(diis_focks) > diis_size:
                diis_focks.pop(0); diis_errors.pop(0)
            F_scf = _diis_extrapolate(diis_focks, diis_errors)
        else:
            F_scf = F

        # Diagonalise in orthonormal basis
        Fp_s = X.T @ F_scf @ X
        eps, Cp_s = np.linalg.eigh(Fp_s)
        C_new = X @ Cp_s
        P_new = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T

        dP_max = float(np.max(np.abs(P_new - P)))

        if verbose:
            print(f"{it:4d}  {E_total:20.10f}  {dE:13.6e}  {dP_max:11.4e}")

        P = P_new.copy()
        C = C_new.copy()
        mo_eps = eps.copy()

        if it > 1 and abs(dE) < e_conv and dP_max < d_conv:
            converged = True
            break

    if verbose:
        status = "converged" if converged else "NOT CONVERGED"
        print(f"\n  KS-DFT {status} in {n_iter} iterations")
        print(f"  E_total = {energy:.10f} Ha")
        print(f"  E_xc    = {E_xc:.10f} Ha")

    return KSResult(
        energy_total=energy,
        energy_electronic=E_elec,
        energy_nuclear=E_nuc,
        energy_xc=E_xc,
        mo_energies=mo_eps,
        mo_coefficients=C,
        density_matrix=P,
        fock_matrix=F,
        n_iter=n_iter,
        converged=converged,
        xc=xc,
        energy_history=energy_history,
    )
