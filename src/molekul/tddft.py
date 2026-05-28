"""Time-dependent DFT in the Tamm-Dancoff approximation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import BasisSet
from .ccsd import transform_mo_full
from .constants import HARTREE_TO_EV
from .dft import (
    build_grid,
    eval_basis_gradient_on_grid,
    eval_basis_on_grid,
    eval_density,
    eval_density_gradient,
    eval_xc,
    ks_scf,
)
from .eri import build_eri
from .integrals import build_dipole_integrals
from .molecule import Molecule


@dataclass
class TDDFTResult:
    """Container for TD-DFT/TDA excited-state output."""

    excitation_energies: np.ndarray
    excitation_eV: np.ndarray
    oscillator_strengths: np.ndarray
    X: np.ndarray
    n_states: int
    functional: str
    n_occ: int
    n_virt: int


def _xc_kernel_rho(functional: str, rho: np.ndarray, sigma: np.ndarray | None = None) -> np.ndarray:
    """Numerical derivative d v_xc / d rho on the ground-state density."""
    rho = np.asarray(rho)
    step = np.maximum(1e-6 * np.maximum(rho, 1.0), 1e-8)
    rho_lo = np.maximum(rho - step, 1e-15)
    rho_hi = rho + step

    if functional.lower() in ("pbe", "gga_pbe") and sigma is not None:
        _, v_hi, _ = eval_xc(functional, rho_hi, sigma)
        _, v_lo, _ = eval_xc(functional, rho_lo, sigma)
    else:
        _, v_hi, _ = eval_xc(functional, rho_hi)
        _, v_lo, _ = eval_xc(functional, rho_lo)
    return (v_hi - v_lo) / (rho_hi - rho_lo)


def _build_tda_matrix(
        mol: Molecule,
        basis: BasisSet,
        functional: str,
        ks_result,
) -> np.ndarray:
    c = ks_result.mo_coefficients
    eps = ks_result.mo_energies
    n_occ = mol.n_alpha
    n_virt = c.shape[1] - n_occ
    dim = n_occ * n_virt

    eri_mo = transform_mo_full(build_eri(basis, mol), c)
    coulomb = eri_mo[:n_occ, n_occ:, :n_occ, n_occ:].reshape(dim, dim)

    coords, weights = build_grid(mol)
    phi = eval_basis_on_grid(basis, mol, coords)
    rho = np.maximum(eval_density(ks_result.density_matrix, phi), 0.0)
    sigma = None
    if functional.lower() in ("pbe", "gga_pbe"):
        dphi = eval_basis_gradient_on_grid(basis, mol, coords)
        sigma = eval_density_gradient(ks_result.density_matrix, phi, dphi)
    fxc = _xc_kernel_rho(functional, rho, sigma)

    mo_values = phi @ c
    ov_products = np.einsum(
        "gi,ga->gia", mo_values[:, :n_occ], mo_values[:, n_occ:]
    ).reshape(len(coords), dim)
    xc_kernel = np.einsum("g,gp,gq->pq", weights * fxc, ov_products, ov_products)

    orbital_gaps = np.array(
        [eps[n_occ + a] - eps[i] for i in range(n_occ) for a in range(n_virt)]
    )
    # Closed-shell singlet TDA: alpha and beta spin channels add equally.
    return np.diag(orbital_gaps) + 2.0 * coulomb + 2.0 * xc_kernel


def _oscillator_strengths(
        transition_vectors: np.ndarray,
        dip_mo: np.ndarray,
        omega: np.ndarray,
) -> np.ndarray:
    td = np.einsum("kia,xia->kx", transition_vectors, dip_mo)
    td_sq = np.sum(td ** 2, axis=1)
    return 2.0 * (2.0 / 3.0) * omega * td_sq


def tddft_tda(
        mol: Molecule,
        basis_fn: BasisSet,
        functional: str = "lda",
        n_states: int = 5,
        verbose: bool = False,
) -> TDDFTResult:
    """Compute singlet TD-DFT excitation energies in the TDA."""
    functional = functional.lower().strip()
    if functional not in ("lda", "lsda", "svwn", "svwn5", "pbe", "gga_pbe"):
        raise ValueError("TD-DFT supports 'lda' and 'pbe' functionals.")

    ks_result = ks_scf(mol, basis_fn, xc=functional, verbose=verbose)
    if not ks_result.converged:
        raise ValueError("TD-DFT requires a converged KS-DFT ground state.")

    n_occ = mol.n_alpha
    n_virt = ks_result.mo_coefficients.shape[1] - n_occ
    n_roots = min(n_states, n_occ * n_virt)

    a_matrix = _build_tda_matrix(mol, basis_fn, functional, ks_result)
    eigenvalues, eigenvectors = np.linalg.eigh(a_matrix)
    positive = np.where(eigenvalues > 0.0)[0]
    order = positive[np.argsort(eigenvalues[positive])][:n_roots]

    omega = eigenvalues[order]
    x_vectors = eigenvectors[:, order].T.reshape(len(order), n_occ, n_virt)

    dip_ao = build_dipole_integrals(basis_fn, mol)
    c = ks_result.mo_coefficients
    dip_mo = np.einsum("xmn,mi,na->xia", dip_ao, c[:, :n_occ], c[:, n_occ:])
    osc = _oscillator_strengths(x_vectors, dip_mo, omega)

    if verbose:
        for idx, energy in enumerate(omega, start=1):
            print(f"  TD-DFT/TDA state {idx}: {energy:.8f} Ha")

    return TDDFTResult(
        excitation_energies=omega,
        excitation_eV=omega * HARTREE_TO_EV,
        oscillator_strengths=osc,
        X=x_vectors,
        n_states=len(order),
        functional=functional,
        n_occ=n_occ,
        n_virt=n_virt,
    )
