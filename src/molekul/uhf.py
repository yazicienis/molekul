"""Unrestricted Hartree-Fock (UHF)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .basis import BasisSet
from .eri import build_eri
from .integrals import build_core_hamiltonian, build_overlap
from .molecule import Molecule
from .rhf import _sad_initial_density, _symmetric_orthogonalizer


@dataclass
class UHFResult:
    """Container for unrestricted Hartree-Fock output."""

    energy_total: float
    energy_hf: float
    energy_electronic: float
    energy_nuclear: float
    Ca: np.ndarray
    Cb: np.ndarray
    epsa: np.ndarray
    epsb: np.ndarray
    Pa: np.ndarray
    Pb: np.ndarray
    fock_alpha: np.ndarray
    fock_beta: np.ndarray
    S2: float
    converged: bool
    n_iter: int
    n_occ_a: int
    n_occ_b: int
    n_basis: int
    energy_history: list[float] = field(default_factory=list)


def _build_uhf_focks(
        h_core: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        eri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_total = pa + pb
    j_total = np.einsum("ls,mnls->mn", p_total, eri, optimize=True)
    ka = np.einsum("ls,mlns->mn", pa, eri, optimize=True)
    kb = np.einsum("ls,mlns->mn", pb, eri, optimize=True)
    return h_core + j_total - ka, h_core + j_total - kb


def _uhf_energy(
        h_core: np.ndarray,
        fa: np.ndarray,
        fb: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        e_nuc: float,
) -> tuple[float, float]:
    e_elec = 0.5 * (
        np.einsum("mn,mn->", pa, h_core + fa)
        + np.einsum("mn,mn->", pb, h_core + fb)
    )
    return float(e_elec), float(e_elec + e_nuc)


def _density_from_fock(
        fock: np.ndarray,
        x_orth: np.ndarray,
        n_occ: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp = x_orth.T @ fock @ x_orth
    eps, cp = np.linalg.eigh(fp)
    coeff = x_orth @ cp
    density = coeff[:, :n_occ] @ coeff[:, :n_occ].T
    return eps, coeff, density


def _spin_squared(
        ca: np.ndarray,
        cb: np.ndarray,
        overlap: np.ndarray,
        n_alpha: int,
        n_beta: int,
) -> float:
    sz = 0.5 * (n_alpha - n_beta)
    occ_overlap = ca[:, :n_alpha].T @ overlap @ cb[:, :n_beta]
    return float(sz * (sz + 1.0) + n_beta - np.sum(occ_overlap ** 2))


def _diis_extrapolate_pair(
        focks_a: list[np.ndarray],
        focks_b: list[np.ndarray],
        errors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    m = len(errors)
    if m < 2:
        return focks_a[-1].copy(), focks_b[-1].copy()

    bmat = np.zeros((m + 1, m + 1))
    for i in range(m):
        for j in range(i, m):
            value = float(np.dot(errors[i], errors[j]))
            bmat[i, j] = bmat[j, i] = value
    bmat[m, :] = -1.0
    bmat[:, m] = -1.0

    rhs = np.zeros(m + 1)
    rhs[m] = -1.0
    try:
        coeffs = np.linalg.solve(bmat, rhs)
    except np.linalg.LinAlgError:
        return focks_a[-1].copy(), focks_b[-1].copy()

    fa = np.zeros_like(focks_a[0])
    fb = np.zeros_like(focks_b[0])
    for coeff, fock_a, fock_b in zip(coeffs[:m], focks_a, focks_b):
        fa += coeff * fock_a
        fb += coeff * fock_b
    return fa, fb


def uhf_scf(
        mol: Molecule,
        basis_fn: BasisSet,
        verbose: bool = False,
        max_iter: int = 100,
        conv_tol: float = 1e-9,
) -> UHFResult:
    """Run an unrestricted Hartree-Fock SCF calculation."""
    n_alpha = mol.n_alpha
    n_beta = mol.n_beta
    if n_alpha < n_beta:
        raise ValueError("UHF expects n_alpha >= n_beta for the chosen spin projection.")

    overlap = build_overlap(basis_fn, mol)
    h_core = build_core_hamiltonian(basis_fn, mol)
    eri = build_eri(basis_fn, mol)
    e_nuc = mol.nuclear_repulsion_energy()
    n_basis = overlap.shape[0]

    if n_alpha > n_basis or n_beta > n_basis:
        raise ValueError("More same-spin electrons than basis functions.")

    x_orth = _symmetric_orthogonalizer(overlap)

    if n_alpha == n_beta:
        p_sad = _sad_initial_density(basis_fn, mol)
        pa = 0.5 * p_sad
        pb = 0.5 * p_sad
    else:
        pa = np.zeros_like(h_core)
        pb = np.zeros_like(h_core)

    energy = 0.0
    energy_history: list[float] = []
    converged = False
    n_iter = 0

    diis_focks_a: list[np.ndarray] = []
    diis_focks_b: list[np.ndarray] = []
    diis_errors: list[np.ndarray] = []
    diis_size = 8
    diis_start = 2

    if verbose:
        print(f"\n{'Iter':>4}  {'E_total (Hartree)':>20}  {'dE':>13}  {'dP_max':>13}  {'DIIS':>4}")
        print("-" * 60)

    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        fa, fb = _build_uhf_focks(h_core, pa, pb, eri)
        e_elec, e_total = _uhf_energy(h_core, fa, fb, pa, pb, e_nuc)
        energy_history.append(e_total)

        dE = e_total - energy
        energy = e_total

        error_a = fa @ pa @ overlap - overlap @ pa @ fa
        error_b = fb @ pb @ overlap - overlap @ pb @ fb
        diis_focks_a.append(fa.copy())
        diis_focks_b.append(fb.copy())
        diis_errors.append(np.concatenate([error_a.ravel(), error_b.ravel()]))
        if len(diis_errors) > diis_size:
            diis_focks_a.pop(0)
            diis_focks_b.pop(0)
            diis_errors.pop(0)

        use_diis = iteration >= diis_start and len(diis_errors) >= 2
        if use_diis:
            fa_use, fb_use = _diis_extrapolate_pair(diis_focks_a, diis_focks_b, diis_errors)
        else:
            fa_use = fa
            fb_use = fb

        epsa, ca, pa_new = _density_from_fock(fa_use, x_orth, n_alpha)
        epsb, cb, pb_new = _density_from_fock(fb_use, x_orth, n_beta)
        dP_max = float(max(np.max(np.abs(pa_new - pa)), np.max(np.abs(pb_new - pb))))
        pa, pb = pa_new, pb_new

        if verbose:
            tag = f"{len(diis_errors):>2}" if use_diis else " -"
            print(f"{iteration:>4}  {e_total:>20.10f}  {dE:>13.6e}  {dP_max:>13.6e}  {tag:>4}")

        if iteration > 1 and abs(dE) < conv_tol and dP_max < conv_tol:
            converged = True
            break

    fa_final, fb_final = _build_uhf_focks(h_core, pa, pb, eri)
    e_elec_final, e_total_final = _uhf_energy(h_core, fa_final, fb_final, pa, pb, e_nuc)
    epsa_final, ca_final, pa_final = _density_from_fock(fa_final, x_orth, n_alpha)
    epsb_final, cb_final, pb_final = _density_from_fock(fb_final, x_orth, n_beta)

    fa_final, fb_final = _build_uhf_focks(h_core, pa_final, pb_final, eri)
    e_elec_final, e_total_final = _uhf_energy(
        h_core, fa_final, fb_final, pa_final, pb_final, e_nuc
    )
    s2 = _spin_squared(ca_final, cb_final, overlap, n_alpha, n_beta)

    if verbose:
        status = "CONVERGED" if converged else "NOT CONVERGED"
        print(f"\nUHF {status} in {n_iter} iterations")
        print(f"  E_nuclear    = {e_nuc:>18.10f} Hartree")
        print(f"  E_electronic = {e_elec_final:>18.10f} Hartree")
        print(f"  E_total      = {e_total_final:>18.10f} Hartree")
        print(f"  <S^2>        = {s2:>18.10f}")

    return UHFResult(
        energy_total=e_total_final,
        energy_hf=e_total_final,
        energy_electronic=e_elec_final,
        energy_nuclear=e_nuc,
        Ca=ca_final,
        Cb=cb_final,
        epsa=epsa_final,
        epsb=epsb_final,
        Pa=pa_final,
        Pb=pb_final,
        fock_alpha=fa_final,
        fock_beta=fb_final,
        S2=s2,
        converged=converged,
        n_iter=n_iter,
        n_occ_a=n_alpha,
        n_occ_b=n_beta,
        n_basis=n_basis,
        energy_history=energy_history,
    )
