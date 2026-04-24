"""
Restricted Hartree-Fock (RHF) SCF with DIIS convergence acceleration.

Theory
------
For a closed-shell molecule with N electrons and n basis functions:

  F_μν = H_μν + G_μν
  G_μν = Σ_{λσ} P_{λσ} [(μν|λσ) - ½(μλ|νσ)]

where P_{λσ} = 2 Σ_{i=1}^{N/2} C_{λi} C_{σi}  (density matrix, factor 2 for spin)

SCF in orthonormal basis via X = S^{-½}:
  F' = X^T F X,   F' C' = C' ε,   C = X C'

Convergence: |ΔE| < e_conv  AND  max|ΔP| < d_conv

Energy:
  E_elec = ½ tr[P (H_core + F)]
  E_total = E_elec + E_nuc

DIIS (Pulay 1980):
  Error vector e_i = F_i P_i S - S P_i F_i
  Extrapolated Fock: F = Σ c_i F_i  subject to Σ c_i = 1

References
----------
Szabo & Ostlund, "Modern Quantum Chemistry", Chapter 3.
Pulay, Chem. Phys. Lett. 73, 393 (1980).
Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure Theory", Chapter 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .basis import BasisSet
from .integrals import build_overlap, build_core_hamiltonian
from .eri import build_eri


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RHFResult:
    """Container for RHF/SCF output."""
    energy_total: float        # Total energy (Hartree)
    energy_electronic: float   # Electronic energy (Hartree)
    energy_nuclear: float      # Nuclear repulsion energy (Hartree)
    mo_energies: np.ndarray    # MO eigenvalues ε_i, shape (n_basis,)
    mo_coefficients: np.ndarray  # MO coefficients C, shape (n_basis, n_basis)
    density_matrix: np.ndarray   # AO density matrix P, shape (n_basis, n_basis)
    fock_matrix: np.ndarray      # Final AO Fock matrix F, shape (n_basis, n_basis)
    n_iter: int                  # SCF iterations used
    converged: bool              # True if SCF converged
    energy_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core SCF routines
# ---------------------------------------------------------------------------

def _build_fock(H_core: np.ndarray, P: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """
    Build the Fock matrix in the AO basis.

    F_μν = H_μν + J_μν - ½ K_μν

    where:
      J_μν = Σ_{λσ} P_{λσ} (μν|λσ)   [Coulomb]
      K_μν = Σ_{λσ} P_{λσ} (μλ|νσ)   [Exchange]

    ERI tensor convention: eri[μ,ν,λ,σ] = (μν|λσ)  (chemists' notation)
    """
    J = np.einsum("ls,mnls->mn", P, eri)
    K = np.einsum("ls,mlns->mn", P, eri)
    return H_core + J - 0.5 * K


def _diis_extrapolate(
        fock_list: List[np.ndarray],
        error_list: List[np.ndarray],
) -> np.ndarray:
    """
    Pulay DIIS extrapolation.

    Minimise ||Σ c_i e_i||^2 subject to Σ c_i = 1 using the
    B-matrix approach. Returns the extrapolated Fock matrix.
    """
    m = len(fock_list)
    B = np.zeros((m + 1, m + 1))
    for i in range(m):
        for j in range(i, m):
            val = np.einsum("ij,ij->", error_list[i], error_list[j])
            B[i, j] = val
            B[j, i] = val
    B[m, :] = -1.0
    B[:, m] = -1.0
    B[m, m] = 0.0

    rhs = np.zeros(m + 1)
    rhs[m] = -1.0

    try:
        coeffs = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        return fock_list[-1].copy()

    F_extrap = np.zeros_like(fock_list[0])
    for c, F in zip(coeffs[:m], fock_list):
        F_extrap += c * F
    return F_extrap


def _symmetric_orthogonalizer(S: np.ndarray) -> np.ndarray:
    """
    Compute X = S^{-1/2} by symmetric (Löwdin) orthogonalisation.

    X = U s^{-1/2} U^T  where  S = U s U^T.
    """
    s_vals, s_vecs = np.linalg.eigh(S)
    s_vals = np.where(s_vals > 1e-10, s_vals, 1e-10)   # guard against near-zero
    return s_vecs @ np.diag(s_vals ** -0.5) @ s_vecs.T


# Fractional occupations for the diagonal SAD guess (spherically averaged).
# Format: list of occupations in the same order as BasisSet.basis_functions(),
# i.e. shell-by-shell, then angular component within each shell.
# STO-3G shell ordering per element: H/He → 1s; Li–Ne → 1s, 2s, 2px, 2py, 2pz.
_SAD_OCC: dict = {
    "H":  [1.0],
    "He": [2.0],
    "Li": [2.0, 1.0,   0.0, 0.0, 0.0],
    "Be": [2.0, 2.0,   0.0, 0.0, 0.0],
    "B":  [2.0, 2.0,   1/3, 1/3, 1/3],
    "C":  [2.0, 2.0,   2/3, 2/3, 2/3],
    "N":  [2.0, 2.0,   1.0, 1.0, 1.0],
    "O":  [2.0, 2.0,   4/3, 4/3, 4/3],
    "F":  [2.0, 2.0,   5/3, 5/3, 5/3],
    "Ne": [2.0, 2.0,   2.0, 2.0, 2.0],
}


def _sad_initial_density(basis, molecule) -> np.ndarray:
    """
    Superposition of Atomic Densities (SAD) initial guess.

    Builds a diagonal density matrix P where P[μ,μ] = fractional occupation
    of basis function μ based on the neutral atom's ground-state configuration
    (p electrons distributed equally over px/py/pz for spherical symmetry).

    Falls back to uniform distribution for unknown elements.
    """
    bfs = basis.basis_functions(molecule)
    n_basis = len(bfs)
    P = np.zeros((n_basis, n_basis))

    # Group basis function indices by atom
    atom_bf_indices: dict = {}
    for bf_idx, (atom_idx, *_) in enumerate(bfs):
        atom_bf_indices.setdefault(atom_idx, []).append(bf_idx)

    for atom_idx, atom in enumerate(molecule.atoms):
        indices = atom_bf_indices.get(atom_idx, [])
        sym = atom.symbol
        if sym in _SAD_OCC and len(indices) == len(_SAD_OCC[sym]):
            for global_idx, occ in zip(indices, _SAD_OCC[sym]):
                P[global_idx, global_idx] = occ
        else:
            # Uniform fallback: spread atom's electrons equally
            n_el = getattr(atom, "atomic_number", len(indices))
            occ = n_el / max(len(indices), 1)
            for global_idx in indices:
                P[global_idx, global_idx] = occ

    return P


# ---------------------------------------------------------------------------
# Main SCF driver
# ---------------------------------------------------------------------------

def rhf_scf(
        molecule,
        basis: BasisSet,
        *,
        max_iter: int = 100,
        e_conv: float = 1e-10,
        d_conv: float = 1e-8,
        diis_start: int = 2,
        diis_size: int = 8,
        level_shift: float = 0.2,
        verbose: bool = False,
) -> RHFResult:
    """
    Run a Restricted Hartree-Fock SCF calculation.

    Parameters
    ----------
    molecule     : Molecule — must be closed-shell (multiplicity=1)
    basis        : BasisSet (e.g. ``STO3G``)
    max_iter     : maximum number of SCF iterations
    e_conv       : convergence threshold for |ΔE| (Hartree)
    d_conv       : convergence threshold for max|ΔP|
    diis_start   : iteration at which to activate DIIS
    diis_size    : maximum number of stored DIIS vectors
    level_shift  : virtual-orbital energy shift (Hartree) added to DIIS Fock to
                   prevent wrong-state convergence (Saunders & Hillier 1973).
                   The converged density is unaffected; set to 0 to disable.
    verbose      : print iteration table if True

    Returns
    -------
    RHFResult
    """
    if molecule.multiplicity != 1:
        raise ValueError(
            f"rhf_scf requires a singlet (multiplicity=1), "
            f"got multiplicity={molecule.multiplicity}."
        )
    n_occ = molecule.n_alpha   # closed-shell: n_alpha == n_beta

    # --- One-electron integrals and ERI tensor ---------------------------------
    S = build_overlap(basis, molecule)
    H_core = build_core_hamiltonian(basis, molecule)
    eri = build_eri(basis, molecule)
    E_nuc = molecule.nuclear_repulsion_energy()

    # --- Orthogonalisation matrix X = S^{-1/2} --------------------------------
    X = _symmetric_orthogonalizer(S)

    # --- Initial guess: Superposition of Atomic Densities (SAD) --------------
    # SAD gives a chemically sensible starting density (spherically-averaged
    # neutral-atom occupations) that avoids the wrong orbital-occupation
    # traps that Hcore diagonalisation produces for symmetric molecules (e.g.
    # N2: Hcore puts πg(antibonding) occupied instead of 3σg).
    P = _sad_initial_density(basis, molecule)

    energy = 0.0
    energy_history: List[float] = []
    converged = False
    n_iter = 0

    diis_focks: List[np.ndarray] = []
    diis_errors: List[np.ndarray] = []

    if verbose:
        print(f"\n{'Iter':>4}  {'E_total (Hartree)':>20}  {'ΔE':>13}  {'ΔP_max':>13}  {'DIIS':>4}")
        print("-" * 60)

    for it in range(1, max_iter + 1):
        n_iter = it

        # Build Fock matrix
        F = _build_fock(H_core, P, eri)

        # Electronic energy: E_elec = ½ tr[P(H + F)]
        E_elec = 0.5 * np.einsum("mn,mn->", P, H_core + F)
        E_total = E_elec + E_nuc
        energy_history.append(E_total)

        dE = E_total - energy
        energy = E_total

        # DIIS error vector: e = FPS - SPF  (measure of [F,P] non-commutativity)
        error = F @ P @ S - S @ P @ F

        # Level-shifted Fock for DIIS: F_LS = F + σ(S − S·(P/2)·S)
        # Raises virtual orbital energies by σ without changing converged density.
        # Derivation: orthonormal-basis shift F' + σ(I − P'_occ) back-transformed to AO.
        if level_shift > 0.0 and n_occ < S.shape[0]:
            SP = S @ (0.5 * P)  # S · (P/2)
            F_diis = F + level_shift * (S - SP @ S)
        else:
            F_diis = F
        diis_focks.append(F_diis)
        diis_errors.append(error.copy())
        if len(diis_focks) > diis_size:
            diis_focks.pop(0)
            diis_errors.pop(0)

        # DIIS extrapolation after warm-up
        use_diis = (it >= diis_start) and (len(diis_focks) >= 2)
        F_use = _diis_extrapolate(diis_focks, diis_errors) if use_diis else F_diis

        # Diagonalise in orthonormal basis
        Fp = X.T @ F_use @ X
        _, Cp = np.linalg.eigh(Fp)
        C_new = X @ Cp

        P_new = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T
        dP_max = float(np.max(np.abs(P_new - P)))
        P = P_new
        C = C_new

        if verbose:
            tag = f"{len(diis_focks):>2}" if use_diis else " -"
            print(f"{it:>4}  {E_total:>20.10f}  {dE:>13.6e}  {dP_max:>13.6e}  {tag:>4}")

        if it > 1 and abs(dE) < e_conv and dP_max < d_conv:
            converged = True
            break

    # --- Final quantities from converged density ------------------------------
    F_final = _build_fock(H_core, P, eri)
    E_elec_final = 0.5 * np.einsum("mn,mn->", P, H_core + F_final)
    E_total_final = E_elec_final + E_nuc

    Fp_final = X.T @ F_final @ X
    eps_final, Cp_final = np.linalg.eigh(Fp_final)
    C_final = X @ Cp_final

    if verbose:
        status = "CONVERGED" if converged else "NOT CONVERGED"
        print(f"\nSCF {status} in {n_iter} iterations")
        print(f"  E_nuclear    = {E_nuc:>18.10f} Hartree")
        print(f"  E_electronic = {E_elec_final:>18.10f} Hartree")
        print(f"  E_total      = {E_total_final:>18.10f} Hartree")

    return RHFResult(
        energy_total=E_total_final,
        energy_electronic=E_elec_final,
        energy_nuclear=E_nuc,
        mo_energies=eps_final,
        mo_coefficients=C_final,
        density_matrix=P,
        fock_matrix=F_final,
        n_iter=n_iter,
        converged=converged,
        energy_history=energy_history,
    )
