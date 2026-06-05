# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Enis Yazici

"""
Nuclear gradients of the RHF total energy.

dE/dR_{Ai} is approximated by central finite differences:

    dE/dR_{Ai} ≈ [E(R + h·ê_{Ai}) − E(R − h·ê_{Ai})] / (2h)

where ê_{Ai} is the unit displacement of atom A along Cartesian axis i.

Accuracy: O(h²) truncation error.  For h = 1e-3 bohr this is ~1e-6 Ha/bohr,
which is well below typical geometry-optimisation convergence thresholds.

Note: because six of the 3N degrees of freedom are translations/rotations
(with zero gradient), the gradient along those modes will be numerically small
but not exactly zero.  The optimizer handles this gracefully.
"""

from __future__ import annotations

import numpy as np

from .atoms import Atom
from .molecule import Molecule
from .basis import BasisSet
from .rhf import RHFResult, rhf_scf
from .integrals import build_overlap, build_core_hamiltonian
from .eri import build_eri


_INTEGRAL_DERIV_STEP = 1e-4


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _displaced(molecule: Molecule, atom_idx: int, coord_idx: int, delta: float) -> Molecule:
    """Return a copy of molecule with atom_idx coordinate coord_idx shifted by delta (bohr)."""
    new_atoms = []
    for k, atom in enumerate(molecule.atoms):
        coords = atom.coords.copy()
        if k == atom_idx:
            coords[coord_idx] += delta
        new_atoms.append(Atom(atom.symbol, coords))
    return Molecule(
        atoms=new_atoms,
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
        name=molecule.name,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def numerical_gradient(
        molecule: Molecule,
        basis: BasisSet,
        *,
        h: float = 1e-3,
        scf_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Compute the RHF nuclear gradient by central finite differences.

    Parameters
    ----------
    molecule   : Molecule at which to evaluate the gradient
    basis      : BasisSet (e.g. STO3G)
    h          : finite-difference step size in bohr (default 1e-3)
    scf_kwargs : optional dict of keyword arguments forwarded to rhf_scf

    Returns
    -------
    grad : np.ndarray, shape (n_atoms, 3)
        dE/dR_{Ai} in Hartree/bohr for each atom A and Cartesian axis i.
    """
    if scf_kwargs is None:
        scf_kwargs = {}

    n_atoms = molecule.n_atoms
    grad = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        for j in range(3):
            mol_p = _displaced(molecule, i, j, +h)
            mol_m = _displaced(molecule, i, j, -h)
            E_p = rhf_scf(mol_p, basis, **scf_kwargs).energy_total
            E_m = rhf_scf(mol_m, basis, **scf_kwargs).energy_total
            grad[i, j] = (E_p - E_m) / (2.0 * h)

    return grad



def _matrix_derivative(builder, basis: BasisSet, molecule: Molecule,
                       h: float = _INTEGRAL_DERIV_STEP) -> np.ndarray:
    """Central finite-difference derivative of an integral matrix builder."""
    ref = builder(basis, molecule)
    deriv = np.zeros((molecule.n_atoms, 3, *ref.shape), dtype=float)
    for atom_idx in range(molecule.n_atoms):
        for coord_idx in range(3):
            mol_p = _displaced(molecule, atom_idx, coord_idx, +h)
            mol_m = _displaced(molecule, atom_idx, coord_idx, -h)
            deriv[atom_idx, coord_idx] = (
                builder(basis, mol_p) - builder(basis, mol_m)
            ) / (2.0 * h)
    return deriv


def overlap_derivative(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """dS_mn/dR_Ax, shape (n_atoms, 3, n_basis, n_basis)."""
    return _matrix_derivative(build_overlap, basis, molecule)


def hcore_derivative(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """d(T+V)_mn/dR_Ax, shape (n_atoms, 3, n_basis, n_basis)."""
    return _matrix_derivative(build_core_hamiltonian, basis, molecule)


def eri_derivative(basis: BasisSet, molecule: Molecule) -> np.ndarray:
    """d(mn|ls)/dR_Ax, shape (n_atoms, 3, n_basis, n_basis, n_basis, n_basis)."""
    return _matrix_derivative(build_eri, basis, molecule)


def nuclear_repulsion_derivative(molecule: Molecule) -> np.ndarray:
    """Derivative of nuclear repulsion energy, shape (n_atoms, 3)."""
    grad = np.zeros((molecule.n_atoms, 3), dtype=float)
    for i, atom_i in enumerate(molecule.atoms):
        Zi = float(atom_i.Z)
        Ri = atom_i.coords
        for j, atom_j in enumerate(molecule.atoms):
            if i == j:
                continue
            Zj = float(atom_j.Z)
            diff = Ri - atom_j.coords
            r = np.linalg.norm(diff)
            grad[i] -= Zi * Zj * diff / (r ** 3)
    return grad


def rhf_gradient(
        mol: Molecule,
        basis_fn: BasisSet,
        rhf_result: RHFResult,
) -> np.ndarray:
    """
    RHF nuclear gradient in Hartree/Bohr, shape (n_atoms, 3).

    Phase 18 validates STO-3G. Integral derivatives are evaluated by central
    finite differences of the existing integral builders while the electronic
    response is handled by the stationary RHF gradient expression.
    """
    if not rhf_result.converged:
        raise ValueError("RHF gradient requires a converged RHF result.")

    P = rhf_result.density_matrix
    C = rhf_result.mo_coefficients
    eps = rhf_result.mo_energies
    n_occ = mol.n_electrons // 2

    C_occ = C[:, :n_occ]
    eps_occ = eps[:n_occ]
    W = 2.0 * (C_occ * eps_occ) @ C_occ.T

    dS = overlap_derivative(basis_fn, mol)
    dH = hcore_derivative(basis_fn, mol)
    dERI = eri_derivative(basis_fn, mol)
    dEnuc = nuclear_repulsion_derivative(mol)

    grad = np.zeros((mol.n_atoms, 3), dtype=float)
    for atom_idx in range(mol.n_atoms):
        for coord_idx in range(3):
            dS_x = dS[atom_idx, coord_idx]
            dH_x = dH[atom_idx, coord_idx]
            dERI_x = dERI[atom_idx, coord_idx]

            one_e = np.einsum("mn,mn->", P, dH_x)
            coul = 0.5 * np.einsum("mn,ls,mnls->", P, P, dERI_x)
            exch = -0.25 * np.einsum("ml,ns,mnls->", P, P, dERI_x)
            pulay = -np.einsum("mn,mn->", W, dS_x)

            grad[atom_idx, coord_idx] = one_e + coul + exch + pulay + dEnuc[atom_idx, coord_idx]

    grad -= grad.sum(axis=0, keepdims=True) / mol.n_atoms
    return grad


def gradient_norm(grad: np.ndarray) -> float:
    """RMS norm of the gradient: sqrt(mean(g²)), in Hartree/bohr."""
    return float(np.sqrt(np.mean(grad ** 2)))


def max_gradient(grad: np.ndarray) -> float:
    """Maximum absolute gradient component, in Hartree/bohr."""
    return float(np.max(np.abs(grad)))
