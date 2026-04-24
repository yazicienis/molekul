"""
Numerical nuclear gradient of the RHF total energy.

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
from .rhf import rhf_scf


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


def gradient_norm(grad: np.ndarray) -> float:
    """RMS norm of the gradient: sqrt(mean(g²)), in Hartree/bohr."""
    return float(np.sqrt(np.mean(grad ** 2)))


def max_gradient(grad: np.ndarray) -> float:
    """Maximum absolute gradient component, in Hartree/bohr."""
    return float(np.max(np.abs(grad)))
