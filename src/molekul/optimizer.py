"""
Geometry optimizer for RHF energies using numerical gradients.

Algorithm
---------
scipy.optimize.minimize with method='BFGS'.  The energy and gradient are
both supplied (analytical-quality Hessian approximation via BFGS update).

Coordinates are optimised in Cartesian bohr.  Translations and rotations
have near-zero numerical gradients and are not explicitly projected out;
the BFGS directions along them remain small and do not cause problems for
small molecules.

Outputs
-------
trajectory : multi-frame XYZ file (Angstrom, one frame per optimizer step)
history    : JSON file with per-step energy, gradient norms, and coordinates

Convergence criteria (checked at every step via scipy callback)
---------------------------------------------------------------
Either scipy's own BFGS convergence (||grad|| < gtol) or:
    max |dE/dR| < grad_tol  (Hartree/bohr)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from .atoms import Atom
from .basis import BasisSet
from .grad import numerical_gradient, gradient_norm, max_gradient
from .io_xyz import write_xyz_trajectory
from .molecule import Molecule
from .rhf import rhf_scf


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    """Container for geometry optimisation output."""
    converged: bool
    energy_initial: float          # Hartree
    energy_final: float            # Hartree
    n_steps: int                   # number of gradient evaluations
    final_molecule: Molecule
    grad_rms_final: float          # RMS gradient (Hartree/bohr)
    grad_max_final: float          # max |gradient| (Hartree/bohr)
    energy_history: List[float]    # E at each gradient evaluation
    grad_rms_history: List[float]  # RMS grad at each gradient evaluation
    grad_max_history: List[float]  # max |grad| at each gradient evaluation
    trajectory: List[Molecule]     # one Molecule per gradient evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mol_from_flat(x: np.ndarray, template: Molecule) -> Molecule:
    """Reconstruct a Molecule from a flat (3*n_atoms,) coordinate array (bohr)."""
    coords = x.reshape(-1, 3)
    atoms = [
        Atom(template.atoms[i].symbol, coords[i].copy())
        for i in range(template.n_atoms)
    ]
    return Molecule(
        atoms=atoms,
        charge=template.charge,
        multiplicity=template.multiplicity,
        name=template.name,
    )


def _mol_to_flat(molecule: Molecule) -> np.ndarray:
    """Return a flat (3*n_atoms,) array of Cartesian coordinates in bohr."""
    return np.array([a.coords for a in molecule.atoms], dtype=float).ravel()


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------

def optimize_geometry(
        molecule: Molecule,
        basis: BasisSet,
        *,
        grad_tol: float = 1e-4,      # max |dE/dR| convergence threshold (Ha/bohr)
        max_steps: int = 100,
        h_grad: float = 1e-3,        # finite-difference step for gradient (bohr)
        traj_path: Optional[str] = None,
        history_path: Optional[str] = None,
        verbose: bool = True,
) -> OptResult:
    """
    Optimise molecular geometry by minimising the RHF total energy.

    Parameters
    ----------
    molecule     : initial Molecule (all-bohr internally)
    basis        : BasisSet (e.g. STO3G)
    grad_tol     : max absolute gradient component for convergence (Ha/bohr)
    max_steps    : maximum number of gradient evaluations (scipy maxiter)
    h_grad       : finite-difference step for numerical gradient (bohr)
    traj_path    : if given, write multi-frame XYZ trajectory here
    history_path : if given, write JSON optimisation history here
    verbose      : print progress table if True

    Returns
    -------
    OptResult
    """
    # Storage accumulated during optimisation
    energy_history: List[float] = []
    grad_rms_history: List[float] = []
    grad_max_history: List[float] = []
    trajectory: List[Molecule] = []
    call_count = [0]   # mutable counter for closure

    def energy_and_grad(x: np.ndarray):
        """Objective: returns (E, flat_gradient) at coordinates x."""
        mol = _mol_from_flat(x, molecule)
        result = rhf_scf(mol, basis)
        E = result.energy_total

        grad = numerical_gradient(mol, basis, h=h_grad)
        g_rms = gradient_norm(grad)
        g_max = max_gradient(grad)

        energy_history.append(E)
        grad_rms_history.append(g_rms)
        grad_max_history.append(g_max)
        trajectory.append(mol)
        call_count[0] += 1

        step = call_count[0]
        if verbose:
            print(f"  step {step:>3d}  E = {E:>18.10f} Ha  "
                  f"|g|_rms = {g_rms:.3e}  |g|_max = {g_max:.3e} Ha/bohr")

        return E, grad.ravel()

    x0 = _mol_to_flat(molecule)
    E0 = rhf_scf(molecule, basis).energy_total

    if verbose:
        print(f"\nGeometry optimisation — BFGS / numerical gradient")
        print(f"Molecule : {molecule.name or 'unnamed'}  "
              f"({molecule.n_atoms} atoms, {molecule.n_electrons} electrons)")
        print(f"Basis    : STO-3G    grad step h = {h_grad:.0e} bohr")
        print(f"Tol      : max|grad| < {grad_tol:.0e} Ha/bohr   max_steps = {max_steps}")
        print(f"{'─'*72}")

    opt = minimize(
        fun=energy_and_grad,
        x0=x0,
        jac=True,          # energy_and_grad returns (f, grad) together
        method="BFGS",
        options={
            "gtol": grad_tol,
            "maxiter": max_steps,
            "disp": False,
        },
    )

    final_mol = _mol_from_flat(opt.x, molecule)
    g_final = numerical_gradient(final_mol, basis, h=h_grad)
    g_rms_final = gradient_norm(g_final)
    g_max_final = max_gradient(g_final)
    E_final = rhf_scf(final_mol, basis).energy_total

    converged = opt.success or (g_max_final < grad_tol)

    if verbose:
        print(f"{'─'*72}")
        status = "CONVERGED" if converged else "NOT CONVERGED (max_steps reached)"
        print(f"\n{status}")
        print(f"  Steps          : {call_count[0]}")
        print(f"  E_initial      : {E0:>18.10f} Ha")
        print(f"  E_final        : {E_final:>18.10f} Ha")
        print(f"  ΔE             : {E_final - E0:>+.6e} Ha")
        print(f"  |g|_max final  : {g_max_final:.3e} Ha/bohr")

    # --- Write trajectory ---
    if traj_path is not None:
        p = Path(traj_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Attach energy as comment in each frame
        for k, (mol_k, E_k) in enumerate(zip(trajectory, energy_history)):
            mol_k.name = f"step_{k+1}  E={E_k:.10f}_Ha"
        write_xyz_trajectory(trajectory, p)
        if verbose:
            print(f"  Trajectory     : {p}")

    # --- Write JSON history ---
    if history_path is not None:
        p = Path(history_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        history = {
            "molecule": molecule.name,
            "basis": "STO-3G",
            "method": "RHF/numerical-BFGS",
            "h_grad_bohr": h_grad,
            "grad_tol": grad_tol,
            "converged": converged,
            "n_steps": call_count[0],
            "energy_initial_Ha": E0,
            "energy_final_Ha": E_final,
            "grad_rms_final_Ha_per_bohr": g_rms_final,
            "grad_max_final_Ha_per_bohr": g_max_final,
            "steps": [
                {
                    "step": i + 1,
                    "energy_Ha": energy_history[i],
                    "grad_rms_Ha_per_bohr": grad_rms_history[i],
                    "grad_max_Ha_per_bohr": grad_max_history[i],
                    "coords_bohr": trajectory[i].coords_bohr.tolist(),
                }
                for i in range(len(energy_history))
            ],
        }
        p.write_text(json.dumps(history, indent=2))
        if verbose:
            print(f"  History        : {p}")

    return OptResult(
        converged=converged,
        energy_initial=E0,
        energy_final=E_final,
        n_steps=call_count[0],
        final_molecule=final_mol,
        grad_rms_final=g_rms_final,
        grad_max_final=g_max_final,
        energy_history=energy_history,
        grad_rms_history=grad_rms_history,
        grad_max_history=grad_max_history,
        trajectory=trajectory,
    )
