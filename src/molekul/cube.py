"""
CUBE file export for electron density and molecular orbitals.

Gaussian CUBE format
--------------------
The CUBE format stores volumetric scalar data on a regular 3-D grid in
atomic units (bohr).  File layout:

    Line 1  : free comment
    Line 2  : free comment
    Line 3  : NATOMS  XORIG  YORIG  ZORIG          (bohr)
    Line 4  : NX  DX  0  0    (voxels along x, step vector x)
    Line 5  : NY   0  DY 0
    Line 6  : NZ   0   0 DZ
    Lines 7…: Z  CHARGE  X  Y  Z  (one per atom, bohr)
    Data    : values, 6 per line, outer→inner loop: x → y → z

Coordinates are all in bohr.  Density is in e/bohr³; orbital amplitude
in bohr^{-3/2}.  Both are read correctly by VMD, VESTA, Avogadro, and Jmol.

Public API
----------
make_grid            — auto-compute bounding-box grid
eval_basis_grid      — evaluate all contracted basis functions at grid pts
eval_density_grid    — compute ρ(r) = Σ_{μν} P_{μν} φ_μ φ_ν
eval_orbital_grid    — compute ψ_i(r) = Σ_μ C_{μi} φ_μ
write_cube           — write a 3-D array to a .cube file
export_density       — convenience: RHF result → density CUBE
export_orbital       — convenience: RHF result + MO index → orbital CUBE

References
----------
Gaussian manual: https://gaussian.com/cubegen/
VMD CUBE reader:  https://www.ks.uiuc.edu/Research/vmd/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .basis import BasisSet, norm_primitive
from .molecule import Molecule
from .rhf import RHFResult


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def make_grid(
        molecule: Molecule,
        margin: float = 4.0,
        step: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Build an axis-aligned regular grid that encloses the molecule.

    Parameters
    ----------
    molecule : Molecule (coordinates in bohr)
    margin   : padding beyond the outermost atoms on each side (bohr)
    step     : voxel edge length (bohr)

    Returns
    -------
    origin : (3,) array — grid origin in bohr
    steps  : (3,) array — voxel size [dx, dy, dz] in bohr (equal here)
    shape  : (nx, ny, nz) integer tuple
    """
    coords = molecule.coords_bohr        # (n_atoms, 3)
    lo = coords.min(axis=0) - margin
    hi = coords.max(axis=0) + margin
    nx = int(np.ceil((hi[0] - lo[0]) / step)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / step)) + 1
    nz = int(np.ceil((hi[2] - lo[2]) / step)) + 1
    origin = lo
    steps  = np.array([step, step, step])
    return origin, steps, (nx, ny, nz)


def grid_points(
        origin: np.ndarray,
        steps: np.ndarray,
        shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Return all grid-point Cartesian coordinates.

    Parameters
    ----------
    origin : (3,)  grid origin in bohr
    steps  : (3,)  voxel sizes [dx, dy, dz]
    shape  : (nx, ny, nz)

    Returns
    -------
    pts : (nx*ny*nz, 3) array in bohr, order (x outer, z inner)
    """
    nx, ny, nz = shape
    xs = origin[0] + np.arange(nx) * steps[0]
    ys = origin[1] + np.arange(ny) * steps[1]
    zs = origin[2] + np.arange(nz) * steps[2]
    # meshgrid indexing='ij' gives (nx, ny, nz) arrays
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts.astype(np.float64)


# ---------------------------------------------------------------------------
# Basis function evaluation on a grid
# ---------------------------------------------------------------------------

def eval_basis_grid(
        basis: BasisSet,
        molecule: Molecule,
        pts: np.ndarray,
) -> np.ndarray:
    """
    Evaluate all contracted Cartesian Gaussian basis functions at grid points.

    Each contracted function φ_μ is:
        φ_μ(r) = Σ_k N_k c_k (x-Ax)^lx (y-Ay)^ly (z-Az)^lz exp(-α_k |r-A|²)

    where N_k = norm_primitive(α_k, lx, ly, lz).

    Parameters
    ----------
    basis    : BasisSet
    molecule : Molecule (coords in bohr)
    pts      : (N, 3) array of evaluation points (bohr)

    Returns
    -------
    phi : (n_basis, N) array — φ_μ(r_i) in bohr^{-3/2}
    """
    bfs   = basis.basis_functions(molecule)
    coords = molecule.coords_bohr        # (n_atoms, 3)
    N      = pts.shape[0]
    n_bf   = len(bfs)
    phi    = np.zeros((n_bf, N))

    for mu, (ai, lx, ly, lz, sh) in enumerate(bfs):
        A   = coords[ai]           # (3,)
        rA  = pts - A              # (N, 3)
        r2A = np.einsum("ij,ij->i", rA, rA)   # (N,)

        # Polynomial part (handles lx=ly=lz=0 correctly: 1^0 = 1)
        xA = rA[:, 0]; yA = rA[:, 1]; zA = rA[:, 2]
        if lx == 0 and ly == 0 and lz == 0:
            ang = np.ones(N)
        else:
            ang = (xA ** lx) * (yA ** ly) * (zA ** lz)

        # Radial part: sum over primitives
        radial = np.zeros(N)
        for alpha, coeff in zip(sh.exponents, sh.coefficients):
            nk = norm_primitive(float(alpha), lx, ly, lz)
            radial += nk * coeff * np.exp(-alpha * r2A)

        phi[mu] = ang * radial

    return phi


# ---------------------------------------------------------------------------
# Density and orbital evaluation
# ---------------------------------------------------------------------------

def eval_density_grid(
        P: np.ndarray,
        phi: np.ndarray,
) -> np.ndarray:
    """
    Compute the electron density at grid points.

    ρ(r) = Σ_{μν} P_{μν} φ_μ(r) φ_ν(r)

    For RHF, P already contains the factor of 2 for spin, so this gives
    the total (alpha + beta) electron density in electrons/bohr³.

    Parameters
    ----------
    P   : (n_basis, n_basis) density matrix (P_μν = 2 Σ_i C_μi C_νi for RHF)
    phi : (n_basis, N) basis-function values

    Returns
    -------
    rho : (N,) array  — electron density in e/bohr³
    """
    Pphi = P @ phi      # (n_basis, N)
    return np.einsum("mi,mi->i", phi, Pphi)   # Σ_μ φ_μ (Σ_ν P_μν φ_ν)


def eval_orbital_grid(
        C: np.ndarray,
        mo_idx: int,
        phi: np.ndarray,
) -> np.ndarray:
    """
    Compute a molecular orbital amplitude at grid points.

    ψ_i(r) = Σ_μ C_{μi} φ_μ(r)

    Note: orbital amplitude can be negative (unlike density).
    The MO is normalized: ∫ |ψ_i|² dr = 1.

    Parameters
    ----------
    C      : (n_basis, n_basis) MO coefficient matrix  (column mo_idx = MO i)
    mo_idx : 0-based MO index (0 = lowest energy)
    phi    : (n_basis, N) basis-function values

    Returns
    -------
    psi : (N,) array — ψ_{mo_idx}(r) in bohr^{-3/2}
    """
    c_i = C[:, mo_idx]           # (n_basis,)
    return c_i @ phi             # Σ_μ C_{μi} φ_μ  →  (N,)


# ---------------------------------------------------------------------------
# CUBE file writer
# ---------------------------------------------------------------------------

def write_cube(
        path: str | Path,
        molecule: Molecule,
        data_3d: np.ndarray,
        origin: np.ndarray,
        steps: np.ndarray,
        comment1: str = "CUBE file generated by MOLEKUL",
        comment2: str = "",
) -> None:
    """
    Write volumetric data to a Gaussian CUBE file.

    Parameters
    ----------
    path     : output file path (usually *.cube)
    molecule : Molecule with bohr coordinates
    data_3d  : (nx, ny, nz) array — scalar field on the grid
    origin   : (3,) grid origin in bohr
    steps    : (3,) voxel sizes [dx, dy, dz] in bohr
    comment1 : first comment line (≤ 72 chars recommended)
    comment2 : second comment line
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = data_3d.shape
    n_atoms = molecule.n_atoms

    lines = []

    # Header (2 comment lines)
    lines.append(comment1)
    lines.append(comment2 or " ")

    # Natoms + origin
    lines.append(f"  {n_atoms:5d}   {origin[0]:12.6f}   {origin[1]:12.6f}   {origin[2]:12.6f}")

    # Grid vectors (orthogonal: diagonal)
    lines.append(f"  {nx:5d}   {steps[0]:12.6f}    0.000000    0.000000")
    lines.append(f"  {ny:5d}    0.000000   {steps[1]:12.6f}    0.000000")
    lines.append(f"  {nz:5d}    0.000000    0.000000   {steps[2]:12.6f}")

    # Atomic data: Z  nuclear_charge  x  y  z
    for atom in molecule.atoms:
        Z = float(atom.Z)
        x, y, z = atom.coords
        lines.append(f"  {int(Z):5d}   {Z:12.6f}   {x:12.6f}   {y:12.6f}   {z:12.6f}")

    # Volumetric data: 6 values per line, x outer / z inner
    flat = data_3d.ravel()          # C-order = x outer, z inner  ✓
    data_lines = []
    for i in range(0, len(flat), 6):
        chunk = flat[i: i + 6]
        data_lines.append("  " + "  ".join(f"{v:12.5E}" for v in chunk))
    lines.extend(data_lines)

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def export_density(
        path: str | Path,
        molecule: Molecule,
        basis: BasisSet,
        result: RHFResult,
        margin: float = 4.0,
        step:   float = 0.2,
) -> Tuple[Path, dict]:
    """
    Export the RHF electron density to a CUBE file.

    Parameters
    ----------
    path     : output .cube path
    molecule : Molecule (bohr)
    basis    : BasisSet
    result   : RHFResult from rhf_scf()
    margin   : bohr padding around molecule (default 4.0)
    step     : voxel size in bohr (default 0.2)

    Returns
    -------
    (cube_path, stats)  where stats has keys:
        n_electrons_integrated, max_density, grid_shape
    """
    origin, steps, shape = make_grid(molecule, margin=margin, step=step)
    pts = grid_points(origin, steps, shape)

    phi = eval_basis_grid(basis, molecule, pts)
    rho = eval_density_grid(result.density_matrix, phi)

    # Numerical integral ∫ρ dr ≈ Σ_i ρ_i × dV
    dV = float(np.prod(steps))
    n_elec_integrated = float(rho.sum() * dV)

    rho_3d = rho.reshape(shape)

    write_cube(
        path,
        molecule,
        rho_3d,
        origin,
        steps,
        comment1="Electron density from RHF/STO-3G — MOLEKUL",
        comment2=f"molecule={molecule.name}  step={step:.3f}bohr  margin={margin:.1f}bohr",
    )

    stats = {
        "n_electrons_integrated": n_elec_integrated,
        "max_density":            float(rho.max()),
        "min_density":            float(rho.min()),
        "grid_shape":             shape,
        "n_grid_points":          int(np.prod(shape)),
        "dV_bohr3":               dV,
        "margin_bohr":            margin,
        "step_bohr":              step,
    }
    return Path(path), stats


def export_orbital(
        path: str | Path,
        molecule: Molecule,
        basis: BasisSet,
        result: RHFResult,
        mo_idx: int,
        margin: float = 4.0,
        step:   float = 0.2,
) -> Tuple[Path, dict]:
    """
    Export one molecular orbital to a CUBE file.

    Parameters
    ----------
    path     : output .cube path
    molecule : Molecule (bohr)
    basis    : BasisSet
    result   : RHFResult from rhf_scf()
    mo_idx   : 0-based MO index (0 = lowest energy, HOMO = n_occ-1)
    margin   : bohr padding (default 4.0)
    step     : voxel size in bohr (default 0.2)

    Returns
    -------
    (cube_path, stats)
    """
    n_occ = molecule.n_alpha
    label = (
        "HOMO"  if mo_idx == n_occ - 1 else
        "LUMO"  if mo_idx == n_occ     else
        f"MO{mo_idx + 1}"
    )
    eps_ev = result.mo_energies[mo_idx] * 27.211396132   # Ha → eV

    origin, steps, shape = make_grid(molecule, margin=margin, step=step)
    pts = grid_points(origin, steps, shape)

    phi = eval_basis_grid(basis, molecule, pts)
    psi = eval_orbital_grid(result.mo_coefficients, mo_idx, phi)

    # Numerical checks
    dV = float(np.prod(steps))
    norm_integrated = float((psi ** 2).sum() * dV)   # should ≈ 1

    psi_3d = psi.reshape(shape)

    write_cube(
        path,
        molecule,
        psi_3d,
        origin,
        steps,
        comment1=f"{label} (MO {mo_idx + 1}) from RHF/STO-3G — MOLEKUL",
        comment2=(f"molecule={molecule.name}  eps={eps_ev:.4f}eV  "
                  f"step={step:.3f}bohr  margin={margin:.1f}bohr"),
    )

    stats = {
        "mo_idx":          mo_idx,
        "mo_label":        label,
        "mo_energy_ha":    float(result.mo_energies[mo_idx]),
        "mo_energy_ev":    eps_ev,
        "norm_integrated": norm_integrated,
        "max_amplitude":   float(psi.max()),
        "min_amplitude":   float(psi.min()),
        "grid_shape":      shape,
        "n_grid_points":   int(np.prod(shape)),
        "dV_bohr3":        dV,
        "step_bohr":       step,
    }
    return Path(path), stats
