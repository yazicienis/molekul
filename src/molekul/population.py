"""
Mulliken population analysis and molecular dipole moment.

Theory
------
Mulliken population analysis (Mulliken 1955):

    PS             — AO population matrix, P is density matrix, S is overlap
    N_μ = (PS)_{μμ} — gross orbital population
    N_A = Σ_{μ∈A} N_μ — gross atomic population
    q_A = Z_A − N_A   — Mulliken charge on atom A

Total electrons check:  Σ_A N_A = tr(PS) = N_elec

Dipole moment:
    μ = Σ_A Z_A R_A  −  Σ_{μν} P_{μν} ⟨φ_μ|r|φ_ν⟩

    Nuclear term: positive charges at nuclear positions
    Electronic term: negative charges smeared as ρ(r) = Σ_{μν} P_{μν} φ_μ φ_ν

    Units: atomic units (ea_0).  Conversion: 1 ea_0 = 2.541746473 D

Dipole integrals  D^c_{μν} = ⟨φ_μ|r_c|φ_ν⟩  via the shift identity:

    r_c · G_A(l_c) = G_A(l_c + 1) + A_c · G_A(l_c)

    ⟨G_A(lx,ly,lz)| x |G_B(lx',ly',lz')⟩
        = S(A,lx+1,ly,lz; B,lx',ly',lz') + Ax · S(A,lx,ly,lz; B,lx',ly',lz')

This reuses overlap_primitive from integrals.py; no new recurrence is needed.

Known limitations of Mulliken analysis
---------------------------------------
1. Basis-set dependence: charges change significantly with basis; STO-3G
   gives qualitative but not quantitative charges.
2. No physical observable: Mulliken charges are not measurable; they depend
   on the arbitrary partitioning of the overlap.
3. Diffuse basis functions: with augmented (+ or aug-cc) basis sets Mulliken
   populations can be negative or exceed Z — the analysis breaks down.
4. Better alternatives: Natural Population Analysis (NPA/NBO), QTAIM, Hirshfeld.

Public API
----------
build_dipole_integrals  — three (n_basis, n_basis) matrices D_x, D_y, D_z
mulliken_populations    — MullikenResult (populations + charges)
dipole_moment           — DipoleResult  (au and Debye)
analyze                 — convenience: run both from an RHFResult

References
----------
Mulliken, J. Chem. Phys. 23, 1833 (1955).
Szabo & Ostlund, "Modern Quantum Chemistry", §3.4 and Appendix B.
Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure Theory", §9.7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .basis import BasisSet, norm_primitive
from .integrals import build_overlap, overlap_primitive
from .molecule import Molecule
from .rhf import RHFResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEBYE_PER_AU = 2.541746473   # 1 ea_0 = 2.5418 D


# ---------------------------------------------------------------------------
# Dipole integrals
# ---------------------------------------------------------------------------

def _dipole_primitive_x(lx1, ly1, lz1, A, a,
                         lx2, ly2, lz2, B, b) -> float:
    """
    ⟨G_A(lx1,ly1,lz1)| x |G_B(lx2,ly2,lz2)⟩  for un-normalized primitives.

    Uses:  x · G_A(lx) = G_A(lx+1) + Ax · G_A(lx)
    """
    s_raised = overlap_primitive(lx1 + 1, ly1, lz1, A, a,
                                  lx2,     ly2, lz2, B, b)
    s_plain  = overlap_primitive(lx1,     ly1, lz1, A, a,
                                  lx2,     ly2, lz2, B, b)
    return s_raised + A[0] * s_plain


def _dipole_primitive_y(lx1, ly1, lz1, A, a,
                         lx2, ly2, lz2, B, b) -> float:
    """⟨G_A| y |G_B⟩ for un-normalized primitives."""
    s_raised = overlap_primitive(lx1, ly1 + 1, lz1, A, a,
                                  lx2, ly2,     lz2, B, b)
    s_plain  = overlap_primitive(lx1, ly1,     lz1, A, a,
                                  lx2, ly2,     lz2, B, b)
    return s_raised + A[1] * s_plain


def _dipole_primitive_z(lx1, ly1, lz1, A, a,
                         lx2, ly2, lz2, B, b) -> float:
    """⟨G_A| z |G_B⟩ for un-normalized primitives."""
    s_raised = overlap_primitive(lx1, ly1, lz1 + 1, A, a,
                                  lx2, ly2, lz2,     B, b)
    s_plain  = overlap_primitive(lx1, ly1, lz1,     A, a,
                                  lx2, ly2, lz2,     B, b)
    return s_raised + A[2] * s_plain


def _contracted_dipole(func_prim, bf1, A, bf2, B) -> float:
    """
    Evaluate a contracted dipole component by summing over primitive pairs.
    """
    lx1, ly1, lz1, sh1 = bf1
    lx2, ly2, lz2, sh2 = bf2
    N1 = sh1.norms(lx1, ly1, lz1)
    N2 = sh2.norms(lx2, ly2, lz2)
    result = 0.0
    for a, c1, n1 in zip(sh1.exponents, sh1.coefficients, N1):
        for b, c2, n2 in zip(sh2.exponents, sh2.coefficients, N2):
            result += n1 * c1 * n2 * c2 * func_prim(
                lx1, ly1, lz1, A, float(a),
                lx2, ly2, lz2, B, float(b),
            )
    return result


def build_dipole_integrals(
        basis: BasisSet,
        molecule: Molecule,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the three N×N dipole integral matrices D_x, D_y, D_z.

    D^c_{μν} = ⟨φ_μ | r_c | φ_ν⟩   (c ∈ {x, y, z}), bohr units.

    These are symmetric: D^c_{μν} = D^c_{νμ}.

    Parameters
    ----------
    basis    : BasisSet
    molecule : Molecule (coordinates in bohr)

    Returns
    -------
    Dx, Dy, Dz : each (n_basis, n_basis) float64 array
    """
    bfs    = basis.basis_functions(molecule)
    coords = np.array([a.coords for a in molecule.atoms])
    n      = len(bfs)
    Dx = np.zeros((n, n))
    Dy = np.zeros((n, n))
    Dz = np.zeros((n, n))

    for i, (ai, lx1, ly1, lz1, sh1) in enumerate(bfs):
        A = coords[ai]
        for j, (aj, lx2, ly2, lz2, sh2) in enumerate(bfs):
            if j < i:
                Dx[i, j] = Dx[j, i]
                Dy[i, j] = Dy[j, i]
                Dz[i, j] = Dz[j, i]
                continue
            B = coords[aj]
            bf1 = (lx1, ly1, lz1, sh1)
            bf2 = (lx2, ly2, lz2, sh2)
            Dx[i, j] = _contracted_dipole(_dipole_primitive_x, bf1, A, bf2, B)
            Dy[i, j] = _contracted_dipole(_dipole_primitive_y, bf1, A, bf2, B)
            Dz[i, j] = _contracted_dipole(_dipole_primitive_z, bf1, A, bf2, B)

    return Dx, Dy, Dz


# ---------------------------------------------------------------------------
# Mulliken population analysis
# ---------------------------------------------------------------------------

@dataclass
class MullikenResult:
    """
    Container for Mulliken population analysis results.

    Attributes
    ----------
    gross_orbital_pop : (n_basis,) — gross orbital populations N_μ = (PS)_{μμ}
    gross_atomic_pop  : (n_atoms,) — gross atomic populations N_A = Σ_{μ∈A} N_μ
    mulliken_charges  : (n_atoms,) — q_A = Z_A - N_A
    total_electrons   : float      — Σ_A N_A = tr(PS)  (should equal N_elec)
    atom_symbols      : list of str — for display
    orbital_labels    : list of str — for display
    """
    gross_orbital_pop: np.ndarray
    gross_atomic_pop: np.ndarray
    mulliken_charges: np.ndarray
    total_electrons: float
    atom_symbols: List[str]
    orbital_labels: List[str]


def mulliken_populations(
        P: np.ndarray,
        S: np.ndarray,
        basis: BasisSet,
        molecule: Molecule,
) -> MullikenResult:
    """
    Compute Mulliken gross atomic populations and charges.

    Parameters
    ----------
    P        : (n_basis, n_basis) density matrix  (RHF: P_{μν} = 2 Σ_i C_{μi} C_{νi})
    S        : (n_basis, n_basis) overlap matrix
    basis    : BasisSet
    molecule : Molecule

    Returns
    -------
    MullikenResult
    """
    PS = P @ S   # (n_basis, n_basis)

    # Gross orbital populations: diagonal of PS
    gross_orbital_pop = np.diag(PS).copy()   # (n_basis,)

    # Map each basis function to its atom index
    bfs = basis.basis_functions(molecule)    # list of (atom_idx, lx, ly, lz, shell)

    # Orbital labels (for display)
    _lxyz_to_name = {
        (0, 0, 0): "s",
        (1, 0, 0): "px", (0, 1, 0): "py", (0, 0, 1): "pz",
        (2, 0, 0): "dx2", (0, 2, 0): "dy2", (0, 0, 2): "dz2",
        (1, 1, 0): "dxy", (1, 0, 1): "dxz", (0, 1, 1): "dyz",
    }
    orbital_labels = []
    for ai, lx, ly, lz, sh in bfs:
        sym = molecule.atoms[ai].symbol
        orb = _lxyz_to_name.get((lx, ly, lz), f"l{lx}{ly}{lz}")
        orbital_labels.append(f"{sym}{ai+1}-{orb}")

    # Gross atomic populations: sum orbital pops per atom
    n_atoms = molecule.n_atoms
    gross_atomic_pop = np.zeros(n_atoms)
    for mu, (ai, *_) in enumerate(bfs):
        gross_atomic_pop[ai] += gross_orbital_pop[mu]

    # Mulliken charges
    Z = np.array([float(a.Z) for a in molecule.atoms])
    mulliken_charges = Z - gross_atomic_pop

    total_electrons = float(gross_orbital_pop.sum())

    atom_symbols = [a.symbol for a in molecule.atoms]

    return MullikenResult(
        gross_orbital_pop=gross_orbital_pop,
        gross_atomic_pop=gross_atomic_pop,
        mulliken_charges=mulliken_charges,
        total_electrons=total_electrons,
        atom_symbols=atom_symbols,
        orbital_labels=orbital_labels,
    )


# ---------------------------------------------------------------------------
# Dipole moment
# ---------------------------------------------------------------------------

@dataclass
class DipoleResult:
    """
    Molecular dipole moment in atomic units and Debye.

    The dipole is computed relative to the coordinate origin.
    For a neutral molecule the result is origin-independent.
    For a charged system (e.g. HeH+) the dipole depends on the origin.

    Attributes
    ----------
    nuclear_au    : (3,) — nuclear contribution Σ_A Z_A R_A  [ea_0]
    electronic_au : (3,) — electronic contribution −Σ_{μν} P_{μν} D_{μν}  [ea_0]
    total_au      : (3,) — total dipole vector  [ea_0]
    total_debye   : (3,) — total dipole vector  [Debye]
    magnitude_au   : float — |μ| in ea_0
    magnitude_debye: float — |μ| in Debye
    """
    nuclear_au:     np.ndarray
    electronic_au:  np.ndarray
    total_au:       np.ndarray
    total_debye:    np.ndarray
    magnitude_au:   float
    magnitude_debye: float


def dipole_moment(
        P: np.ndarray,
        basis: BasisSet,
        molecule: Molecule,
        *,
        Dx: np.ndarray | None = None,
        Dy: np.ndarray | None = None,
        Dz: np.ndarray | None = None,
) -> DipoleResult:
    """
    Compute the molecular electric dipole moment.

    μ_c = Σ_A Z_A R_{Ac}  −  Σ_{μν} P_{μν} D^c_{μν}

    Parameters
    ----------
    P        : (n_basis, n_basis) density matrix
    basis    : BasisSet
    molecule : Molecule
    Dx, Dy, Dz : pre-built dipole integral matrices (computed if not given)

    Returns
    -------
    DipoleResult
    """
    if Dx is None or Dy is None or Dz is None:
        Dx, Dy, Dz = build_dipole_integrals(basis, molecule)

    # Nuclear contribution
    coords = molecule.coords_bohr   # (n_atoms, 3)
    Z      = np.array([float(a.Z) for a in molecule.atoms])
    mu_nuc = (Z[:, None] * coords).sum(axis=0)   # (3,)

    # Electronic contribution: −Σ_{μν} P_{μν} D^c_{μν}  = − tr(P D^c)
    # Note: both P and D are symmetric, so tr(PD) = Σ_{μν} P_{μν} D_{νμ} = Σ_{μν} P_{μν} D_{μν}
    mu_elec = np.array([
        -float(np.einsum("mn,mn->", P, Dx)),
        -float(np.einsum("mn,mn->", P, Dy)),
        -float(np.einsum("mn,mn->", P, Dz)),
    ])

    total_au    = mu_nuc + mu_elec
    total_debye = total_au * DEBYE_PER_AU
    mag_au      = float(np.linalg.norm(total_au))
    mag_debye   = float(np.linalg.norm(total_debye))

    return DipoleResult(
        nuclear_au=mu_nuc,
        electronic_au=mu_elec,
        total_au=total_au,
        total_debye=total_debye,
        magnitude_au=mag_au,
        magnitude_debye=mag_debye,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

@dataclass
class PopulationResult:
    """Combined Mulliken + dipole results."""
    mulliken:  MullikenResult
    dipole:    DipoleResult
    S:         np.ndarray   # overlap matrix (stored for downstream use)
    Dx: np.ndarray
    Dy: np.ndarray
    Dz: np.ndarray


def analyze(
        molecule: Molecule,
        basis: BasisSet,
        result: RHFResult,
) -> PopulationResult:
    """
    Run full Mulliken population analysis and dipole moment from an RHFResult.

    Parameters
    ----------
    molecule : Molecule (bohr)
    basis    : BasisSet
    result   : RHFResult from rhf_scf()

    Returns
    -------
    PopulationResult with .mulliken and .dipole sub-results
    """
    S = build_overlap(basis, molecule)
    Dx, Dy, Dz = build_dipole_integrals(basis, molecule)

    mull  = mulliken_populations(result.density_matrix, S, basis, molecule)
    dip   = dipole_moment(result.density_matrix, basis, molecule,
                          Dx=Dx, Dy=Dy, Dz=Dz)

    return PopulationResult(mulliken=mull, dipole=dip, S=S, Dx=Dx, Dy=Dy, Dz=Dz)
