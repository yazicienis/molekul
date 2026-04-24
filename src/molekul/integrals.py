"""
One-electron integrals: overlap (S), kinetic (T), nuclear attraction (V).

Uses the McMurchie-Davidson (MD) recurrence scheme.
Reference: McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978).
           Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory",
           Chapter 9.

All coordinates in atomic units (bohr).
"""

from functools import lru_cache
from math import pi, sqrt, exp
from typing import List, Tuple

import numpy as np
from scipy.special import hyp1f1  # for Boys function via 1F1

from .basis import BasisSet, Shell, norm_primitive


# ---------------------------------------------------------------------------
# Boys function  F_n(x) = integral_0^1 t^{2n} exp(-x t^2) dt
# Relation to confluent hypergeometric:
#   F_n(x) = 1F1(n+1/2; n+3/2; -x) / (2n+1)
# ---------------------------------------------------------------------------
def _boys(n: int, x: float) -> float:
    if x < 1e-8:
        return 1.0 / (2 * n + 1)
    return hyp1f1(n + 0.5, n + 1.5, -x) / (2 * n + 1)


# ---------------------------------------------------------------------------
# McMurchie-Davidson E^{ij}_t coefficients
#
# Correct recurrence (MEST eq 9.2.11, Helgaker, Jørgensen, Olsen):
#   E^{i+1,j}_t = (1/2p)*E^{i,j}_{t-1} + X_PA*E^{i,j}_t + (t+1)*E^{i,j}_{t+1}
#   E^{i,j+1}_t = (1/2p)*E^{i,j}_{t-1} + X_PB*E^{i,j}_t + (t+1)*E^{i,j}_{t+1}
#
# where  Qx  = Ax - Bx,  P = (a*A + b*B)/(a+b),
#        X_PA = P - A = -b*Qx/p,   X_PB = P - B = a*Qx/p
#
# Boundary:  E^{0,0}_0 = exp(-mu*Qx^2),  E^{0,0}_{t≠0} = 0
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _E(i: int, j: int, t: int, Qx: float, a: float, b: float) -> float:
    """
    McMurchie-Davidson Hermite Gaussian expansion coefficient E^{ij}_t.
    Qx = Ax - Bx  (signed difference along one Cartesian axis).
    """
    p = a + b
    if t < 0 or t > i + j:
        return 0.0
    if i == 0 and j == 0:
        return exp(-a * b / p * Qx * Qx) if t == 0 else 0.0
    if i > 0:
        XPA = -b * Qx / p
        return (
            (1 / (2 * p)) * _E(i - 1, j, t - 1, Qx, a, b)
            + XPA * _E(i - 1, j, t, Qx, a, b)
            + (t + 1) * _E(i - 1, j, t + 1, Qx, a, b)
        )
    # j > 0, i == 0
    XPB = a * Qx / p
    return (
        (1 / (2 * p)) * _E(i, j - 1, t - 1, Qx, a, b)
        + XPB * _E(i, j - 1, t, Qx, a, b)
        + (t + 1) * _E(i, j - 1, t + 1, Qx, a, b)
    )


# ---------------------------------------------------------------------------
# Overlap integral between two primitive Cartesian Gaussians
# ---------------------------------------------------------------------------
def _overlap_1d(i: int, j: int, Qx: float, a: float, b: float) -> float:
    """1-D overlap component: integral x^i x^j exp(-a(x-Ax)^2) exp(-b(x-Bx)^2) dx / normalization."""
    return sqrt(pi / (a + b)) * _E(i, j, 0, Qx, a, b)


def overlap_primitive(lx1, ly1, lz1, A, a,
                      lx2, ly2, lz2, B, b) -> float:
    """
    Overlap integral <g1|g2> for two UN-normalized primitives.
    A, B: 3-element coordinates (bohr).
    """
    Qx, Qy, Qz = A[0] - B[0], A[1] - B[1], A[2] - B[2]
    sx = _overlap_1d(lx1, lx2, Qx, a, b)
    sy = _overlap_1d(ly1, ly2, Qy, a, b)
    sz = _overlap_1d(lz1, lz2, Qz, a, b)
    return sx * sy * sz


# ---------------------------------------------------------------------------
# Kinetic energy integral
# T_{ij} = b*(2*(lj_x+ly_j+lz_j)+3)*S_{ij}
#         - 2*b^2*(S_{i,j+2x} + S_{i,j+2y} + S_{i,j+2z})
#         - 0.5*(lj_x*(lj_x-1)*S_{i,j-2x} + ...)
# ---------------------------------------------------------------------------
def kinetic_primitive(lx1, ly1, lz1, A, a,
                      lx2, ly2, lz2, B, b) -> float:
    """Kinetic energy <g1|-1/2 nabla^2|g2> for UN-normalized primitives."""
    b2 = b * b

    def S(l1x, l1y, l1z, l2x, l2y, l2z):
        return overlap_primitive(l1x, l1y, l1z, A, a,
                                 l2x, l2y, l2z, B, b)

    # Diagonal term
    term0 = b * (2 * (lx2 + ly2 + lz2) + 3) * S(lx1, ly1, lz1, lx2, ly2, lz2)

    # +2 terms
    term_p2 = -2.0 * b2 * (
        S(lx1, ly1, lz1, lx2 + 2, ly2, lz2) +
        S(lx1, ly1, lz1, lx2, ly2 + 2, lz2) +
        S(lx1, ly1, lz1, lx2, ly2, lz2 + 2)
    )

    # -2 terms
    term_m2 = -0.5 * (
        lx2 * (lx2 - 1) * S(lx1, ly1, lz1, lx2 - 2, ly2, lz2) +
        ly2 * (ly2 - 1) * S(lx1, ly1, lz1, lx2, ly2 - 2, lz2) +
        lz2 * (lz2 - 1) * S(lx1, ly1, lz1, lx2, ly2, lz2 - 2)
    )

    return term0 + term_p2 + term_m2


# ---------------------------------------------------------------------------
# Hermite Coulomb auxiliary integral R^n_{tuv}(p, P, C)
# ---------------------------------------------------------------------------
def _R(t: int, u: int, v: int, n: int, p: float, PC: np.ndarray) -> float:
    """
    Auxiliary Hermite Coulomb integral via MD recurrence.
    PC = P - C  (P is Gaussian product centre, C is nuclear position).
    """
    PC2 = np.dot(PC, PC)
    val = 0.0
    if t == u == v == 0:
        return ((-2 * p) ** n) * _boys(n, p * PC2)
    if t == u == 0:
        if v > 1:
            val += (v - 1) * _R(t, u, v - 2, n + 1, p, PC)
        val += PC[2] * _R(t, u, v - 1, n + 1, p, PC)
    elif t == 0:
        if u > 1:
            val += (u - 1) * _R(t, u - 2, v, n + 1, p, PC)
        val += PC[1] * _R(t, u - 1, v, n + 1, p, PC)
    else:
        if t > 1:
            val += (t - 1) * _R(t - 2, u, v, n + 1, p, PC)
        val += PC[0] * _R(t - 1, u, v, n + 1, p, PC)
    return val


def nuclear_primitive(lx1, ly1, lz1, A, a,
                      lx2, ly2, lz2, B, b,
                      C: np.ndarray, Z: float) -> float:
    """
    Nuclear attraction integral <g1| -Z/|r-C| |g2> for UN-normalized primitives.
    C: nuclear position (bohr), Z: nuclear charge.
    """
    p = a + b
    P = (a * np.asarray(A) + b * np.asarray(B)) / p
    PC = P - np.asarray(C)
    Qx, Qy, Qz = np.asarray(A) - np.asarray(B)

    # Product E coefficients
    Ex = np.array([_E(lx1, lx2, t, Qx, a, b) for t in range(lx1 + lx2 + 1)])
    Ey = np.array([_E(ly1, ly2, u, Qy, a, b) for u in range(ly1 + ly2 + 1)])
    Ez = np.array([_E(lz1, lz2, v, Qz, a, b) for v in range(lz1 + lz2 + 1)])

    total = 0.0
    for t, ex in enumerate(Ex):
        for u, ey in enumerate(Ey):
            for v, ez in enumerate(Ez):
                total += ex * ey * ez * _R(t, u, v, 0, p, PC)

    return -Z * (2 * pi / p) * total


# ---------------------------------------------------------------------------
# Contracted integral builders
# ---------------------------------------------------------------------------
def _contracted_integral(func, bf1, A, bf2, B, **kwargs) -> float:
    """
    Evaluate a contracted integral by summing over all primitive pairs.

    bf1 = (lx1, ly1, lz1, Shell1)
    bf2 = (lx2, ly2, lz2, Shell2)
    func signature: func(lx1,ly1,lz1,A,a, lx2,ly2,lz2,B,b, **kwargs)
    """
    lx1, ly1, lz1, sh1 = bf1
    lx2, ly2, lz2, sh2 = bf2
    N1 = sh1.norms(lx1, ly1, lz1)
    N2 = sh2.norms(lx2, ly2, lz2)
    result = 0.0
    for i, (a, c1, n1) in enumerate(zip(sh1.exponents, sh1.coefficients, N1)):
        for j, (b, c2, n2) in enumerate(zip(sh2.exponents, sh2.coefficients, N2)):
            result += n1 * c1 * n2 * c2 * func(lx1, ly1, lz1, A, a,
                                                 lx2, ly2, lz2, B, b,
                                                 **kwargs)
    return result


# ---------------------------------------------------------------------------
# Public API: build full one-electron integral matrices
# ---------------------------------------------------------------------------

def build_overlap(basis: BasisSet, molecule) -> np.ndarray:
    """
    Build the N×N overlap matrix S_{ij} = <phi_i|phi_j>.
    """
    bfs = basis.basis_functions(molecule)
    coords = np.array([a.coords for a in molecule.atoms])
    n = len(bfs)
    S = np.zeros((n, n))
    for i, (ai, lx1, ly1, lz1, sh1) in enumerate(bfs):
        A = coords[ai]
        for j, (aj, lx2, ly2, lz2, sh2) in enumerate(bfs):
            if j < i:
                S[i, j] = S[j, i]
                continue
            B = coords[aj]
            S[i, j] = _contracted_integral(
                overlap_primitive,
                (lx1, ly1, lz1, sh1), A,
                (lx2, ly2, lz2, sh2), B,
            )
    return S


def build_kinetic(basis: BasisSet, molecule) -> np.ndarray:
    """
    Build the N×N kinetic energy matrix T_{ij} = <phi_i|-½∇²|phi_j>.
    """
    bfs = basis.basis_functions(molecule)
    coords = np.array([a.coords for a in molecule.atoms])
    n = len(bfs)
    T = np.zeros((n, n))
    for i, (ai, lx1, ly1, lz1, sh1) in enumerate(bfs):
        A = coords[ai]
        for j, (aj, lx2, ly2, lz2, sh2) in enumerate(bfs):
            if j < i:
                T[i, j] = T[j, i]
                continue
            B = coords[aj]
            T[i, j] = _contracted_integral(
                kinetic_primitive,
                (lx1, ly1, lz1, sh1), A,
                (lx2, ly2, lz2, sh2), B,
            )
    return T


def build_nuclear(basis: BasisSet, molecule) -> np.ndarray:
    """
    Build the N×N nuclear attraction matrix V_{ij} = <phi_i|Σ_C -Z_C/|r-C||phi_j>.
    """
    bfs = basis.basis_functions(molecule)
    coords = np.array([a.coords for a in molecule.atoms])
    n = len(bfs)
    V = np.zeros((n, n))
    for i, (ai, lx1, ly1, lz1, sh1) in enumerate(bfs):
        A = coords[ai]
        for j, (aj, lx2, ly2, lz2, sh2) in enumerate(bfs):
            if j < i:
                V[i, j] = V[j, i]
                continue
            B = coords[aj]
            v_ij = 0.0
            for k, atom in enumerate(molecule.atoms):
                C = coords[k]
                Z = float(atom.Z)
                v_ij += _contracted_integral(
                    nuclear_primitive,
                    (lx1, ly1, lz1, sh1), A,
                    (lx2, ly2, lz2, sh2), B,
                    C=C, Z=Z,
                )
            V[i, j] = v_ij
    return V


def build_core_hamiltonian(basis: BasisSet, molecule) -> np.ndarray:
    """H_core = T + V  (one-electron Hamiltonian)."""
    return build_kinetic(basis, molecule) + build_nuclear(basis, molecule)
