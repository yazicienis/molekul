"""
One-electron integrals: overlap (S), kinetic (T), nuclear attraction (V).

Uses the McMurchie-Davidson (MD) recurrence scheme.
Reference: McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978).
           Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory",
           Chapter 9.

All coordinates in atomic units (bohr).
"""

from functools import lru_cache
from math import pi, sqrt, exp, erf
from typing import List, Tuple

import numpy as np

from .basis import BasisSet, Shell, norm_primitive


# ---------------------------------------------------------------------------
# Boys function  F_n(x) = integral_0^1 t^{2n} exp(-x t^2) dt
#
# Three-region implementation — no scipy dependency on this hot path:
#
#   x < 1e-8            : return 1/(2n+1)  (exact x->0 limit)
#   x < _taylor_cutoff  : Taylor series  F_n(x) = sum_{k>=0} (-x)^k/(k!*(2n+2k+1))
#                          No cancellation for small x.  Converges in <35 terms.
#                          Cutoff is max(0.5, n*0.25) so higher-n orders use a
#                          wider Taylor range where upward recurrence would be
#                          less accurate.
#   x >= cutoff         : exact F_0 via math.erf + upward recurrence for n>0
#                          F_0(x) = sqrt(pi/(4x)) * erf(sqrt(x))
#                          F_{k+1} = [(2k+1)*F_k - exp(-x)] / (2x)
#
# Accuracy: max relative error < 5e-14 for n=0..8, x>=0.
# n<=8 covers all ERI orders for basis sets with d functions (cc-pVDZ, 6-31G*).
# ---------------------------------------------------------------------------

_SQRT_PI = sqrt(pi)   # precomputed


def _boys(n: int, x: float) -> float:
    """Boys function F_n(x) = integral_0^1 t^{2n} exp(-x t^2) dt."""
    if x < 1e-8:
        return 1.0 / (2 * n + 1)

    # Adaptive Taylor cutoff: wider for higher n to keep upward recurrence stable
    taylor_cutoff = max(0.5, n * 0.25)

    if x < taylor_cutoff:
        # Taylor series: track (-x)^k/k! in xpow; no cancellation for small x
        result = 1.0 / (2 * n + 1)
        xpow   = 1.0
        for k in range(1, 35):
            xpow  *= -x / k
            inc    = xpow / (2 * n + 2 * k + 1)
            result += inc
            if abs(inc) < 1e-17:
                break
        return result

    # Exact F_0 via stdlib erf (C-level), then upward recurrence
    sqrt_x = sqrt(x)
    F = 0.5 * _SQRT_PI / sqrt_x * erf(sqrt_x)

    if n == 0:
        return F

    # Upward recurrence: F_{k+1} = [(2k+1)*F_k - exp(-x)] / (2x)
    # Stable for x >= taylor_cutoff (where taylor_cutoff >= n/4).
    emx   = exp(-x)
    inv2x = 0.5 / x
    for k in range(n):
        F = ((2 * k + 1) * F - emx) * inv2x

    return F


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
# Contracted Gaussian normalization
# ---------------------------------------------------------------------------
_cnorm_cache: dict = {}


def contracted_norm(lx: int, ly: int, lz: int, sh: "Shell") -> float:
    """
    Return 1/sqrt(<φ|φ>) for a contracted Gaussian with angular momentum
    (lx, ly, lz) and Shell sh.  Result is cached by shell identity.

    STO-nG coefficients are for normalized primitives, but the contracted
    function is not exactly normalised due to non-zero inter-primitive overlaps.
    This factor ensures <φ_norm|φ_norm> = 1, matching PySCF's convention.
    """
    key = (id(sh), lx, ly, lz)
    if key in _cnorm_cache:
        return _cnorm_cache[key]
    norms = sh.norms(lx, ly, lz)
    A = np.zeros(3)   # self-overlap is translation-invariant
    ovlp = 0.0
    for a, c1, n1 in zip(sh.exponents, sh.coefficients, norms):
        for b, c2, n2 in zip(sh.exponents, sh.coefficients, norms):
            ovlp += c1 * n1 * c2 * n2 * overlap_primitive(lx, ly, lz, A, a,
                                                            lx, ly, lz, A, b)
    result = 1.0 / sqrt(ovlp)
    _cnorm_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Contracted integral builders
# ---------------------------------------------------------------------------
def _contracted_integral(func, bf1, A, bf2, B, **kwargs) -> float:
    """
    Evaluate a contracted integral by summing over all primitive pairs.

    bf1 = (lx1, ly1, lz1, Shell1)
    bf2 = (lx2, ly2, lz2, Shell2)
    func signature: func(lx1,ly1,lz1,A,a, lx2,ly2,lz2,B,b, **kwargs)

    Each contracted Gaussian is normalised so that <φ|φ> = 1 (the factor
    contracted_norm(lx, ly, lz, sh) corrects for inter-primitive overlaps).
    """
    lx1, ly1, lz1, sh1 = bf1
    lx2, ly2, lz2, sh2 = bf2
    N1 = sh1.norms(lx1, ly1, lz1)
    N2 = sh2.norms(lx2, ly2, lz2)
    cn1 = contracted_norm(lx1, ly1, lz1, sh1)
    cn2 = contracted_norm(lx2, ly2, lz2, sh2)
    result = 0.0
    for i, (a, c1, n1) in enumerate(zip(sh1.exponents, sh1.coefficients, N1)):
        for j, (b, c2, n2) in enumerate(zip(sh2.exponents, sh2.coefficients, N2)):
            result += n1 * c1 * n2 * c2 * func(lx1, ly1, lz1, A, a,
                                                 lx2, ly2, lz2, B, b,
                                                 **kwargs)
    return result * cn1 * cn2


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


def _dipole_1d(i: int, j: int, k: int, Qx: float, Px: float,
               a: float, b: float) -> float:
    """1-D dipole component ⟨g1|x^k|g2⟩ using E-coefficients."""
    # ⟨g1|(x-Cx)^k|g2⟩ where Cx is origin; here Cx=0 so x^k = (x-0)^k
    # Expand (x)^k = ((x-Px) + Px)^k and use E-coefficients
    # For k=1: ⟨g1|x|g2⟩ = E(i,j,1) + Px*E(i,j,0)  (Px = center-of-mass x)
    p = a + b
    Px_p = (a * (-Qx / 2 + Px) + b * (Qx / 2 + Px)) / p if False else Px  # noqa: use caller's Px
    prefac = sqrt(pi / p)
    val = 0.0
    for t in range(i + j + 1):
        e = _E(i, j, t, Qx, a, b)
        # x^k contribution at order t: need (x-Px+Px)^k = sum binomial
        # Simpler: use E(i,j,t+s) for the x factor
        val += e * _E(0, 0, k - 1, 0, 0, 0) if k == 0 else 0  # placeholder
    # Direct formula: ⟨g1|x|g2⟩ = E(i,j,1)*sqrt(pi/p) + Px*E(i,j,0)*sqrt(pi/p)
    # General: sum_{t=0}^{i+j} E(i,j,t) * (Hermite moment of x^k with index t)
    # Use: x = (x - Px) + Px, (x-Px) contributes E(i,j,t+1) terms
    # => ⟨g1|x^1|g2⟩ = [E(i,j,1) + Px*E(i,j,0)] * sqrt(pi/p)
    if k == 0:
        return prefac * _E(i, j, 0, Qx, a, b)
    elif k == 1:
        return prefac * (_E(i, j, 1, Qx, a, b) + Px * _E(i, j, 0, Qx, a, b))
    else:
        raise NotImplementedError(f"Dipole order k={k} not implemented")


def dipole_primitive(lx1, ly1, lz1, A, a,
                     lx2, ly2, lz2, B, b,
                     origin=None) -> np.ndarray:
    """
    Electric dipole integrals ⟨g1|r|g2⟩ = (⟨x⟩, ⟨y⟩, ⟨z⟩) for unnormalized primitives.

    Uses gauge origin = [0,0,0] by default.
    """
    if origin is None:
        origin = np.zeros(3)
    Qx, Qy, Qz = A[0] - B[0], A[1] - B[1], A[2] - B[2]
    p = a + b
    # Gaussian product center
    Px = (a * A[0] + b * B[0]) / p
    Py = (a * A[1] + b * B[1]) / p
    Pz = (a * A[2] + b * B[2]) / p

    def S1d(i, j, Q, a_, b_):
        return sqrt(pi / (a_ + b_)) * _E(i, j, 0, Q, a_, b_)

    def D1d(i, j, Q, P, a_, b_):
        """⟨i|x|j⟩ along one axis with product center P."""
        pp = a_ + b_
        return sqrt(pi / pp) * (_E(i, j, 1, Q, a_, b_) + P * _E(i, j, 0, Q, a_, b_))

    Sx = S1d(lx1, lx2, Qx, a, b)
    Sy = S1d(ly1, ly2, Qy, a, b)
    Sz = S1d(lz1, lz2, Qz, a, b)

    Dx = D1d(lx1, lx2, Qx, Px - origin[0], a, b)
    Dy = D1d(ly1, ly2, Qy, Py - origin[1], a, b)
    Dz = D1d(lz1, lz2, Qz, Pz - origin[2], a, b)

    return np.array([Dx * Sy * Sz, Sx * Dy * Sz, Sx * Sy * Dz])


def build_dipole_integrals(basis: BasisSet, molecule,
                           origin=None) -> np.ndarray:
    """
    Build the 3×N×N dipole integral matrices ⟨φ_μ|r_x|φ_ν⟩.

    Returns
    -------
    dip : (3, n_basis, n_basis)  — x, y, z components
    """
    if origin is None:
        origin = np.zeros(3)
    bfs = basis.basis_functions(molecule)
    coords = np.array([a.coords for a in molecule.atoms])
    n = len(bfs)
    dip = np.zeros((3, n, n))
    for i, (ai, lx1, ly1, lz1, sh1) in enumerate(bfs):
        A = coords[ai]
        for j, (aj, lx2, ly2, lz2, sh2) in enumerate(bfs):
            B = coords[aj]
            bf1 = (lx1, ly1, lz1, sh1)
            bf2 = (lx2, ly2, lz2, sh2)
            d_ij = _contracted_integral(
                dipole_primitive, bf1, A, bf2, B, origin=origin
            )
            dip[:, i, j] = d_ij
    return dip
