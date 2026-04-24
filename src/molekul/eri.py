"""
Two-electron electron repulsion integrals (ERIs).

    (ab|cd) = ∫∫ φ_a(r₁) φ_b(r₁) (1/r₁₂) φ_c(r₂) φ_d(r₂) dr₁ dr₂

Uses the McMurchie–Davidson (MD) scheme.

Reference
---------
Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure Theory" (MEST),
Section 9.9, in particular eq. (9.9.18).

Angular momentum covered
------------------------
l = 0 (s) and l = 1 (p) — sufficient for all STO-3G elements H through Ne.
The formula and code are general for any l; the only limit is the angular
components registered in basis.py (_ANG_COMPONENTS covers s, p, d).

Formula summary
---------------
(ab|cd) = (2π^{5/2}) / (p·q·√(p+q))
         × Σ_{t,u,v}   E^{ax,bx}_t · E^{ay,by}_u · E^{az,bz}_v
         × Σ_{τ,ν,φ}   (−1)^{τ+ν+φ}
                        · E^{cx,dx}_τ · E^{cy,dy}_ν · E^{cz,dz}_φ
                        · R_{t+τ, u+ν, v+φ}(0, α, P−Q)

where
  p = a+b,   q = c+d,   α = p·q/(p+q)
  P = (a·A + b·B)/p,   Q = (c·C + d·D)/q
  AB  = A−B  (for E on bra pair),  CD = C−D (for E on ket pair)
  R_{t,u,v}(n, α, P−Q) — Hermite Coulomb auxiliary integral (same as
  nuclear-attraction but with α instead of p and PQ instead of PC).
"""

from math import pi, sqrt

import numpy as np

from .basis import BasisSet
from .integrals import _E, _R, contracted_norm


# ---------------------------------------------------------------------------
# Primitive ERI
# ---------------------------------------------------------------------------

def eri_primitive(
        lx1: int, ly1: int, lz1: int, A, a: float,
        lx2: int, ly2: int, lz2: int, B, b: float,
        lx3: int, ly3: int, lz3: int, C, c: float,
        lx4: int, ly4: int, lz4: int, D, d: float,
) -> float:
    """
    (g₁g₂|g₃g₄) for four UN-normalized Cartesian Gaussian primitives.

    Parameters
    ----------
    lx1, ly1, lz1 : angular momentum of primitive 1
    A             : centre of primitive 1, shape (3,), in bohr
    a             : exponent of primitive 1
    (and so on for primitives 2, 3, 4)

    Returns
    -------
    float : integral value in atomic units (Hartree·bohr^{-1} collapsed)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    p = a + b
    q = c + d
    alpha = p * q / (p + q)

    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    PQ = P - Q          # vector from Q to P (argument of R)

    # Coordinate separations for E-coefficient recurrences
    ABx = float(A[0] - B[0])
    ABy = float(A[1] - B[1])
    ABz = float(A[2] - B[2])
    CDx = float(C[0] - D[0])
    CDy = float(C[1] - D[1])
    CDz = float(C[2] - D[2])

    # E coefficients for the bra pair (functions 1 and 2)
    Ex1 = [_E(lx1, lx2, t, ABx, a, b) for t in range(lx1 + lx2 + 1)]
    Ey1 = [_E(ly1, ly2, u, ABy, a, b) for u in range(ly1 + ly2 + 1)]
    Ez1 = [_E(lz1, lz2, v, ABz, a, b) for v in range(lz1 + lz2 + 1)]

    # E coefficients for the ket pair (functions 3 and 4)
    Ex2 = [_E(lx3, lx4, t, CDx, c, d) for t in range(lx3 + lx4 + 1)]
    Ey2 = [_E(ly3, ly4, u, CDy, c, d) for u in range(ly3 + ly4 + 1)]
    Ez2 = [_E(lz3, lz4, v, CDz, c, d) for v in range(lz3 + lz4 + 1)]

    total = 0.0
    for t, ex in enumerate(Ex1):
        for u, ey in enumerate(Ey1):
            for v, ez in enumerate(Ez1):
                bra = ex * ey * ez
                if bra == 0.0:
                    continue
                for tau, fx in enumerate(Ex2):
                    for nu, fy in enumerate(Ey2):
                        for phi, fz in enumerate(Ez2):
                            sign = (-1) ** (tau + nu + phi)
                            total += (bra * sign * fx * fy * fz
                                      * _R(t + tau, u + nu, v + phi, 0, alpha, PQ))

    return 2.0 * pi**2.5 / (p * q * sqrt(p + q)) * total


# ---------------------------------------------------------------------------
# Contracted ERI
# ---------------------------------------------------------------------------

def _contracted_eri(bf1, A, bf2, B, bf3, C, bf4, D) -> float:
    """
    Contracted (ab|cd) for four basis functions.

    Parameters
    ----------
    bf1 = (lx, ly, lz, Shell), A = centre  [and similarly for bf2, bf3, bf4]
    """
    lx1, ly1, lz1, sh1 = bf1
    lx2, ly2, lz2, sh2 = bf2
    lx3, ly3, lz3, sh3 = bf3
    lx4, ly4, lz4, sh4 = bf4

    N1 = sh1.norms(lx1, ly1, lz1)
    N2 = sh2.norms(lx2, ly2, lz2)
    N3 = sh3.norms(lx3, ly3, lz3)
    N4 = sh4.norms(lx4, ly4, lz4)
    cn = (contracted_norm(lx1, ly1, lz1, sh1) * contracted_norm(lx2, ly2, lz2, sh2)
          * contracted_norm(lx3, ly3, lz3, sh3) * contracted_norm(lx4, ly4, lz4, sh4))

    result = 0.0
    for a_exp, c1, n1 in zip(sh1.exponents, sh1.coefficients, N1):
        for b_exp, c2, n2 in zip(sh2.exponents, sh2.coefficients, N2):
            for c_exp, c3, n3 in zip(sh3.exponents, sh3.coefficients, N3):
                for d_exp, c4, n4 in zip(sh4.exponents, sh4.coefficients, N4):
                    result += (n1 * c1 * n2 * c2 * n3 * c3 * n4 * c4
                               * eri_primitive(
                                   lx1, ly1, lz1, A, a_exp,
                                   lx2, ly2, lz2, B, b_exp,
                                   lx3, ly3, lz3, C, c_exp,
                                   lx4, ly4, lz4, D, d_exp,
                               ))
    return result * cn


# ---------------------------------------------------------------------------
# Full ERI tensor with 8-fold permutation symmetry
# ---------------------------------------------------------------------------

def build_eri(basis: BasisSet, molecule) -> np.ndarray:
    """
    Build the N×N×N×N ERI tensor (ab|cd) in chemists' notation.

    Exploits the 8-fold permutation symmetry:
        (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc)
                = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)

    Unique elements (those with compound index ij ≥ kl, where
    ij = i(i+1)/2+j for j≤i) are computed once and broadcast
    to all eight permuted positions.

    Parameters
    ----------
    basis    : BasisSet object (e.g. STO3G)
    molecule : Molecule object

    Returns
    -------
    np.ndarray, shape (n, n, n, n), all ERIs in atomic units.
    """
    bfs = basis.basis_functions(molecule)
    coords = np.array([atom.coords for atom in molecule.atoms])
    n = len(bfs)
    eri = np.zeros((n, n, n, n))

    for i in range(n):
        ai, lx1, ly1, lz1, sh1 = bfs[i]
        A = coords[ai]
        for j in range(i + 1):           # j ≤ i
            aj, lx2, ly2, lz2, sh2 = bfs[j]
            B = coords[aj]
            ij = i * (i + 1) // 2 + j
            for k in range(n):
                ak, lx3, ly3, lz3, sh3 = bfs[k]
                C = coords[ak]
                for l in range(k + 1):   # l ≤ k
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:           # unique pair condition
                        continue
                    al, lx4, ly4, lz4, sh4 = bfs[l]
                    D = coords[al]

                    val = _contracted_eri(
                        (lx1, ly1, lz1, sh1), A,
                        (lx2, ly2, lz2, sh2), B,
                        (lx3, ly3, lz3, sh3), C,
                        (lx4, ly4, lz4, sh4), D,
                    )

                    # Broadcast 8-fold symmetry
                    eri[i, j, k, l] = val
                    eri[j, i, k, l] = val
                    eri[i, j, l, k] = val
                    eri[j, i, l, k] = val
                    eri[k, l, i, j] = val
                    eri[l, k, i, j] = val
                    eri[k, l, j, i] = val
                    eri[l, k, j, i] = val

    return eri
