"""
Møller-Plesset second-order perturbation theory (MP2).

Theory
------
MP2 is the simplest post-HF electron-correlation method.  Starting from a
converged RHF reference, the second-order energy correction is:

    E_MP2 = Σ_{i,j,a,b}  (ia|jb)[2(ia|jb) − (ib|ja)]
            ─────────────────────────────────────────────
                      ε_i + ε_j − ε_a − ε_b

where
  i, j  — occupied MO indices   (0 … n_occ−1)
  a, b  — virtual  MO indices   (n_occ … n_basis−1)
  (ia|jb) — two-electron integral in MO basis (chemists' notation)
  ε_p   — canonical orbital energy from RHF diagonalization

The denominator is always negative for a stable RHF reference, so E_MP2 < 0
(correlation energy lowers the total energy).

AO → MO integral transformation
---------------------------------
Four sequential quarter-transforms (N^5 scaling overall):

    1. g[i,ν,λ,σ] = Σ_μ  C_{μi} · eri[μ,ν,λ,σ]
    2. g[i,a,λ,σ] = Σ_ν  C_{νa} · g[i,ν,λ,σ]
    3. g[i,a,j,σ] = Σ_λ  C_{λj} · g[i,a,λ,σ]
    4. (ia|jb)    = Σ_σ  C_{σb} · g[i,a,j,σ]

Only the occupied×virtual×occupied×virtual block is built, which reduces
memory from n^4 to n_occ² × n_virt² and is sufficient for MP2.

Limitations
-----------
• Closed-shell (RHF) only — no UMP2, no ROHF reference.
• Full 4-index in-core storage.  For n_basis > ~100 consider density-fitting
  or out-of-core algorithms (not implemented here).
• STO-3G MP2 provides only qualitative correlation corrections; larger basis
  sets are needed for quantitative results.
• Geometry optimization at MP2 level requires MP2 gradients (not implemented).

Public API
----------
mp2_energy(molecule, basis, rhf_result) → MP2Result
transform_iajb(eri_ao, C, nocc)         → (n_occ, n_virt, n_occ, n_virt) array

References
----------
Møller & Plesset, Phys. Rev. 46, 618 (1934).
Szabo & Ostlund, "Modern Quantum Chemistry", §6.5.
Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure Theory", Ch. 14.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import BasisSet
from .eri import build_eri
from .molecule import Molecule
from .rhf import RHFResult


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MP2Result:
    """
    Output of an MP2 energy calculation.

    Attributes
    ----------
    energy_mp2    : MP2 correlation energy  E_MP2  (Hartree, negative)
    energy_total  : E_HF + E_MP2           (Hartree)
    energy_hf     : RHF total energy       (Hartree)
    n_occ         : number of occupied MOs
    n_virt        : number of virtual MOs
    n_basis       : total number of basis functions
    """
    energy_mp2:   float
    energy_total: float
    energy_hf:    float
    n_occ:        int
    n_virt:       int
    n_basis:      int


# ---------------------------------------------------------------------------
# AO → MO transformation  (ia|jb) block
# ---------------------------------------------------------------------------

def transform_iajb(
        eri_ao: np.ndarray,
        C: np.ndarray,
        nocc: int,
) -> np.ndarray:
    """
    Transform the AO-basis ERI tensor to the (ia|jb) MO block.

    Parameters
    ----------
    eri_ao : (n, n, n, n) AO two-electron integrals  (μν|λσ)
    C      : (n, n) MO coefficient matrix, columns are MOs
    nocc   : number of occupied MOs

    Returns
    -------
    iajb : (n_occ, n_virt, n_occ, n_virt) float64 array
           iajb[i, a, j, b] = (ia|jb)  where i,j occupied, a,b virtual
    """
    C_occ = C[:, :nocc]          # (n_basis, n_occ)
    C_vir = C[:, nocc:]          # (n_basis, n_virt)

    # Four quarter-transforms — each is an O(N^5) contraction but with the
    # small virtual dimension kept throughout so memory stays O(nocc²×nvirt²).

    # 1.  (μν|λσ) × C_{μi}  →  (i, ν, λ, σ)
    tmp = np.einsum("mi,mnls->inls", C_occ, eri_ao, optimize=True)

    # 2.  (i, ν, λ, σ) × C_{νa}  →  (i, a, λ, σ)
    tmp = np.einsum("na,inls->ials", C_vir, tmp, optimize=True)

    # 3.  (i, a, λ, σ) × C_{λj}  →  (i, a, j, σ)
    tmp = np.einsum("lj,ials->iajs", C_occ, tmp, optimize=True)

    # 4.  (i, a, j, σ) × C_{σb}  →  (i, a, j, b)
    iajb = np.einsum("sb,iajs->iajb", C_vir, tmp, optimize=True)

    return iajb


# ---------------------------------------------------------------------------
# MP2 energy
# ---------------------------------------------------------------------------

def _mp2_correlation(
        iajb: np.ndarray,
        eps_occ: np.ndarray,
        eps_vir: np.ndarray,
) -> float:
    """
    Compute E_MP2 from the (ia|jb) MO integrals and orbital energies.

    E_MP2 = Σ_{i,j,a,b}  (ia|jb)[2(ia|jb) − (ib|ja)]
            ──────────────────────────────────────────────
                      ε_i + ε_j − ε_a − ε_b

    The denominator matrix is built by broadcasting:
        denom[i,a,j,b] = ε_i + ε_j − ε_a − ε_b
    which is always negative for a stable RHF reference.
    """
    nocc  = len(eps_occ)
    nvirt = len(eps_vir)

    # Denominator  shape: (nocc, 1, nocc, 1) + (nocc, 1, 1, 1) − ... → broadcast
    denom = (
          eps_occ[:, None, None, None]   # ε_i
        + eps_occ[None, None, :, None]   # ε_j
        - eps_vir[None, :, None, None]   # −ε_a
        - eps_vir[None, None, None, :]   # −ε_b
    )  # shape: (nocc, nvirt, nocc, nvirt)

    # (ib|ja) = iajb[i, b, j, a]  →  transpose to match iajb[i, a, j, b]
    ibja = iajb.transpose(0, 3, 2, 1)   # iajb[i,a,j,b] → ibja[i,a,j,b] = (ib|ja)

    numerator = iajb * (2.0 * iajb - ibja)

    return float(np.sum(numerator / denom))


def mp2_energy(
        molecule: Molecule,
        basis: BasisSet,
        rhf_result: RHFResult,
        *,
        eri_ao: np.ndarray | None = None,
) -> MP2Result:
    """
    Compute the MP2 correlation energy from a converged RHF reference.

    Parameters
    ----------
    molecule   : Molecule (bohr coordinates)
    basis      : BasisSet (e.g. STO3G)
    rhf_result : converged RHFResult from rhf_scf()
    eri_ao     : pre-built AO ERI tensor (built internally if None)

    Returns
    -------
    MP2Result
    """
    if not rhf_result.converged:
        raise ValueError("MP2 requires a converged RHF reference.")

    nocc = molecule.n_electrons // 2
    C    = rhf_result.mo_coefficients   # (n_basis, n_basis)
    eps  = rhf_result.mo_energies       # (n_basis,)
    nbasis = C.shape[0]
    nvirt  = nbasis - nocc

    if nvirt < 1:
        raise ValueError("MP2 requires at least one virtual orbital.")

    # Build AO ERI tensor if not supplied
    if eri_ao is None:
        eri_ao = build_eri(basis, molecule)

    # Transform to (ia|jb) MO block
    iajb = transform_iajb(eri_ao, C, nocc)

    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]

    e_mp2 = _mp2_correlation(iajb, eps_occ, eps_vir)

    return MP2Result(
        energy_mp2=e_mp2,
        energy_total=rhf_result.energy_total + e_mp2,
        energy_hf=rhf_result.energy_total,
        n_occ=nocc,
        n_virt=nvirt,
        n_basis=nbasis,
    )
