"""
Configuration Interaction Singles (CIS) for excited states.

Theory
------
CIS expresses excited states as a linear combination of singly-excited
determinants on top of the RHF reference:

  |Ψ_K⟩ = Σ_{ia} t_ia^K |Φᵢᵃ⟩

where i runs over occupied MOs and a over virtual MOs.

The CIS Hamiltonian matrix in the (ia) basis is:

  A_{ia,jb} = δᵢⱼ δₐᵦ (εₐ - εᵢ) + ⟨aj||ib⟩

For singlet excited states (using spatial orbitals):
  A_{ia,jb} = δᵢⱼ δₐᵦ (εₐ - εᵢ) + 2(ia|jb) - (ij|ab)

Excitation energies: Δε_K = eigenvalues of A (all positive for stable HF).

Oscillator strengths:
  f_K = (2/3) Δε_K |⟨0|r|K⟩|²

where the transition dipole is:
  ⟨0|r_x|K⟩ = Σ_{ia} t_ia^K ⟨i|x|a⟩   (in MO basis)

References
----------
Foresman et al., J. Phys. Chem. 96, 135 (1992).
Szabo & Ostlund, "Modern Quantum Chemistry", Appendix C.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .rhf import RHFResult
from .integrals import build_dipole_integrals


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CISResult:
    """Container for CIS excited-state output."""
    excitation_energies: np.ndarray   # (n_states,) in Hartree
    excitation_energies_ev: np.ndarray  # (n_states,) in eV
    oscillator_strengths: np.ndarray  # (n_states,) dimensionless
    t_amplitudes: np.ndarray          # (n_states, n_occ, n_virt) CI coefficients
    n_occ: int
    n_virt: int
    n_basis: int
    n_states: int                     # number of states computed


# ---------------------------------------------------------------------------
# MO-basis ERI transformation
# ---------------------------------------------------------------------------

def _ao_to_mo_eri(eri_ao: np.ndarray, C: np.ndarray,
                  n_occ: int, n_virt: int) -> np.ndarray:
    """
    Transform AO ERIs to MO basis for the (ov|ov) and (oo|vv) blocks.

    Returns
    -------
    eri_mo_iajb : (n_occ, n_virt, n_occ, n_virt) — (ia|jb)
    eri_mo_ijab : (n_occ, n_occ, n_virt, n_virt) — (ij|ab)
    """
    C_occ  = C[:, :n_occ]           # (n_basis, n_occ)
    C_virt = C[:, n_occ:n_occ+n_virt]  # (n_basis, n_virt)

    # Half-transform: (μν|λσ) → (iν|λσ) → (ia|λσ) → (ia|jb)
    # Step 1: (μν|λσ) → (pν|λσ) for p in {occ, virt}
    # We do a 4-index transform using einsum in stages for memory efficiency
    n = C.shape[0]

    # (ia|jb) = Σ_{μνλσ} C_μi C_νa (μν|λσ) C_λj C_σb
    # Stage 1: contract μ
    tmp1 = np.einsum("mi,mnls->inls", C_occ, eri_ao)    # (n_occ, n, n, n)
    # Stage 2: contract ν → a
    tmp2 = np.einsum("inls,na->ials", tmp1, C_virt)      # (n_occ, n_virt, n, n)
    # Stage 3: contract λ → j
    tmp3 = np.einsum("ials,lj->iajs", tmp2, C_occ)       # (n_occ, n_virt, n_occ, n)
    # Stage 4: contract σ → b
    iajb = np.einsum("iajs,sb->iajb", tmp3, C_virt)      # (n_occ, n_virt, n_occ, n_virt)

    # (ij|ab) = Σ_{μνλσ} C_μi C_νj (μν|λσ) C_λa C_σb
    # Stage 1: contract μ → i (reuse tmp1)
    # Stage 2: contract ν → j
    tmp2b = np.einsum("inls,nj->ijls", tmp1, C_occ)      # (n_occ, n_occ, n, n)
    # Stage 3: contract λ → a
    tmp3b = np.einsum("ijls,la->ijas", tmp2b, C_virt)    # (n_occ, n_occ, n_virt, n)
    # Stage 4: contract σ → b
    ijab = np.einsum("ijas,sb->ijab", tmp3b, C_virt)     # (n_occ, n_occ, n_virt, n_virt)

    return iajb, ijab


# ---------------------------------------------------------------------------
# CIS matrix construction
# ---------------------------------------------------------------------------

def _build_cis_matrix(eps: np.ndarray, iajb: np.ndarray, ijab: np.ndarray,
                      n_occ: int, n_virt: int) -> np.ndarray:
    """
    Build the CIS Hamiltonian matrix A (singlet).

    A_{ia,jb} = δᵢⱼ δₐᵦ (εₐ - εᵢ) + 2*(ia|jb) - (ij|ab)

    Matrix is (n_occ*n_virt) × (n_occ*n_virt), indexed as [i*n_virt+a, j*n_virt+b].
    """
    nov = n_occ * n_virt
    A = np.zeros((nov, nov))

    # Diagonal: orbital energy differences
    eps_occ  = eps[:n_occ]
    eps_virt = eps[n_occ:n_occ + n_virt]
    for i in range(n_occ):
        for a in range(n_virt):
            ia = i * n_virt + a
            A[ia, ia] = eps_virt[a] - eps_occ[i]

    # Off-diagonal: 2*(ia|jb) - (ij|ab)
    # Reshape to (n_occ*n_virt, n_occ*n_virt)
    two_iajb = 2.0 * iajb.reshape(nov, nov)                   # (ia|jb)
    ijab_r   = ijab.transpose(0, 2, 1, 3).reshape(nov, nov)   # (ij|ab) → ia,jb order

    A += two_iajb - ijab_r
    return A


# ---------------------------------------------------------------------------
# Oscillator strengths
# ---------------------------------------------------------------------------

def _oscillator_strengths(t: np.ndarray, dip_mo: np.ndarray,
                          omega: np.ndarray) -> np.ndarray:
    """
    f_K = (2/3) Δε_K Σ_x |⟨0|x|K⟩|²

    dip_mo : (3, n_occ, n_virt) transition dipole integrals in MO basis
    t      : (n_states, n_occ, n_virt) CI coefficients
    omega  : (n_states,) excitation energies
    """
    # Transition dipole: ⟨0|r_x|K⟩ = Σ_{ia} t_ia^K dip_x[i,a]
    # (n_states, 3)
    td = np.einsum("kia,xia->kx", t, dip_mo)
    td_sq = np.sum(td**2, axis=1)   # (n_states,)
    return (2.0 / 3.0) * omega * td_sq


# ---------------------------------------------------------------------------
# Main CIS driver
# ---------------------------------------------------------------------------

def cis_excitations(molecule, basis, rhf: RHFResult,
                    n_states: int = 5,
                    verbose: bool = True) -> CISResult:
    """
    Compute CIS singlet excitation energies and oscillator strengths.

    Parameters
    ----------
    molecule  : Molecule
    basis     : BasisSet (e.g. STO3G)
    rhf       : RHFResult from rhf_scf()
    n_states  : number of lowest excited states to return
    verbose   : print progress

    Returns
    -------
    CISResult
    """
    from .eri import build_eri
    from .integrals import build_dipole_integrals

    C   = rhf.mo_coefficients
    eps = rhf.mo_energies
    n_basis = C.shape[0]
    n_occ   = molecule.n_alpha
    n_virt  = n_basis - n_occ
    n_states = min(n_states, n_occ * n_virt)

    if verbose:
        print(f"\n  CIS: n_occ={n_occ}, n_virt={n_virt}, dim={n_occ*n_virt}")

    # 1. Build ERIs in MO basis
    if verbose:
        print("  Building MO ERIs …")
    eri_ao = build_eri(basis, molecule)
    iajb, ijab = _ao_to_mo_eri(eri_ao, C, n_occ, n_virt)

    # 2. Build CIS matrix
    if verbose:
        print("  Building CIS matrix …")
    A = _build_cis_matrix(eps, iajb, ijab, n_occ, n_virt)

    # 3. Diagonalise (full diagonalisation — small enough for STO-3G)
    if verbose:
        print("  Diagonalising …")
    omega_all, t_all = np.linalg.eigh(A)

    # Keep only positive excitation energies and first n_states
    pos = omega_all > 0
    omega_all = omega_all[pos]
    t_all     = t_all[:, pos]
    omega = omega_all[:n_states]
    t_mat = t_all[:, :n_states].T          # (n_states, n_occ*n_virt)
    t     = t_mat.reshape(n_states, n_occ, n_virt)

    # 4. Transition dipoles
    dip_ao = build_dipole_integrals(basis, molecule)   # (3, n_basis, n_basis)
    # Transform to MO: ⟨i|r|a⟩ = C[:,i].T @ dip @ C[:,a]
    dip_mo = np.einsum("xmn,mi,na->xia", dip_ao, C[:, :n_occ], C[:, n_occ:n_occ+n_virt])

    f = _oscillator_strengths(t, dip_mo, omega)
    f *= 2.0   # singlet: |⟨0|r|K⟩|² = 2*|Σ_ia t_ia ⟨i|r|a⟩|² (both spin channels)

    HARTREE_TO_EV = 27.211386245988

    if verbose:
        print(f"\n  {'State':>5}  {'ΔE (Ha)':>12}  {'ΔE (eV)':>10}  {'f':>10}")
        print("  " + "-" * 46)
        for k in range(n_states):
            print(f"  {k+1:>5}  {omega[k]:>12.6f}  {omega[k]*HARTREE_TO_EV:>10.4f}  {f[k]:>10.6f}")

    return CISResult(
        excitation_energies=omega,
        excitation_energies_ev=omega * HARTREE_TO_EV,
        oscillator_strengths=f,
        t_amplitudes=t,
        n_occ=n_occ,
        n_virt=n_virt,
        n_basis=n_basis,
        n_states=n_states,
    )
