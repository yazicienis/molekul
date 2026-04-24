"""
Coupled Cluster Singles and Doubles (CCSD).

Theory
------
CCSD is a post-HF correlation method that includes all single and double
excitation amplitudes exactly within the one-particle basis:

  |Ψ_CCSD⟩ = exp(T̂₁ + T̂₂) |Φ₀⟩

This implementation works in the **spin-orbital** basis so the Stanton et al.
(1991) equations apply without modification. For a closed-shell (RHF) molecule
with n spatial MOs, we create 2n spin-orbitals (α spin = even indices,
β spin = odd indices).

Antisymmetrized spin-orbital integrals in physicists' notation:
  <pq||rs> = <pq|rs> − <pq|sr>
  <pq|rs>  = (pr|qs)  (chemists' MO integral)

Correlation energy:
  E_corr = ¼ Σ_{ijab} <ij||ab> τ_{ij}^{ab}  +  Σ_{ia} f_{ia} t_i^a

where τ_{ij}^{ab} = t_{ij}^{ab} + t_i^a t_j^b (in spin-orbital antisymm sense).
For canonical RHF the f_{ia} term vanishes.

Amplitude equations are iterated with DIIS acceleration.
Initial guess: t2_{ij}^{ab}(0) = <ij||ab> / D_{ijab}  (= MP2 amplitudes)
              t1_i^a(0)        = 0

Scaling: O(N⁶) per iteration, O(N⁴) spin-orbital storage.

Public API
----------
ccsd_energy(molecule, basis, rhf_result) → CCSDResult

References
----------
Stanton et al., J. Chem. Phys. 94, 4334 (1991).
Crawford & Schaefer, Rev. Comp. Chem. 14, 33 (2000).
Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure Theory", Ch. 13.
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
class CCSDResult:
    """Output of a CCSD energy calculation."""
    energy_ccsd:  float         # CCSD correlation energy  (Hartree, negative)
    energy_total: float         # E_HF + E_CCSD            (Hartree)
    energy_hf:    float         # RHF total energy         (Hartree)
    energy_mp2:   float         # MP2 correlation energy   (Hartree, from initial T2)
    t1:           np.ndarray    # Spin-orbital T1 amplitudes  (n_occ_so, n_virt_so)
    t2:           np.ndarray    # Spin-orbital T2 amplitudes  (n_occ_so,)*2 + (n_virt_so,)*2
    converged:    bool
    n_iter:       int
    n_occ:        int           # spatial occupied MOs
    n_virt:       int           # spatial virtual MOs
    n_basis:      int


# ---------------------------------------------------------------------------
# Spin-orbital integral construction
# ---------------------------------------------------------------------------

def _build_so_integrals(
        eri_mo: np.ndarray,
        eps:    np.ndarray,
        nocc:   int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build spin-orbital antisymmetrized ERI and Fock arrays from spatial MO
    integrals (RHF).

    Spin-orbital indexing: α spin = 2*p,  β spin = 2*p+1  (p = spatial index).

    Physicists' antisymmetrized integral:
      <pq||rs>  =  <pq|rs> − <pq|sr>
      <pq|rs>   =  δ_{σp,σr} δ_{σq,σs}  (p_spatial r_spatial | q_spatial s_spatial)_chem

    Parameters
    ----------
    eri_mo : (n, n, n, n) spatial MO ERI in chemists' notation (pq|rs)
    eps    : (n,) canonical MO energies
    nocc   : number of occupied spatial MOs

    Returns
    -------
    eri_so   : (2n, 2n, 2n, 2n) antisymmetrized spin-orbital ERI
    eps_so   : (2n,) spin-orbital orbital energies
    fock_so  : (2n, 2n) diagonal Fock matrix in spin-orbital basis
    """
    n    = len(eps)
    n_so = 2 * n

    # Spin-orbital energies: pair each spatial eigenvalue α,β
    eps_so = np.zeros(n_so)
    eps_so[0::2] = eps   # α
    eps_so[1::2] = eps   # β

    # Fock (diagonal for canonical RHF, off-diagonal between spins = 0)
    fock_so = np.diag(eps_so)

    # Build spin-orbital antisymmetrized ERI
    # eri_so[p,q,r,s] = <pq||rs>  (physicists', antisymmetrized)
    # <p_σ q_τ|r_σ' s_τ'> = δ_{σσ'} δ_{ττ'} eri_mo[p_sp, r_sp, q_sp, s_sp]
    # Note: chemists' (pr|qs) = eri_mo[p,r,q,s]

    eri_so = np.zeros((n_so, n_so, n_so, n_so))

    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    # chemists' (pr|qs)
                    prqs = float(eri_mo[p, r, q, s])
                    # chemists' (ps|qr)
                    psqr = float(eri_mo[p, s, q, r])

                    antisym_same = prqs - psqr   # same spin: <pq||rs>_same
                    antisym_αβαβ = prqs           # αβ→αβ exchange
                    antisym_αββα = -psqr          # αβ→βα (swap r,s)

                    pa, pb = 2*p, 2*p+1
                    qa, qb = 2*q, 2*q+1
                    ra, rb = 2*r, 2*r+1
                    sa, sb = 2*s, 2*s+1

                    # αααα
                    eri_so[pa, qa, ra, sa] = antisym_same
                    # ββββ
                    eri_so[pb, qb, rb, sb] = antisym_same
                    # αβαβ
                    eri_so[pa, qb, ra, sb] = antisym_αβαβ
                    # βαβα
                    eri_so[pb, qa, rb, sa] = antisym_αβαβ
                    # αββα
                    eri_so[pa, qb, rb, sa] = antisym_αββα
                    # βααβ
                    eri_so[pb, qa, ra, sb] = antisym_αββα

    return eri_so, eps_so, fock_so


# ---------------------------------------------------------------------------
# AO → MO full 4-index transformation
# ---------------------------------------------------------------------------

def transform_mo_full(eri_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Full AO→MO 4-index transformation.

    (pq|rs)_MO = Σ_{μνλσ} C_{μp} C_{νq} C_{λr} C_{σs} (μν|λσ)_AO

    O(N⁵) cost, O(N⁴) storage.
    """
    tmp = np.einsum("pqrs,pi->iqrs", eri_ao, C, optimize=True)
    tmp = np.einsum("iqrs,qj->ijrs", tmp,    C, optimize=True)
    tmp = np.einsum("ijrs,rk->ijks", tmp,    C, optimize=True)
    return np.einsum("ijks,sl->ijkl", tmp,   C, optimize=True)


# ---------------------------------------------------------------------------
# CCSD energy (spin-orbital)
# ---------------------------------------------------------------------------

def _ccsd_energy_so(
        t1:     np.ndarray,
        t2:     np.ndarray,
        fov:    np.ndarray,
        oovv:   np.ndarray,
) -> float:
    """
    CCSD correlation energy in spin-orbital space (Stanton Eq. 9):

      E_corr = Σ_{ia} f_{ia} t_i^a
             + ¼ Σ_{ijab} <ij||ab> (t_{ij}^{ab} + t_i^a t_j^b − t_j^a t_i^b)

    oovv[i,j,a,b] = <ij||ab>  (already antisymmetrized).
    For canonical RHF, f_{ia} = 0.
    """
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ja,ib->ijab", t1, t1)
    e_singles = float(np.einsum("ia,ia->", fov, t1))
    e_doubles  = 0.25 * float(np.einsum("ijab,ijab->", oovv, tau))
    return e_singles + e_doubles


# ---------------------------------------------------------------------------
# Stanton intermediates (spin-orbital, Eqs. 3–8)
# ---------------------------------------------------------------------------

def _make_intermediates_so(
        t1:    np.ndarray,
        t2:    np.ndarray,
        fock:  np.ndarray,
        eri:   np.ndarray,
        nocc:  int,
) -> tuple:
    """
    Build Stanton effective Fock and W intermediates in spin-orbital space.

    Indices: i,j,k,l ∈ [0, nocc),  a,b,c,d ∈ [nocc, n_so).
    eri[p,q,r,s] = <pq||rs>  (antisymmetrized, physicists').

    Returns (Fvv, Fmi, Fme, Woooo, Wvvvv, Wovvo) as described in Stanton (1991).
    """
    o = slice(0, nocc)
    v = slice(nocc, None)

    fvv  = fock[v, v]   # virtual-virtual Fock
    foo  = fock[o, o]   # occupied-occupied Fock
    fov  = fock[o, v]   # occupied-virtual (=0 for canonical RHF)

    oovv = eri[o, o, v, v]
    ooov = eri[o, o, o, v]
    ovvv = eri[o, v, v, v]
    oooo = eri[o, o, o, o]
    vvvv = eri[v, v, v, v]
    ovvo = eri[o, v, v, o]

    # τ and τ̃  (Stanton Eqs. before 3)
    tau       = t2 + np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ja,ib->ijab", t1, t1)
    tau_tilde = t2 + 0.5 * (np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ja,ib->ijab", t1, t1))

    # ------------------------------------------------------------------
    # Fvv[a,e]  (Stanton Eq. 3)
    #
    # F_ae = (1−δ_ae) f_ae − ½ Σm f_me t_m^a
    #      + Σmf t_m^f <ma||fe>
    #      − ½ Σmnf τ̃_{mn}^{af} <mn||ef>
    #
    # <ma||fe> = eri[o,v,v,v][m,a,f,e]? No:
    #   eri[o,v,v,v]: indices (m, a_virt, f_virt, e_virt)
    #   ovvv[m,a,f,e] = <ma||fe>  ✓ (m occ, a,f,e virt, physicists' antisymm)
    # Wait: ovvv = eri[o,v,v,v], so ovvv[m,a,f,e] = <ma||fe>?
    # In physicists' notation <pq||rs> with p=m(occ), q=a(virt), r=f(virt), s=e(virt)
    # eri[m, a+nocc, f+nocc, e+nocc] = <m,a||f,e> ✓
    # ------------------------------------------------------------------
    Fvv = fvv.copy()
    Fvv[np.diag_indices_from(Fvv)] = 0.0   # (1−δ_ae): zero diagonal
    Fvv -= 0.5 * np.einsum("me,ma->ae", fov, t1)
    Fvv += np.einsum("mf,mafe->ae", t1, ovvv)
    Fvv -= 0.5 * np.einsum("mnaf,mnef->ae", tau_tilde, oovv)

    # ------------------------------------------------------------------
    # Fmi[m,i]  (Stanton Eq. 4)
    #
    # F_mi = (1−δ_mi) f_mi + ½ Σe f_me t_i^e
    #      + Σne t_n^e <mn||ie>
    #      + ½ Σnef τ̃_{in}^{ef} <mn||ef>
    #
    # ooov[m,n,i,e] = <mn||ie>  (m,n,i occ, e virt) ✓
    # ------------------------------------------------------------------
    Fmi = foo.copy()
    Fmi[np.diag_indices_from(Fmi)] = 0.0   # (1−δ_mi)
    Fmi += 0.5 * np.einsum("me,ie->mi", fov, t1)
    Fmi += np.einsum("ne,mnie->mi", t1, ooov)
    Fmi += 0.5 * np.einsum("inef,mnef->mi", tau_tilde, oovv)

    # ------------------------------------------------------------------
    # Fme[m,e]  (Stanton Eq. 5)
    #
    # F_me = f_me + Σnf t_n^f <mn||ef>
    # ------------------------------------------------------------------
    Fme = fov.copy()
    Fme += np.einsum("nf,mnef->me", t1, oovv)

    # ------------------------------------------------------------------
    # Woooo[m,n,i,j]  (Stanton Eq. 6)
    #
    # W_{mnij} = <mn||ij> + P̂(ij) Σe t_j^e <mn||ie>
    #          + ¼ Σef τ_{ij}^{ef} <mn||ef>
    # ------------------------------------------------------------------
    Woooo = oooo.copy()
    Woooo += np.einsum("je,mnie->mnij", t1, ooov)
    Woooo -= np.einsum("ie,mnje->mnij", t1, ooov)   # P̂(ij) antisymm
    Woooo += 0.25 * np.einsum("ijef,mnef->mnij", tau, oovv)

    # ------------------------------------------------------------------
    # Wvvvv[a,b,e,f]  (Stanton Eq. 7)
    #
    # W_{abef} = <ab||ef> − P̂(ab) Σm t_m^b <am||ef>
    #          + ¼ Σmn τ_{mn}^{ab} <mn||ef>
    #
    # <am||ef> = ovvv[m,a,e,f] with reorder? Let's be careful:
    # <am||ef>: a virt, m occ, e virt, f virt
    # eri[a+nocc, m, e+nocc, f+nocc] — but our slice is eri[o,v,v,v] with occ first.
    # Need <am||ef> = -<ma||ef>? No: <am||ef> = eri[a_global, m_global, e_global, f_global]
    # = eri[nocc+a, m, nocc+e, nocc+f]
    # This is the (v,o,v,v) block. Let voov or similar.
    # For convenience: <am||ef> = -<ma||ef> = -ovvv[m,a,e,f]  (antisymm <pq||rs>=-<qp||rs>)
    # ovvv[m,a,e,f] = <ma||ef>  → <am||ef> = -ovvv[m,a,e,f] ✓
    # ------------------------------------------------------------------
    Wvvvv = vvvv.copy()
    Wvvvv -= np.einsum("mb,maef->abef", t1, ovvv)
    Wvvvv += np.einsum("ma,mbef->abef", t1, ovvv)   # P̂(ab) antisymm
    Wvvvv += 0.25 * np.einsum("mnab,mnef->abef", tau, oovv)

    # ------------------------------------------------------------------
    # Wovvo[m,b,e,j]  (Stanton Eq. 8)
    #
    # W_{mbej} = <mb||ej> + Σf t_j^f <mb||ef>
    #          − Σn t_n^b <mn||ej>
    #          − ½ Σnf (t_{jn}^{fb} + t_j^f t_n^b) <mn||ef>
    #
    # ovvo[m,b,e,j] = <mb||ej>  (m occ, b virt, e virt, j occ) ✓
    # ovvv[m,b,e,f] = <mb||ef>  ✓
    # ooov[m,n,e,j] = <mn||ej>? No: ooov = eri[o,o,o,v], ooov[m,n,i,e]=<mn||ie>
    # <mn||ej>: e virt, j occ → this is eri[o,o,v,o] block.
    # oovo = eri[o,o,v,o], oovo[m,n,e,j] = <mn||ej> ✓
    # But note: <mn||ej> = -<mn||je> = -ooov[m,n,j,e] (antisymm in last 2 if same type?)
    # No: <mn||ej> has e(virt) and j(occ) in positions 3,4 → different from ooov.
    # So: oovo[m,n,e,j] = eri[o,o,v,o][m,n,e,j] = <mn||ej> ✓
    # ------------------------------------------------------------------
    oovo = eri[o, o, v, o]   # <mn||ej>

    Wovvo = ovvo.copy()
    Wovvo += np.einsum("jf,mbef->mbej", t1, ovvv)
    Wovvo -= np.einsum("nb,mnej->mbej", t1, oovo)
    Wovvo -= 0.5 * np.einsum("jnfb,mnef->mbej", t2, oovv)
    Wovvo -= 0.5 * np.einsum("jf,nb,mnef->mbej", t1, t1, oovv)

    return Fvv, Fmi, Fme, Woooo, Wvvvv, Wovvo


# ---------------------------------------------------------------------------
# T1 residual (Stanton Eq. 1, spin-orbital)
# ---------------------------------------------------------------------------

def _t1_residual_so(
        t1:   np.ndarray,
        t2:   np.ndarray,
        fov:  np.ndarray,
        Fvv:  np.ndarray,
        Fmi:  np.ndarray,
        Fme:  np.ndarray,
        eri:  np.ndarray,
        nocc: int,
) -> np.ndarray:
    """
    T1 residual in spin-orbital space (Stanton Eq. 1).

    R_{ia} = f_{ia} + Σe t_i^e F_ae − Σm t_m^a F_mi
           + Σme t_{im}^{ae} F_me
           + Σme t_m^e <ma||ie>
           − ½ Σmne t_{mn}^{ae} <nm||ei>
           + ½ Σmef t_{im}^{ef} <ma||ef>
    """
    o = slice(0, nocc)
    v = slice(nocc, None)

    ovov = eri[o, v, o, v]   # <ia||jb>
    oovv = eri[o, o, v, v]   # <ij||ab>
    ooov = eri[o, o, o, v]   # <ij||ka>
    ovvv = eri[o, v, v, v]   # <ia||bc>

    R1 = fov.copy()
    R1 += np.einsum("ie,ae->ia", t1, Fvv)
    R1 -= np.einsum("ma,mi->ia", t1, Fmi)
    R1 += np.einsum("imae,me->ia", t2, Fme)

    # Σme t_m^e <ma||ie>: ovov[m,a,i,e] = eri_so[m_occ,a_virt,i_occ,e_virt] = <ma||ie> ✓
    R1 += np.einsum("me,maie->ia", t1, ovov)

    # −½ Σmne t_{mn}^{ae} <nm||ei>
    # <nm||ei> = oovv? No: <nm||ei> = eri[n,m,e,i] but e virt, i occ → <no||vo> block
    # = eri[o,o,v,o][n,m,e,i]  (or use antisymm: <nm||ei> = -<mn||ei> = -ooov? no...)
    # <nm||ei>: n,m occ, e virt, i occ → this is eri[o,o,v,o][n,m,e,i] = oovo[n,m,e,i]
    # Wait: oovo = eri[o,o,v,o], oovo[n,m,e,i] = <nm||ei>? Let's check:
    # eri[o,o,v,o][n,m,e,i] = eri_so[n_global, m_global, e+nocc, i_global] = <nm||ei> ✓
    # But we can also use: <nm||ei> = -<mn||ie> = -ooov[m,n,i,e]
    # (antisymm in first two indices: <nm||ei> = -<mn||ei>, and swap last two: = +<mn||ie>? No)
    # Let me be precise: <pq||rs> antisymm means <pq||rs> = -<qp||rs> = -<pq||sr>
    # So <nm||ei> = -<mn||ei> = -(-<mn||ie>) = <mn||ie> = ooov[m,n,i,e] ✓
    R1 -= 0.5 * np.einsum("mnae,mnie->ia", t2, ooov)

    # +½ Σmef t_{im}^{ef} <ma||ef>
    # <ma||ef>: m occ, a virt, e,f virt → ovvv[m,a,e,f] ✓
    R1 += 0.5 * np.einsum("imef,maef->ia", t2, ovvv)

    return R1


# ---------------------------------------------------------------------------
# T2 residual (Stanton Eq. 2, spin-orbital)
# ---------------------------------------------------------------------------

def _t2_residual_so(
        t1:    np.ndarray,
        t2:    np.ndarray,
        fov:   np.ndarray,
        Fvv:   np.ndarray,
        Fmi:   np.ndarray,
        Fme:   np.ndarray,
        Woooo: np.ndarray,
        Wvvvv: np.ndarray,
        Wovvo: np.ndarray,
        eri:   np.ndarray,
        nocc:  int,
) -> np.ndarray:
    """
    T2 residual in spin-orbital space (Stanton Eq. 2).

    R_{ij}^{ab} = <ij||ab>
                + P̂(ab)[Σe t_{ij}^{ae} F_{be}]
                − P̂(ij)[Σm t_{im}^{ab} F_{mj}]
                + ½ Σmn τ_{mn}^{ab} W_{mnij}
                + ½ Σef τ_{ij}^{ef} W_{abef}
                + P̂(ij)P̂(ab)[Σme t_{im}^{ae} W_{mbej}
                              + Σe t_i^e (aj|be) − Σm t_m^a (mb|ij)]

    P̂(ab)[X_{ab}] = X_{ab} − X_{ba}

    Update: t2_new = t2 − R / D2  (Jacobi step, DIIS-accelerated)
    At convergence R = 0.
    """
    o = slice(0, nocc)
    v = slice(nocc, None)

    oovv = eri[o, o, v, v]   # <ij||ab>
    ooov = eri[o, o, o, v]   # <ij||ka>
    ovvv = eri[o, v, v, v]   # <ia||bc>
    ovov = eri[o, v, o, v]   # <ia||jb>
    ovvo = eri[o, v, v, o]   # <ia||bj>

    R2 = oovv.copy()

    # P̂(ab)[Σe t_{ij}^{ae} F_{be}]:  (Fvv has zero diagonal by (1−δ) construction)
    tmp = np.einsum("ijae,be->ijab", t2, Fvv)
    R2 += tmp - tmp.transpose(0, 1, 3, 2)

    # −P̂(ij)[Σm t_{im}^{ab} F_{mj}]
    tmp = np.einsum("imab,mj->ijab", t2, Fmi)
    R2 -= tmp - tmp.transpose(1, 0, 2, 3)

    # ½ Σmn τ_{mn}^{ab} W_{mnij}
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ja,ib->ijab", t1, t1)
    R2 += 0.5 * np.einsum("mnab,mnij->ijab", tau, Woooo)

    # ½ Σef τ_{ij}^{ef} W_{abef}
    R2 += 0.5 * np.einsum("ijef,abef->ijab", tau, Wvvvv)

    # P̂(ij)P̂(ab)[Σme t_{im}^{ae} W_{mbej}]
    tmp = np.einsum("imae,mbej->ijab", t2, Wovvo)
    R2 += tmp - tmp.transpose(1, 0, 2, 3) - tmp.transpose(0, 1, 3, 2) + tmp.transpose(1, 0, 3, 2)

    # P̂(ij)P̂(ab)[Σe t_i^e <aj||be>]
    # <aj||be>: a virt, j occ, b virt, e virt → eri[v,o,v,v]
    # = -<ja||be> = -ovvv[j,a,b,e]
    # Actually <aj||be> = eri[nocc+a, j, nocc+b, nocc+e]
    # Using antisymm: <aj||be> = -<ja||be> = -ovvv[j,a,b,e]
    tmp = -np.einsum("ie,jabe->ijab", t1, ovvv.transpose(1, 0, 2, 3))
    # ovvv[j,a,b,e] → transpose(1,0,2,3)[j,a,b,e] = ovvv[a,j,b,e]? No.
    # Let me redo: ovvv = eri[o,v,v,v], ovvv[i,a,b,c] = <ia||bc>
    # <ja||be> = ovvv[j,a,b,e] (j occ, a,b,e virt) ✓
    # <aj||be> = -<ja||be> = -ovvv[j,a,b,e]
    tmp = -np.einsum("ie,jabe->ijab", t1, ovvv)
    R2 += tmp - tmp.transpose(1, 0, 2, 3) - tmp.transpose(0, 1, 3, 2) + tmp.transpose(1, 0, 3, 2)

    # P̂(ij)P̂(ab)[−Σm t_m^a <mb||ij>]
    # <mb||ij>: m,i,j occ, b virt → eri[o,v,o,o]
    # Let ovoo = eri[o,v,o,o]: ovoo[m,b,i,j] = <mb||ij>
    ovoo = eri[o, v, o, o]
    tmp = -np.einsum("ma,mbij->ijab", t1, ovoo)
    R2 += tmp - tmp.transpose(1, 0, 2, 3) - tmp.transpose(0, 1, 3, 2) + tmp.transpose(1, 0, 3, 2)

    return R2


# ---------------------------------------------------------------------------
# DIIS for flat amplitude vectors
# ---------------------------------------------------------------------------

class _DIIS:
    """DIIS extrapolator for flat-vector amplitudes."""

    def __init__(self, max_vecs: int = 8):
        self._max    = max_vecs
        self._vecs:   list[np.ndarray] = []
        self._errors: list[np.ndarray] = []

    def push(self, vec: np.ndarray, err: np.ndarray) -> None:
        self._vecs.append(vec.copy())
        self._errors.append(err.copy())
        if len(self._vecs) > self._max:
            self._vecs.pop(0)
            self._errors.pop(0)

    def extrapolate(self) -> np.ndarray:
        m = len(self._vecs)
        if m < 2:
            return self._vecs[-1]
        B = np.zeros((m + 1, m + 1))
        for i in range(m):
            for j in range(i, m):
                v = float(np.dot(self._errors[i], self._errors[j]))
                B[i, j] = B[j, i] = v
        B[m, :] = B[:, m] = -1.0
        B[m, m] = 0.0
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        try:
            c = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return self._vecs[-1]
        result = np.zeros_like(self._vecs[0])
        for ci, vi in zip(c[:m], self._vecs):
            result += ci * vi
        return result


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def ccsd_energy(
        molecule:   Molecule,
        basis:      BasisSet,
        rhf_result: RHFResult,
        *,
        eri_ao:     np.ndarray | None = None,
        max_iter:   int   = 100,
        e_conv:     float = 1e-10,
        amp_conv:   float = 1e-8,
        diis_start: int   = 2,
        diis_size:  int   = 8,
        verbose:    bool  = False,
) -> CCSDResult:
    """
    Compute the CCSD correlation energy from a converged RHF reference.

    Uses spin-orbital amplitudes so Stanton (1991) equations apply verbatim.
    Correlation energy is extracted from the spin-orbital expression and equals
    the standard spatial CCSD energy.

    Parameters
    ----------
    molecule   : Molecule (bohr coordinates)
    basis      : BasisSet
    rhf_result : converged RHFResult from rhf_scf()
    eri_ao     : pre-built AO ERI tensor (built internally if None)
    max_iter   : maximum CCSD iterations
    e_conv     : energy convergence threshold (Hartree)
    amp_conv   : amplitude RMS convergence threshold
    diis_start : iteration at which DIIS begins
    diis_size  : DIIS subspace size
    verbose    : print iteration progress

    Returns
    -------
    CCSDResult
    """
    if not rhf_result.converged:
        raise ValueError("CCSD requires a converged RHF reference.")

    nocc   = molecule.n_electrons // 2
    C      = rhf_result.mo_coefficients
    eps    = rhf_result.mo_energies
    nbasis = C.shape[0]
    nvirt  = nbasis - nocc

    if nvirt < 1:
        raise ValueError("CCSD requires at least one virtual orbital.")

    # Build AO ERI if not supplied
    if eri_ao is None:
        eri_ao = build_eri(basis, molecule)

    if verbose:
        print("  CCSD: AO→MO transformation …")

    eri_mo = transform_mo_full(eri_ao, C)

    if verbose:
        print("  CCSD: building spin-orbital integrals …")

    eri_so, eps_so, fock_so = _build_so_integrals(eri_mo, eps, nocc)

    # Spin-orbital occupied/virtual slices
    nocc_so  = 2 * nocc
    nvirt_so = 2 * nvirt

    o = slice(0, nocc_so)
    v = slice(nocc_so, None)

    eps_occ_so = eps_so[:nocc_so]
    eps_vir_so = eps_so[nocc_so:]
    fov_so     = fock_so[o, v]   # = 0 for canonical RHF
    oovv_so    = eri_so[o, o, v, v]

    # Orbital energy denominators (spin-orbital)
    D1 = eps_occ_so[:, None] - eps_vir_so[None, :]
    D2 = (  eps_occ_so[:, None, None, None]
          + eps_occ_so[None, :, None, None]
          - eps_vir_so[None, None, :, None]
          - eps_vir_so[None, None, None, :])

    # Initial T amplitudes
    t1 = np.zeros((nocc_so, nvirt_so))
    t2 = oovv_so / D2                  # antisymmetric by construction

    e_mp2  = _ccsd_energy_so(t1, t2, fov_so, oovv_so)
    e_ccsd = e_mp2
    converged = False

    diis = _DIIS(diis_size)

    if verbose:
        print(f"  {'Iter':>4}  {'E_corr':>18}  {'ΔE':>12}  {'rms|R|':>12}")
        print(f"  {'----':>4}  {'------':>18}  {'--':>12}  {'------':>12}")
        print(f"  {'MP2':>4}  {e_mp2:18.12f}  {'(initial)':>12}")

    for iteration in range(1, max_iter + 1):
        e_old = e_ccsd

        Fvv, Fmi, Fme, Woooo, Wvvvv, Wovvo = _make_intermediates_so(
            t1, t2, fock_so, eri_so, nocc_so
        )

        R1 = _t1_residual_so(t1, t2, fov_so, Fvv, Fmi, Fme, eri_so, nocc_so)
        R2 = _t2_residual_so(t1, t2, fov_so, Fvv, Fmi, Fme,
                              Woooo, Wvvvv, Wovvo, eri_so, nocc_so)

        # Jacobi update: R1/R2 are numerators N; t_new = N/D
        new_t1 = R1 / D1
        new_t2 = R2 / D2

        err  = np.concatenate([(new_t1 - t1).ravel(), (new_t2 - t2).ravel()])
        rms  = float(np.sqrt(np.mean(err**2)))
        flat = np.concatenate([new_t1.ravel(), new_t2.ravel()])

        if iteration >= diis_start:
            diis.push(flat, err)
            flat_ex = diis.extrapolate()
        else:
            flat_ex = flat

        t1 = flat_ex[:nocc_so * nvirt_so].reshape(nocc_so, nvirt_so)
        t2 = flat_ex[nocc_so * nvirt_so:].reshape(nocc_so, nocc_so, nvirt_so, nvirt_so)

        e_ccsd = _ccsd_energy_so(t1, t2, fov_so, oovv_so)
        de     = e_ccsd - e_old

        if verbose:
            print(f"  {iteration:4d}  {e_ccsd:18.12f}  {de:12.6e}  {rms:12.6e}")

        if abs(de) < e_conv and rms < amp_conv:
            converged = True
            break

    if not converged and verbose:
        print(f"  WARNING: CCSD did not converge in {max_iter} iterations.")

    return CCSDResult(
        energy_ccsd  = e_ccsd,
        energy_total = rhf_result.energy_total + e_ccsd,
        energy_hf    = rhf_result.energy_total,
        energy_mp2   = e_mp2,
        t1           = t1,
        t2           = t2,
        converged    = converged,
        n_iter       = iteration,
        n_occ        = nocc,
        n_virt       = nvirt,
        n_basis      = nbasis,
    )
