# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Enis Yazici

"""Equation-of-motion CCSD excitation energies.

This module forms the EOM-CCSD-EE effective Hamiltonian by explicitly
projecting ``exp(-T) H exp(T) - E_CCSD`` into the 1h1p + 2h2p determinant
space.  The determinant implementation is intended for the small educational
systems used by MOLEKUL's STO-3G test suite; it preserves the spin-orbital
CCSD convention used in :mod:`molekul.ccsd`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from itertools import combinations
import warnings

import numpy as np

from .basis import BasisSet
from .ccsd import _build_so_integrals, ccsd_energy, transform_mo_full
from .eri import build_eri
from .integrals import build_core_hamiltonian
from .molecule import Molecule
from .rhf import RHFResult


HARTREE_TO_EV = 27.211386245988


@dataclass
class EOMCCSDResult:
    """Output of an EOM-CCSD excitation-energy calculation."""

    excitation_energies: np.ndarray
    excitation_eV: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    n_states: int
    n_occ: int
    n_virt: int


def _annihilate(det: int, orbital: int) -> tuple[int, int] | None:
    if not (det >> orbital) & 1:
        return None
    sign = -1 if ((det & ((1 << orbital) - 1)).bit_count() % 2) else 1
    return det ^ (1 << orbital), sign


def _create(det: int, orbital: int) -> tuple[int, int] | None:
    if (det >> orbital) & 1:
        return None
    sign = -1 if ((det & ((1 << orbital) - 1)).bit_count() % 2) else 1
    return det | (1 << orbital), sign


def _apply_ops(det: int, annihilators: tuple[int, ...],
               creators: tuple[int, ...]) -> tuple[int, int] | None:
    sign = 1
    current = det
    for orbital in annihilators:
        result = _annihilate(current, orbital)
        if result is None:
            return None
        current, s = result
        sign *= s
    for orbital in creators:
        result = _create(current, orbital)
        if result is None:
            return None
        current, s = result
        sign *= s
    return current, sign


def _determinants(n_spin_orb: int, n_elec: int) -> tuple[list[int], dict[int, int]]:
    dets = [sum(1 << p for p in occ) for occ in combinations(range(n_spin_orb), n_elec)]
    return dets, {det: idx for idx, det in enumerate(dets)}


def _excitation_basis(nocc: int, nvirt: int) -> tuple[list[int], list[tuple[str, tuple[int, ...]]]]:
    ref = sum(1 << i for i in range(nocc))
    labels: list[tuple[str, tuple[int, ...]]] = []
    determinants: list[int] = []

    for i in range(nocc):
        for a in range(nvirt):
            det = (ref ^ (1 << i)) | (1 << (nocc + a))
            determinants.append(det)
            labels.append(("single", (i, a)))

    for i, j in combinations(range(nocc), 2):
        for a, b in combinations(range(nvirt), 2):
            det = ref
            det ^= 1 << i
            det ^= 1 << j
            det |= 1 << (nocc + a)
            det |= 1 << (nocc + b)
            determinants.append(det)
            labels.append(("double", (i, j, a, b)))

    return determinants, labels


def _build_hamiltonian(h_so: np.ndarray, eri_so: np.ndarray,
                       dets: list[int], det_index: dict[int, int]) -> np.ndarray:
    n_det = len(dets)
    n_spin_orb = h_so.shape[0]
    hamiltonian = np.zeros((n_det, n_det))

    for col, det in enumerate(dets):
        for q in range(n_spin_orb):
            removed = _annihilate(det, q)
            if removed is None:
                continue
            det_q, sign_q = removed
            for p in range(n_spin_orb):
                created = _create(det_q, p)
                if created is None:
                    continue
                row_det, sign_p = created
                hamiltonian[det_index[row_det], col] += h_so[p, q] * sign_q * sign_p

        for r in range(n_spin_orb):
            removed_r = _annihilate(det, r)
            if removed_r is None:
                continue
            det_r, sign_r = removed_r
            for s in range(n_spin_orb):
                removed_s = _annihilate(det_r, s)
                if removed_s is None:
                    continue
                det_rs, sign_s = removed_s
                for q in range(n_spin_orb):
                    created_q = _create(det_rs, q)
                    if created_q is None:
                        continue
                    det_q, sign_q = created_q
                    for p in range(n_spin_orb):
                        value = eri_so[p, q, r, s]
                        if value == 0.0:
                            continue
                        created_p = _create(det_q, p)
                        if created_p is None:
                            continue
                        row_det, sign_p = created_p
                        sign = sign_r * sign_s * sign_q * sign_p
                        hamiltonian[det_index[row_det], col] += 0.25 * value * sign

    return hamiltonian


def _build_cluster_operator(t1: np.ndarray, t2: np.ndarray, nocc: int,
                            dets: list[int], det_index: dict[int, int]) -> np.ndarray:
    n_det = len(dets)
    nvirt = t1.shape[1]
    cluster = np.zeros((n_det, n_det))

    for col, det in enumerate(dets):
        for i in range(nocc):
            for a in range(nvirt):
                value = t1[i, a]
                if value == 0.0:
                    continue
                applied = _apply_ops(det, (i,), (nocc + a,))
                if applied is None:
                    continue
                row_det, sign = applied
                cluster[det_index[row_det], col] += value * sign

        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvirt):
                    for b in range(nvirt):
                        value = t2[i, j, a, b]
                        if value == 0.0:
                            continue
                        applied = _apply_ops(det, (i, j), (nocc + b, nocc + a))
                        if applied is None:
                            continue
                        row_det, sign = applied
                        cluster[det_index[row_det], col] += 0.25 * value * sign

    return cluster


def _build_spin_squared(n_spatial: int, dets: list[int],
                        det_index: dict[int, int]) -> np.ndarray:
    n_det = len(dets)
    s_plus = np.zeros((n_det, n_det))
    s_minus = np.zeros((n_det, n_det))
    sz = np.zeros(n_det)

    for col, det in enumerate(dets):
        n_alpha = sum((det >> (2 * p)) & 1 for p in range(n_spatial))
        n_beta = sum((det >> (2 * p + 1)) & 1 for p in range(n_spatial))
        sz[col] = 0.5 * (n_alpha - n_beta)

        for p in range(n_spatial):
            beta = 2 * p + 1
            alpha = 2 * p
            raised = _apply_ops(det, (beta,), (alpha,))
            if raised is not None:
                row_det, sign = raised
                s_plus[det_index[row_det], col] += sign
            lowered = _apply_ops(det, (alpha,), (beta,))
            if lowered is not None:
                row_det, sign = lowered
                s_minus[det_index[row_det], col] += sign

    return np.diag(sz ** 2) + 0.5 * (s_plus @ s_minus + s_minus @ s_plus)


def _exp_cluster_action(cluster: np.ndarray, vector: np.ndarray,
                        max_rank: int, sign: float) -> np.ndarray:
    result = vector.copy()
    term = vector.copy()
    for order in range(1, max_rank + 1):
        term = cluster @ term
        result += (sign ** order) * term / factorial(order)
    return result


def _spin_orbital_hcore(h_mo: np.ndarray) -> np.ndarray:
    n_spatial = h_mo.shape[0]
    h_so = np.zeros((2 * n_spatial, 2 * n_spatial))
    h_so[0::2, 0::2] = h_mo
    h_so[1::2, 1::2] = h_mo
    return h_so


def _prepare_integrals(molecule: Molecule, basis: BasisSet,
                       rhf_result: RHFResult) -> tuple[np.ndarray, np.ndarray]:
    eri_ao = build_eri(basis, molecule)
    eri_mo = transform_mo_full(eri_ao, rhf_result.mo_coefficients)
    eri_so, _, _ = _build_so_integrals(
        eri_mo, rhf_result.mo_energies, molecule.n_electrons // 2
    )

    h_ao = build_core_hamiltonian(basis, molecule)
    h_mo = rhf_result.mo_coefficients.T @ h_ao @ rhf_result.mo_coefficients
    return _spin_orbital_hcore(h_mo), eri_so


def _state_amplitudes(eigenvectors: np.ndarray,
                      labels: list[tuple[str, tuple[int, ...]]],
                      n_states: int, nocc: int, nvirt: int) -> tuple[np.ndarray, np.ndarray]:
    r1 = np.zeros((n_states, nocc, nvirt))
    r2 = np.zeros((n_states, nocc, nocc, nvirt, nvirt))
    for state in range(n_states):
        vec = eigenvectors[:, state]
        for idx, (kind, label) in enumerate(labels):
            value = vec[idx].real
            if kind == "single":
                i, a = label
                r1[state, i, a] = value
            else:
                i, j, a, b = label
                r2[state, i, j, a, b] = value
                r2[state, j, i, a, b] = -value
                r2[state, i, j, b, a] = -value
                r2[state, j, i, b, a] = value
    return r1, r2


def eom_ccsd_ee(
        mol: Molecule,
        basis_fn: BasisSet,
        rhf_result: RHFResult,
        n_states: int = 5,
        verbose: bool = False,
) -> EOMCCSDResult:
    """Compute neutral excitation energies with EOM-CCSD-EE."""
    if not rhf_result.converged:
        raise ValueError("EOM-CCSD requires a converged RHF reference.")

    ccsd = ccsd_energy(mol, basis_fn, rhf_result, verbose=verbose)
    if not ccsd.converged:
        raise ValueError("EOM-CCSD requires converged CCSD amplitudes.")

    nocc = 2 * ccsd.n_occ
    nvirt = 2 * ccsd.n_virt
    n_spin_orb = nocc + nvirt
    n_elec = mol.n_electrons

    h_so, eri_so = _prepare_integrals(mol, basis_fn, rhf_result)
    dets, det_index = _determinants(n_spin_orb, n_elec)
    hamiltonian = _build_hamiltonian(h_so, eri_so, dets, det_index)
    cluster = _build_cluster_operator(ccsd.t1, ccsd.t2, nocc, dets, det_index)
    spin_squared = _build_spin_squared(ccsd.n_basis, dets, det_index)

    basis_dets, labels = _excitation_basis(nocc, nvirt)
    basis_indices = [det_index[det] for det in basis_dets]
    dim = len(basis_indices)
    hbar = np.zeros((dim, dim))
    max_rank = min(nocc, nvirt)

    for col, full_idx in enumerate(basis_indices):
        vector = np.zeros(len(dets))
        vector[full_idx] = 1.0
        transformed = _exp_cluster_action(cluster, vector, max_rank, 1.0)
        transformed = hamiltonian @ transformed
        transformed = _exp_cluster_action(cluster, transformed, max_rank, -1.0)
        hbar[:, col] = transformed[basis_indices]

    e_ccsd_electronic = ccsd.energy_total - rhf_result.energy_nuclear
    hbar -= np.eye(dim) * e_ccsd_electronic

    eigenvalues, eigenvectors = np.linalg.eig(hbar)
    max_imag = float(np.max(np.abs(eigenvalues.imag))) if eigenvalues.size else 0.0
    if max_imag > 1e-6:
        warnings.warn(
            f"EOM-CCSD eigenvalues have imaginary components up to {max_imag:.3e}",
            RuntimeWarning,
            stacklevel=2,
        )

    real_mask = np.abs(eigenvalues.imag) <= 1e-6
    positive = np.where(real_mask & (eigenvalues.real > 0.0))[0]
    spin_projected = spin_squared[np.ix_(basis_indices, basis_indices)]
    singlet_indices = []
    for idx in positive:
        vec = eigenvectors[:, idx]
        norm = np.vdot(vec, vec).real
        s2 = np.vdot(vec, spin_projected @ vec).real / norm
        if abs(s2) < 1e-4:
            singlet_indices.append(idx)

    order = np.array(singlet_indices, dtype=int)
    order = order[np.argsort(eigenvalues.real[order])]
    if len(order) < n_states:
        warnings.warn(
            f"EOM-CCSD: requested {n_states} singlet states but only "
            f"{len(order)} found (S²<1e-4 filter). "
            "Try increasing n_states or the diagonalisation space.",
            UserWarning,
            stacklevel=2,
        )
    order = order[:min(n_states, len(order))]
    omega = eigenvalues.real[order]
    selected_vectors = eigenvectors[:, order]
    r1, r2 = _state_amplitudes(selected_vectors, labels, len(order), nocc, nvirt)

    if verbose:
        for idx, energy in enumerate(omega, start=1):
            print(f"  EOM-CCSD state {idx}: {energy:.8f} Ha")

    return EOMCCSDResult(
        excitation_energies=omega,
        excitation_eV=omega * HARTREE_TO_EV,
        r1=r1,
        r2=r2,
        n_states=len(order),
        n_occ=ccsd.n_occ,
        n_virt=ccsd.n_virt,
    )
