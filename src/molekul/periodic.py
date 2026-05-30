"""Periodic crystal utilities and one-electron Bloch-sum integrals."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import ceil, pi
from typing import Sequence

import numpy as np

from .atoms import Atom
from .backend import get_xp, to_cpu, to_device
from .basis import BasisSet
from .eri import _contracted_eri
from .integrals import (
    _contracted_integral,
    kinetic_primitive,
    nuclear_primitive,
    overlap_primitive,
)


@dataclass
class Crystal:
    """Periodic unit cell with Cartesian lattice and atom coordinates in Bohr."""

    lattice: np.ndarray
    atoms: list[Atom] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        self.lattice = np.asarray(self.lattice, dtype=float)
        if self.lattice.ndim != 2 or self.lattice.shape[1] != 3:
            raise ValueError("lattice must have shape (ndim, 3)")
        if self.lattice.shape[0] not in (1, 2, 3):
            raise ValueError("periodic dimension must be 1, 2, or 3")
        if np.linalg.matrix_rank(self.lattice) != self.lattice.shape[0]:
            raise ValueError("lattice vectors must be linearly independent")
        n_electrons = self.n_electrons
        unpaired = self.multiplicity - 1
        if (n_electrons - unpaired) % 2 != 0:
            raise ValueError(
                f"Charge {self.charge} and multiplicity {self.multiplicity} "
                f"are inconsistent with {n_electrons} electrons."
            )

    @property
    def ndim(self) -> int:
        return self.lattice.shape[0]

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_electrons(self) -> int:
        return sum(atom.Z for atom in self.atoms) - self.charge

    @property
    def n_alpha(self) -> int:
        unpaired = self.multiplicity - 1
        return (self.n_electrons + unpaired) // 2

    @property
    def n_beta(self) -> int:
        return self.n_electrons - self.n_alpha

    @property
    def coords_bohr(self) -> np.ndarray:
        return np.stack([atom.coords for atom in self.atoms])

    @property
    def reciprocal_lattice(self) -> np.ndarray:
        """Reciprocal vectors, shape (ndim, 3), Cartesian Bohr^-1."""
        if self.ndim == 3:
            a1, a2, a3 = self.lattice
            volume = float(np.dot(a1, np.cross(a2, a3)))
            if abs(volume) < 1e-14:
                raise ValueError("3D lattice volume is zero")
            return np.array([
                2.0 * pi * np.cross(a2, a3) / volume,
                2.0 * pi * np.cross(a3, a1) / volume,
                2.0 * pi * np.cross(a1, a2) / volume,
            ])
        gram = self.lattice @ self.lattice.T
        return 2.0 * pi * np.linalg.solve(gram, self.lattice)

    def lattice_vectors_in_shell(self, r_max: float) -> np.ndarray:
        """Return all lattice vectors R with |R| <= r_max."""
        if r_max < 0.0:
            raise ValueError("r_max must be non-negative")
        min_len = float(np.min(np.linalg.norm(self.lattice, axis=1)))
        nmax = int(ceil(r_max / min_len)) if min_len > 0.0 else 0
        vectors = []
        for coeffs in product(range(-nmax, nmax + 1), repeat=self.ndim):
            R = np.sum(np.asarray(coeffs)[:, None] * self.lattice, axis=0)
            if np.linalg.norm(R) <= r_max + 1e-12:
                vectors.append(R)
        vectors.sort(key=lambda r: (float(np.linalg.norm(r)), tuple(np.round(r, 12))))
        return np.asarray(vectors, dtype=float)


def monkhorst_pack(lattice: np.ndarray, mesh: tuple[int, ...]) -> np.ndarray:
    """Generate Monkhorst-Pack k-points in Cartesian Bohr^-1."""
    lattice_arr = np.asarray(lattice, dtype=float)
    crystal = Crystal(lattice=lattice_arr, atoms=[])
    if len(mesh) != crystal.ndim:
        raise ValueError("mesh length must match lattice dimensionality")
    if any(n <= 0 for n in mesh):
        raise ValueError("mesh dimensions must be positive")
    reciprocal = crystal.reciprocal_lattice
    kpoints = []
    for idx in product(*(range(n) for n in mesh)):
        frac = np.asarray([i / n for i, n in zip(idx, mesh)], dtype=float)
        kpoints.append(frac @ reciprocal)
    return np.asarray(kpoints, dtype=float)


def _as_kpoints(kpoints: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(kpoints, dtype=float)
    if arr.shape == (3,):
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("kpoints must have shape (n_kpts, 3) or (3,)")
    return arr


def _basis_data(crystal: Crystal, basis: BasisSet):
    bfs = basis.basis_functions(crystal)
    coords = crystal.coords_bohr
    return bfs, coords, len(bfs)


def _overlap_translation(crystal: Crystal, basis: BasisSet, R: np.ndarray) -> np.ndarray:
    bfs, coords, n_basis = _basis_data(crystal, basis)
    S = np.zeros((n_basis, n_basis), dtype=float)
    for i, (ai, lx1, ly1, lz1, sh1) in enumerate(bfs):
        A = coords[ai]
        bf1 = (lx1, ly1, lz1, sh1)
        for j, (aj, lx2, ly2, lz2, sh2) in enumerate(bfs):
            B = coords[aj] + R
            bf2 = (lx2, ly2, lz2, sh2)
            S[i, j] = _contracted_integral(overlap_primitive, bf1, A, bf2, B)
    return S


def _hcore_translation(crystal: Crystal, basis: BasisSet, R: np.ndarray) -> np.ndarray:
    bfs, coords, n_basis = _basis_data(crystal, basis)
    H = np.zeros((n_basis, n_basis), dtype=float)
    nuclear_centers = [(atom.coords + R, float(atom.Z)) for atom in crystal.atoms]
    for i, (ai, lx1, ly1, lz1, sh1) in enumerate(bfs):
        A = coords[ai]
        bf1 = (lx1, ly1, lz1, sh1)
        for j, (aj, lx2, ly2, lz2, sh2) in enumerate(bfs):
            B = coords[aj] + R
            bf2 = (lx2, ly2, lz2, sh2)
            val = _contracted_integral(kinetic_primitive, bf1, A, bf2, B)
            for C, Z in nuclear_centers:
                val += _contracted_integral(
                    nuclear_primitive, bf1, A, bf2, B, C=C, Z=Z
                )
            H[i, j] = val
    return H


def _bloch_sum(crystal: Crystal, basis_fn: BasisSet, kpoints, r_max_factor: float, builder) -> np.ndarray:
    if r_max_factor < 0.0:
        raise ValueError("r_max_factor must be non-negative")
    xp = get_xp()
    kpts = _as_kpoints(kpoints)
    n_basis = basis_fn.n_basis(crystal)
    max_lattice_len = float(np.max(np.linalg.norm(crystal.lattice, axis=1)))
    r_max = r_max_factor * max_lattice_len
    R_vectors = crystal.lattice_vectors_in_shell(r_max)
    result = xp.zeros((len(kpts), n_basis, n_basis), dtype=xp.complex128)
    k_dev = to_device(kpts)
    for R in R_vectors:
        block = to_device(builder(crystal, basis_fn, R).astype(np.complex128))
        R_dev = to_device(R)
        phases = xp.exp(1j * (k_dev @ R_dev))
        result += phases[:, None, None] * block[None, :, :]
    return to_cpu(result).astype(np.complex128)


def bloch_overlap(crystal: Crystal, basis_fn: BasisSet, kpoints, r_max_factor: float = 4.0) -> np.ndarray:
    """Build S(k) for each k-point, shape (n_kpts, n_basis, n_basis)."""
    return _bloch_sum(crystal, basis_fn, kpoints, r_max_factor, _overlap_translation)


def bloch_hcore(crystal: Crystal, basis_fn: BasisSet, kpoints, r_max_factor: float = 4.0) -> np.ndarray:
    """Build approximate one-electron H_core(k), shape (n_kpts, n_basis, n_basis)."""
    return _bloch_sum(crystal, basis_fn, kpoints, r_max_factor, _hcore_translation)

@dataclass
class PeriodicHFResult:
    """Result container for cutoff periodic Hartree-Fock."""

    energy_per_cell: float
    band_energies: np.ndarray
    kpoints: np.ndarray
    mo_coefficients_k: list[np.ndarray]
    density_matrix: np.ndarray
    converged: bool
    n_iter: int
    n_occ: int
    n_basis: int


def _block_phase_sum(blocks: dict[tuple[float, float, float], np.ndarray], kpoints: np.ndarray) -> np.ndarray:
    n_kpts = len(kpoints)
    first = next(iter(blocks.values()))
    out = np.zeros((n_kpts,) + first.shape, dtype=np.complex128)
    for key, block in blocks.items():
        R = np.asarray(key, dtype=float)
        phases = np.exp(1j * (kpoints @ R))
        out += phases.reshape((n_kpts,) + (1,) * block.ndim) * block[None, ...]
    return out


def _block_key(R: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(R, dtype=float), 12))


def _eri_periodic_block(crystal: Crystal, basis: BasisSet, R: np.ndarray, Rp: np.ndarray) -> np.ndarray:
    """ERI block (mu nu_R | lambda sigma_Rp)."""
    bfs, coords, n_basis = _basis_data(crystal, basis)
    out = np.zeros((n_basis, n_basis, n_basis, n_basis), dtype=float)
    for mu, (amu, lx1, ly1, lz1, sh1) in enumerate(bfs):
        A = coords[amu]
        bf1 = (lx1, ly1, lz1, sh1)
        for nu, (anu, lx2, ly2, lz2, sh2) in enumerate(bfs):
            B = coords[anu] + R
            bf2 = (lx2, ly2, lz2, sh2)
            for lam, (alam, lx3, ly3, lz3, sh3) in enumerate(bfs):
                C = coords[alam]
                bf3 = (lx3, ly3, lz3, sh3)
                for sig, (asig, lx4, ly4, lz4, sh4) in enumerate(bfs):
                    D = coords[asig] + Rp
                    bf4 = (lx4, ly4, lz4, sh4)
                    out[mu, nu, lam, sig] = _contracted_eri(bf1, A, bf2, B, bf3, C, bf4, D)
    return out


def _periodic_nuclear_repulsion_cutoff(crystal: Crystal, R_vectors: np.ndarray) -> float:
    """Nuclear repulsion per cell using the same finite real-space shell."""
    e_nn = 0.0
    for ia, atom_a in enumerate(crystal.atoms):
        for ib, atom_b in enumerate(crystal.atoms):
            for R in R_vectors:
                if np.allclose(R, 0.0) and ib <= ia:
                    continue
                distance = np.linalg.norm(atom_a.coords - (atom_b.coords + R))
                if distance > 1e-12:
                    e_nn += 0.5 * atom_a.Z * atom_b.Z / distance
    return float(e_nn)


def _generalized_eigh(F: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve F C = S C eps with Hermitian complex matrices."""
    s_vals, s_vecs = np.linalg.eigh(S)
    s_vals = np.where(s_vals > 1e-10, s_vals, 1e-10)
    X = s_vecs @ np.diag(s_vals ** -0.5) @ s_vecs.conj().T
    Fp = X.conj().T @ F @ X
    eps, Cp = np.linalg.eigh(Fp)
    C = X @ Cp
    return eps.real, C


def _occupations(band_energies: np.ndarray, n_electrons: int) -> np.ndarray:
    """Spin-restricted Aufbau occupations over all k/band states."""
    n_kpts, n_basis = band_energies.shape
    remaining = float(n_electrons * n_kpts)
    occ = np.zeros_like(band_energies, dtype=float)
    states = sorted((band_energies[k, b], k, b) for k in range(n_kpts) for b in range(n_basis))
    for _, k, b in states:
        if remaining <= 0.0:
            break
        fill = min(2.0, remaining)
        occ[k, b] = fill
        remaining -= fill
    return occ


def periodic_hf(
    crystal: Crystal,
    basis_fn: BasisSet,
    kpoints: np.ndarray,
    *,
    r_max_factor: float = 4.0,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    verbose: bool = False,
) -> PeriodicHFResult:
    """
    Run cutoff periodic Hartree-Fock for small 1D cells.

    Phase 20b uses the accepted Phase 20a per-cell nuclear-attraction convention:
    H_core(R) contains nuclei translated with the ket cell R. This is an explicit
    finite-cutoff approximation for the 1D H chain and is superseded by Ewald in
    Phase 20c.
    """
    if crystal.ndim != 1:
        raise NotImplementedError("Phase 20b periodic_hf supports 1D crystals only")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if conv_tol <= 0.0:
        raise ValueError("conv_tol must be positive")

    kpts = _as_kpoints(kpoints)
    n_kpts = len(kpts)
    n_basis = basis_fn.n_basis(crystal)
    max_lattice_len = float(np.max(np.linalg.norm(crystal.lattice, axis=1)))
    R_vectors = crystal.lattice_vectors_in_shell(r_max_factor * max_lattice_len)
    keys = [_block_key(R) for R in R_vectors]

    S_blocks = {key: _overlap_translation(crystal, basis_fn, R) for key, R in zip(keys, R_vectors)}
    H_blocks = {key: _hcore_translation(crystal, basis_fn, R) for key, R in zip(keys, R_vectors)}
    eri_blocks = {
        (key_r, key_rp): _eri_periodic_block(crystal, basis_fn, R, Rp)
        for key_r, R in zip(keys, R_vectors)
        for key_rp, Rp in zip(keys, R_vectors)
    }

    S_k = _block_phase_sum(S_blocks, kpts)
    H_k = _block_phase_sum(H_blocks, kpts)
    S0 = S_blocks[_block_key(np.zeros(3))]
    P = np.zeros((n_basis, n_basis), dtype=float)
    converged = False
    band_energies = np.zeros((n_kpts, n_basis), dtype=float)
    coeffs_k: list[np.ndarray] = [np.eye(n_basis, dtype=np.complex128) for _ in range(n_kpts)]

    for iteration in range(1, max_iter + 1):
        F_blocks: dict[tuple[float, float, float], np.ndarray] = {}
        for key_r in keys:
            G = np.zeros((n_basis, n_basis), dtype=float)
            for key_rp in keys:
                eri = eri_blocks[(key_r, key_rp)]
                coul = np.einsum("ls,mnls->mn", P, eri)
                exch = np.einsum("ls,mlns->mn", P, eri)
                G += coul - 0.5 * exch
            F_blocks[key_r] = H_blocks[key_r] + G

        F_k = _block_phase_sum(F_blocks, kpts)
        for ik in range(n_kpts):
            eps, C = _generalized_eigh(F_k[ik], S_k[ik])
            band_energies[ik] = eps
            coeffs_k[ik] = C

        occ = _occupations(band_energies, crystal.n_electrons)
        P_new = np.zeros((n_basis, n_basis), dtype=np.complex128)
        for ik, C in enumerate(coeffs_k):
            for band in range(n_basis):
                if occ[ik, band] > 0.0:
                    vec = C[:, band]
                    P_new += occ[ik, band] * np.outer(vec, vec.conj()) / n_kpts
        P_new = P_new.real
        trace = float(np.einsum("mn,nm->", P_new, S0).real)
        if abs(trace) > 1e-14:
            P_new *= crystal.n_electrons / trace

        dP = float(np.max(np.abs(P_new - P)))
        P = P_new
        if verbose:
            print(f"Periodic HF iter {iteration:3d}: dP={dP:.6e}")
        if dP < conv_tol:
            converged = True
            break
    else:
        iteration = max_iter

    # Rebuild final Fock blocks from converged density.
    F_blocks = {}
    for key_r in keys:
        G = np.zeros((n_basis, n_basis), dtype=float)
        for key_rp in keys:
            eri = eri_blocks[(key_r, key_rp)]
            G += np.einsum("ls,mnls->mn", P, eri) - 0.5 * np.einsum("ls,mlns->mn", P, eri)
        F_blocks[key_r] = H_blocks[key_r] + G
    F_k = _block_phase_sum(F_blocks, kpts)
    for ik in range(n_kpts):
        eps, C = _generalized_eigh(F_k[ik], S_k[ik])
        band_energies[ik] = eps
        coeffs_k[ik] = C

    e_elec = 0.0
    for ik in range(n_kpts):
        e_elec += np.einsum("mn,nm->", P, (H_k[ik] + 0.5 * F_k[ik])).real
    e_elec /= n_kpts
    e_nn = _periodic_nuclear_repulsion_cutoff(crystal, R_vectors)
    energy = float(e_elec + e_nn)

    return PeriodicHFResult(
        energy_per_cell=energy,
        band_energies=band_energies,
        kpoints=kpts,
        mo_coefficients_k=coeffs_k,
        density_matrix=P,
        converged=converged,
        n_iter=iteration,
        n_occ=max(1, int(np.ceil(crystal.n_electrons / 2))),
        n_basis=n_basis,
    )
