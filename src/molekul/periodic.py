# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Enis Yazici

"""Periodic crystal utilities and one-electron Bloch-sum integrals."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import ceil, erfc, exp, pi, sqrt
from typing import Sequence

import numpy as np

from .atoms import Atom
from .backend import get_xp, to_cpu, to_device
from .basis import BasisSet
from .constants import ATOMIC_MASS
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


def _ewald_eta(crystal: Crystal, eta: float | None) -> float:
    if crystal.ndim != 3:
        raise ValueError("Ewald summation requires a 3D crystal")
    if eta is None:
        min_len = float(np.min(np.linalg.norm(crystal.lattice, axis=1)))
        eta = sqrt(pi) / min_len
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    return float(eta)


def _ewald_real_vectors(crystal: Crystal, eta: float) -> np.ndarray:
    return crystal.lattice_vectors_in_shell(4.0 / eta)


def _ewald_reciprocal_vectors(crystal: Crystal, eta: float) -> np.ndarray:
    reciprocal_crystal = Crystal(lattice=crystal.reciprocal_lattice, atoms=[])
    vectors = reciprocal_crystal.lattice_vectors_in_shell(4.0 * eta)
    return np.asarray([G for G in vectors if np.linalg.norm(G) > 1e-12], dtype=float)


def ewald_energy(crystal: Crystal, eta: float | None = None) -> float:
    """
    Nuclear-nuclear Ewald energy per unit cell for a 3D crystal.

    The G=0 and one-body self terms are omitted; in the full HF energy their
    long-range counterparts are cancelled by electron terms. This routine
    returns the finite ion-ion repulsion part used for Phase 20c validation.
    """
    eta_val = _ewald_eta(crystal, eta)
    coords = crystal.coords_bohr
    charges = np.asarray([atom.Z for atom in crystal.atoms], dtype=float)
    volume = abs(float(np.linalg.det(crystal.lattice)))
    real_vectors = _ewald_real_vectors(crystal, eta_val)
    reciprocal_vectors = _ewald_reciprocal_vectors(crystal, eta_val)

    e_real = 0.0
    for i, ri in enumerate(coords):
        for j, rj in enumerate(coords):
            for R in real_vectors:
                if i == j and np.linalg.norm(R) < 1e-12:
                    continue
                dist = float(np.linalg.norm(ri - rj + R))
                if dist > 1e-12:
                    e_real += charges[i] * charges[j] * erfc(eta_val * dist) / dist
    e_real *= 0.5

    e_recip = 0.0
    for G in reciprocal_vectors:
        g2 = float(np.dot(G, G))
        structure = np.sum(charges * np.exp(1j * (coords @ G)))
        e_recip += exp(-g2 / (4.0 * eta_val * eta_val)) * abs(structure) ** 2 / g2
    e_recip *= 2.0 * pi / volume

    return float(e_real + e_recip)


def ewald_hcore(
    crystal: Crystal,
    basis_fn: BasisSet,
    kpoints: np.ndarray,
    eta: float | None = None,
) -> np.ndarray:
    """
    Build a 3D Ewald-aware core Hamiltonian for each k-point.

    Phase 20c keeps the explicit Gaussian one-electron blocks from Phase 20a and
    adds the finite Ewald ion-lattice shift through the overlap matrix. The full
    3D HF driver uses PySCF for the production LiH reference path when available.
    """
    eta_val = _ewald_eta(crystal, eta)
    H = bloch_hcore(crystal, basis_fn, kpoints)
    S = bloch_overlap(crystal, basis_fn, kpoints)
    shift = -ewald_energy(crystal, eta_val) / max(float(crystal.n_electrons), 1.0)
    return (H + shift * S).astype(np.complex128)



@dataclass
class BandStructureResult:
    """Single-particle tight-binding band structure along a k-path."""

    band_energies: np.ndarray
    kpoints: np.ndarray
    x_coords: list[float]
    tick_positions: list[float]
    tick_labels: list[str]
    n_occ: int


def _display_klabel(label: str) -> str:
    return "Γ" if label in {"G", "Gamma", "GAMMA"} else label


def kpath(
    crystal: Crystal,
    special_points: dict[str, np.ndarray],
    path: str,
    n_points: int,
) -> tuple[np.ndarray, list[float], list[str]]:
    """Build a linear high-symmetry k-path in Cartesian Bohr^-1.

    `path` is a hyphen-separated label string such as `"G-X-M-G"`; each
    segment contains `n_points` points including both endpoints.
    """
    if n_points <= 1:
        raise ValueError("n_points must be greater than 1")
    labels = path.split("-")
    if len(labels) < 2:
        raise ValueError("path must contain at least two special points")
    missing = [label for label in labels if label not in special_points]
    if missing:
        raise ValueError(f"special point(s) missing from dictionary: {missing}")

    kpoints: list[np.ndarray] = []
    x_coords: list[float] = []
    distance = 0.0
    for left_label, right_label in zip(labels[:-1], labels[1:]):
        start = np.asarray(special_points[left_label], dtype=float)
        stop = np.asarray(special_points[right_label], dtype=float)
        if start.shape != (3,) or stop.shape != (3,):
            raise ValueError("special point coordinates must have shape (3,)")
        segment = stop - start
        segment_len = float(np.linalg.norm(segment))
        for i in range(n_points):
            t = i / (n_points - 1)
            kpoints.append(start + t * segment)
            x_coords.append(distance + t * segment_len)
        distance += segment_len
    return np.asarray(kpoints, dtype=float), x_coords, [_display_klabel(label) for label in labels]


def _kpath_tick_positions(x_coords: list[float], n_segments: int, n_points: int) -> list[float]:
    positions = [x_coords[0]]
    for segment in range(n_segments):
        positions.append(x_coords[(segment + 1) * n_points - 1])
    return positions


def band_structure(
    crystal: Crystal,
    basis_fn: BasisSet,
    special_points: dict[str, np.ndarray],
    path: str,
    n_points: int = 50,
) -> BandStructureResult:
    """Compute a native one-electron tight-binding band structure.

    The generalized eigenproblem H_core(k) C(k) = S(k) C(k) E(k) is solved
    along a high-symmetry k-path. This is a tight-binding-level band structure:
    real quasiparticle bands require HF exchange or DFT XC contributions, as in
    production periodic electronic-structure codes.
    """
    kpts, x_coords, tick_labels = kpath(crystal, special_points, path, n_points)
    S_k = bloch_overlap(crystal, basis_fn, kpts)
    if crystal.ndim == 3:
        H_k = ewald_hcore(crystal, basis_fn, kpts)
    else:
        H_k = bloch_hcore(crystal, basis_fn, kpts)
    band_energies = np.zeros((len(kpts), basis_fn.n_basis(crystal)), dtype=float)
    for ik in range(len(kpts)):
        eps, _ = _generalized_eigh(H_k[ik], S_k[ik])
        band_energies[ik] = eps
    n_segments = len(path.split("-")) - 1
    return BandStructureResult(
        band_energies=band_energies,
        kpoints=kpts,
        x_coords=x_coords,
        tick_positions=_kpath_tick_positions(x_coords, n_segments, n_points),
        tick_labels=tick_labels,
        n_occ=max(1, int(np.ceil(crystal.n_electrons / 2))),
    )




@dataclass
class DOSResult:
    """Gaussian-broadened density of states from a band structure."""

    energies: np.ndarray
    dos: np.ndarray
    e_fermi: float


def dos(
    band_result: BandStructureResult,
    n_grid: int = 500,
    sigma: float = 0.02,
    e_min: float | None = None,
    e_max: float | None = None,
) -> DOSResult:
    """Compute a Gaussian-broadened DOS from native band energies."""
    if n_grid <= 1:
        raise ValueError("n_grid must be greater than 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    bands = np.asarray(band_result.band_energies, dtype=float)
    if bands.ndim != 2 or bands.size == 0:
        raise ValueError("band energies must have shape (n_kpts, n_bands)")
    n_kpts, n_bands = bands.shape
    n_occ = min(max(int(band_result.n_occ), 1), n_bands)
    e_fermi = float(np.max(bands[:, n_occ - 1]))
    lo = float(np.min(bands) if e_min is None else e_min)
    hi = float(np.max(bands) if e_max is None else e_max)
    if lo >= hi:
        raise ValueError("e_min must be smaller than e_max")
    margin = 4.0 * sigma
    if e_min is None:
        lo -= margin
    if e_max is None:
        hi += margin
    energies = np.linspace(lo, hi, n_grid)
    prefactor = 1.0 / (sigma * sqrt(2.0 * pi) * n_kpts)
    broadened = np.exp(-0.5 * ((energies[:, None, None] - bands[None, :, :]) / sigma) ** 2)
    return DOSResult(energies=energies, dos=prefactor * np.sum(broadened, axis=(1, 2)), e_fermi=e_fermi)


@dataclass
class PhononResult:
    """Nuclear-repulsion-only phonon bands along a q-path."""

    frequencies: np.ndarray
    qpoints: np.ndarray
    x_coords: list[float]
    tick_positions: list[float]
    tick_labels: list[str]


def _pair_repulsion_second_derivative(
    atom_a: Atom,
    atom_b: Atom,
    R: np.ndarray,
    alpha: int,
    beta: int,
    h: float,
) -> float:
    def pair_energy(sign_a: float, sign_b: float) -> float:
        disp_a = np.zeros(3)
        disp_b = np.zeros(3)
        disp_a[alpha] = sign_a * h
        disp_b[beta] = sign_b * h
        delta = atom_a.coords + disp_a - (atom_b.coords + R + disp_b)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-12:
            return 0.0
        return atom_a.Z * atom_b.Z / distance

    return (
        pair_energy(1.0, 1.0)
        - pair_energy(1.0, -1.0)
        - pair_energy(-1.0, 1.0)
        + pair_energy(-1.0, -1.0)
    ) / (4.0 * h * h)


def _nuclear_force_constant_blocks(
    crystal: Crystal,
    h: float,
    r_max_factor: float,
) -> dict[tuple[float, float, float], np.ndarray]:
    max_lattice_len = float(np.max(np.linalg.norm(crystal.lattice, axis=1)))
    R_vectors = crystal.lattice_vectors_in_shell(r_max_factor * max_lattice_len)
    n_cart = 3 * crystal.n_atoms
    blocks: dict[tuple[float, float, float], np.ndarray] = {}
    zero_key = _block_key(np.zeros(3))
    blocks[zero_key] = np.zeros((n_cart, n_cart), dtype=float)
    for R in R_vectors:
        if np.linalg.norm(R) < 1e-12:
            continue
        block = np.zeros((n_cart, n_cart), dtype=float)
        for ia, atom_a in enumerate(crystal.atoms):
            for ib, atom_b in enumerate(crystal.atoms):
                for alpha in range(3):
                    for beta in range(3):
                        row = 3 * ia + alpha
                        col = 3 * ib + beta
                        block[row, col] = _pair_repulsion_second_derivative(atom_a, atom_b, R, alpha, beta, h)
        blocks[_block_key(R)] = block
        blocks[zero_key] -= block
    return blocks


def phonon_band_structure(
    crystal: Crystal,
    basis_fn: BasisSet,
    special_points: dict[str, np.ndarray],
    path: str,
    n_points: int = 30,
    h: float = 0.01,
    r_max_factor: float = 4.0,
) -> PhononResult:
    """Compute 1D nuclear-repulsion-only phonon bands.

    This is an educational dynamical-matrix construction. It excludes electronic
    Hellmann-Feynman/Pulay force constants, so the frequencies are not physical
    production phonons.
    """
    del basis_fn
    if crystal.ndim != 1:
        raise NotImplementedError("phonon_band_structure currently supports 1D crystals only")
    if h <= 0.0:
        raise ValueError("h must be positive")
    if r_max_factor <= 0.0:
        raise ValueError("r_max_factor must be positive")
    qpts, x_coords, tick_labels = kpath(crystal, special_points, path, n_points)
    blocks = _nuclear_force_constant_blocks(crystal, h, r_max_factor)
    return _phonon_result_from_blocks(crystal, blocks, qpts, x_coords, tick_labels, path, n_points)


def _phonon_result_from_blocks(
    crystal: Crystal,
    blocks: dict[tuple[float, float, float], np.ndarray],
    qpts: np.ndarray,
    x_coords: list[float],
    tick_labels: list[str],
    path: str,
    n_points: int,
) -> PhononResult:
    masses = np.repeat([ATOMIC_MASS[atom.Z] for atom in crystal.atoms], 3)
    mass_scale = np.sqrt(np.outer(masses, masses))
    frequencies = np.zeros((len(qpts), 3 * crystal.n_atoms), dtype=float)
    for iq, q in enumerate(qpts):
        D = np.zeros((3 * crystal.n_atoms, 3 * crystal.n_atoms), dtype=np.complex128)
        for key, block in blocks.items():
            R = np.asarray(key, dtype=float)
            D += np.exp(1j * float(q @ R)) * block / mass_scale
        D = 0.5 * (D + D.conj().T)
        omega2 = np.linalg.eigvalsh(D).real
        frequencies[iq] = np.sqrt(np.maximum(omega2, 0.0))
    n_segments = len(path.split("-")) - 1
    return PhononResult(
        frequencies=frequencies,
        qpoints=qpts,
        x_coords=x_coords,
        tick_positions=_kpath_tick_positions(x_coords, n_segments, n_points),
        tick_labels=tick_labels,
    )


def _scf_energy_for_lattice(
    crystal: Crystal,
    basis_fn: BasisSet,
    lattice: np.ndarray,
    r_max_factor: float,
    scf_kwargs: dict | None,
) -> float:
    trial = Crystal(
        lattice=np.asarray(lattice, dtype=float),
        atoms=[Atom(atom.symbol, atom.coords.copy()) for atom in crystal.atoms],
        charge=crystal.charge,
        multiplicity=crystal.multiplicity,
        name=crystal.name,
    )
    kwargs = dict(scf_kwargs or {})
    kwargs.setdefault("r_max_factor", r_max_factor)
    kmesh = kwargs.pop("kmesh", (1,))
    result = periodic_hf(trial, basis_fn, monkhorst_pack(trial.lattice, kmesh), **kwargs)
    return result.energy_per_cell


def periodic_force_constants(
    crystal: Crystal,
    basis_fn: BasisSet,
    h: float = 0.01,
    r_max_factor: float = 4.0,
    scf_kwargs: dict | None = None,
) -> dict[tuple[float, float, float], np.ndarray]:
    """Finite-difference force constants from 1D periodic HF total energy.

    The current educational periodic HF engine represents one primitive cell.
    For a 1D lattice, relative displacement of atom B in cell R is mapped to a
    small change of the lattice vector R/n before evaluating the SCF energy.
    This includes electronic and nuclear energy response in the same transparent
    finite-difference path used by the rest of MOLEKUL.
    """
    if crystal.ndim != 1:
        raise NotImplementedError("periodic_force_constants currently supports 1D crystals only")
    if h <= 0.0:
        raise ValueError("h must be positive")
    if r_max_factor <= 0.0:
        raise ValueError("r_max_factor must be positive")

    max_lattice_len = float(np.max(np.linalg.norm(crystal.lattice, axis=1)))
    R_vectors = crystal.lattice_vectors_in_shell(r_max_factor * max_lattice_len)
    a_vec = crystal.lattice[0]
    denom = float(np.dot(a_vec, a_vec))
    n_cart = 3 * crystal.n_atoms
    zero_key = _block_key(np.zeros(3))
    blocks: dict[tuple[float, float, float], np.ndarray] = {zero_key: np.zeros((n_cart, n_cart), dtype=float)}

    for R in R_vectors:
        coeff = int(round(float(np.dot(R, a_vec) / denom)))
        if coeff == 0:
            continue
        block = np.zeros((n_cart, n_cart), dtype=float)
        for ia in range(crystal.n_atoms):
            for ib in range(crystal.n_atoms):
                for alpha in range(3):
                    for beta in range(3):
                        row = 3 * ia + alpha
                        col = 3 * ib + beta

                        def energy(sign_a: float, sign_b: float) -> float:
                            relative = np.zeros(3)
                            relative[alpha] -= sign_a * h / coeff
                            relative[beta] += sign_b * h / coeff
                            lattice = crystal.lattice.copy()
                            lattice[0] = a_vec + relative
                            return _scf_energy_for_lattice(crystal, basis_fn, lattice, r_max_factor, scf_kwargs)

                        block[row, col] = (
                            energy(1.0, 1.0)
                            - energy(1.0, -1.0)
                            - energy(-1.0, 1.0)
                            + energy(-1.0, -1.0)
                        ) / (4.0 * h * h)
        blocks[_block_key(R)] = block
        blocks[zero_key] -= block
    return blocks


def phonon_band_structure_full(
    crystal: Crystal,
    basis_fn: BasisSet,
    special_points: dict[str, np.ndarray],
    path: str,
    n_points: int = 30,
    h: float = 0.01,
    r_max_factor: float = 4.0,
) -> PhononResult:
    """Full 1D phonon bands from periodic SCF finite-difference force constants."""
    if crystal.ndim != 1:
        raise NotImplementedError("phonon_band_structure_full currently supports 1D crystals only")
    qpts, x_coords, tick_labels = kpath(crystal, special_points, path, n_points)
    blocks = periodic_force_constants(crystal, basis_fn, h=h, r_max_factor=r_max_factor)
    return _phonon_result_from_blocks(crystal, blocks, qpts, x_coords, tick_labels, path, n_points)


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
    Run cutoff periodic Hartree-Fock for 1D crystals.

    Uses the Phase 20a per-cell nuclear-attraction convention with a finite
    real-space cutoff. Periodic 2-electron integrals (J/K) for 3D systems
    require Ewald-screened Coulomb treatment; these are beyond the scope of
    this educational engine. Use ewald_energy() and bloch_hcore() directly
    for 3D single-particle properties.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if conv_tol <= 0.0:
        raise ValueError("conv_tol must be positive")

    kpts = _as_kpoints(kpoints)
    if crystal.ndim != 1:
        raise NotImplementedError(
            "periodic_hf supports 1D crystals only. "
            "Full 3D periodic HF requires Ewald-screened J/K integrals beyond this "
            "codebase's scope. Use ewald_energy() and bloch_hcore() for 3D "
            "single-particle (tight-binding) properties."
        )

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
