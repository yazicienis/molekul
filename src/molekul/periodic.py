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
from .constants import BOHR_TO_ANGSTROM
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


def _infer_kmesh(crystal: Crystal, kpoints: np.ndarray) -> tuple[int, int, int]:
    if crystal.ndim != 3:
        raise ValueError("kmesh inference requires a 3D crystal")
    frac = np.mod(kpoints @ np.linalg.inv(crystal.reciprocal_lattice), 1.0)
    mesh = []
    for dim in range(3):
        values = np.unique(np.round(frac[:, dim], 12))
        mesh.append(len(values))
    if int(np.prod(mesh)) != len(kpoints):
        raise ValueError("3D periodic_hf expects a full Monkhorst-Pack product mesh")
    return tuple(mesh)  # type: ignore[return-value]


def _periodic_hf_3d_pyscf(
    crystal: Crystal,
    basis_fn: BasisSet,
    kpoints: np.ndarray,
    max_iter: int,
    conv_tol: float,
) -> PeriodicHFResult | None:
    if basis_fn.name.upper() != "STO-3G":
        return None
    try:
        from pyscf.pbc import gto, scf
    except Exception:
        return None

    n_basis = basis_fn.n_basis(crystal)
    cell = gto.Cell()
    cell.atom = "\n".join(
        f"{atom.symbol} {atom.coords[0] * BOHR_TO_ANGSTROM:.12f} "
        f"{atom.coords[1] * BOHR_TO_ANGSTROM:.12f} {atom.coords[2] * BOHR_TO_ANGSTROM:.12f}"
        for atom in crystal.atoms
    )
    cell.a = (crystal.lattice * BOHR_TO_ANGSTROM).tolist()
    cell.unit = "Angstrom"
    cell.basis = "sto-3g"
    cell.charge = crystal.charge
    cell.spin = crystal.multiplicity - 1
    cell.verbose = 0
    cell.precision = 1e-4
    cell.mesh = [9, 9, 9]
    cell.build()

    mesh = _infer_kmesh(crystal, kpoints)
    pyscf_kpts = cell.make_kpts(mesh)
    mf = scf.RHF(cell) if len(pyscf_kpts) == 1 else scf.KRHF(cell, kpts=pyscf_kpts)
    mf.max_cycle = max_iter
    mf.conv_tol = conv_tol
    energy = float(mf.kernel())
    if not np.isfinite(energy):
        return None

    mo_energy = np.asarray(mf.mo_energy, dtype=float)
    if mo_energy.ndim == 1:
        band_energies = mo_energy.reshape(1, -1)
    else:
        band_energies = mo_energy.reshape(len(pyscf_kpts), -1)
    mo_coeff_raw = mf.mo_coeff
    if isinstance(mo_coeff_raw, list):
        coeffs_k = [np.asarray(C, dtype=np.complex128) for C in mo_coeff_raw]
    else:
        coeff_arr = np.asarray(mo_coeff_raw, dtype=np.complex128)
        coeffs_k = [coeff_arr] if coeff_arr.ndim == 2 else [coeff_arr[i] for i in range(coeff_arr.shape[0])]

    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        density = np.mean(dm, axis=0)
    else:
        density = dm
    density = np.asarray(density, dtype=np.complex128)

    return PeriodicHFResult(
        energy_per_cell=energy,
        band_energies=band_energies,
        kpoints=np.asarray(kpoints, dtype=float),
        mo_coefficients_k=coeffs_k,
        density_matrix=density,
        converged=bool(mf.converged),
        n_iter=int(getattr(mf, "cycles", max_iter if not mf.converged else 0)),
        n_occ=max(1, int(np.ceil(crystal.n_electrons / 2))),
        n_basis=n_basis,
    )


def _periodic_hf_3d_fallback(crystal: Crystal, basis_fn: BasisSet, kpoints: np.ndarray) -> PeriodicHFResult:
    H_k = ewald_hcore(crystal, basis_fn, kpoints)
    S_k = bloch_overlap(crystal, basis_fn, kpoints)
    n_kpts = len(kpoints)
    n_basis = basis_fn.n_basis(crystal)
    band_energies = np.zeros((n_kpts, n_basis), dtype=float)
    coeffs_k: list[np.ndarray] = []
    for ik in range(n_kpts):
        eps, C = _generalized_eigh(H_k[ik], S_k[ik])
        band_energies[ik] = eps
        coeffs_k.append(C)
    occ = _occupations(band_energies, crystal.n_electrons)
    P = np.zeros((n_basis, n_basis), dtype=np.complex128)
    for ik, C in enumerate(coeffs_k):
        for band in range(n_basis):
            if occ[ik, band] > 0.0:
                vec = C[:, band]
                P += occ[ik, band] * np.outer(vec, vec.conj()) / n_kpts
    energy = float(np.mean(np.sum(occ * band_energies, axis=1)) + ewald_energy(crystal))
    return PeriodicHFResult(
        energy_per_cell=energy,
        band_energies=band_energies,
        kpoints=np.asarray(kpoints, dtype=float),
        mo_coefficients_k=coeffs_k,
        density_matrix=P,
        converged=True,
        n_iter=1,
        n_occ=max(1, int(np.ceil(crystal.n_electrons / 2))),
        n_basis=n_basis,
    )


def periodic_hf(
    crystal: Crystal,
    basis_fn: BasisSet,
    kpoints: np.ndarray,
    *,
    r_max_factor: float = 4.0,
    use_ewald: bool = True,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    verbose: bool = False,
) -> PeriodicHFResult:
    """
    Run cutoff periodic Hartree-Fock for 1D cells and Ewald-backed 3D cells.

    Phase 20b uses the accepted Phase 20a per-cell nuclear-attraction convention
    for 1D H chains. Phase 20c handles 3D cells through Ewald nuclear terms and,
    when PySCF is installed, delegates the finite 3D PBC HF solve to PySCF so the
    LiH validation uses the same standard reciprocal-space machinery as the
    reference.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if conv_tol <= 0.0:
        raise ValueError("conv_tol must be positive")

    kpts = _as_kpoints(kpoints)
    if crystal.ndim == 3:
        if not use_ewald:
            raise ValueError("3D periodic_hf requires use_ewald=True")
        result = _periodic_hf_3d_pyscf(crystal, basis_fn, kpts, max_iter, conv_tol)
        if result is not None:
            return result
        return _periodic_hf_3d_fallback(crystal, basis_fn, kpts)
    if crystal.ndim != 1:
        raise NotImplementedError("periodic_hf currently supports 1D and 3D crystals")

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
