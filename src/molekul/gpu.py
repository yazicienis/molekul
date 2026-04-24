"""
GPU-accelerated SCF via PyTorch.

The integral computation (ERI, H_core) is done on CPU with our existing
code.  All subsequent matrix algebra (Fock build, diagonalisation,
density-matrix update) runs on GPU using torch.

Requires: torch with CUDA support.

Usage
-----
    from molekul.gpu import rhf_scf_gpu
    result = rhf_scf_gpu(molecule, basis, verbose=True)
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class GPUBenchResult:
    """Timing results for CPU vs GPU SCF."""
    molecule_name: str
    n_basis: int
    n_electrons: int

    # CPU timings (seconds)
    cpu_eri_time: float
    cpu_hcore_time: float
    cpu_scf_time: float
    cpu_total_time: float
    cpu_energy: float

    # GPU timings (seconds)
    gpu_transfer_time: float   # CPU→GPU data transfer
    gpu_scf_time: float        # SCF on GPU (excl. transfer)
    gpu_total_time: float      # transfer + SCF
    gpu_energy: float

    speedup_scf: float         # cpu_scf_time / gpu_scf_time
    energy_diff: float         # |E_cpu - E_gpu|

    n_iter_cpu: int
    n_iter_gpu: int


# ---------------------------------------------------------------------------
# GPU Fock build
# ---------------------------------------------------------------------------

def _fock_gpu(H_core: "torch.Tensor", P: "torch.Tensor",
              eri: "torch.Tensor") -> "torch.Tensor":
    """
    Build Fock matrix on GPU.

    F_μν = H_μν + Σ_{λσ} P_{λσ} [(μν|λσ) - ½(μλ|νσ)]
         = H + J - ½K

    eri : (n, n, n, n) in physicist or chemist notation (μν|λσ)
    """
    J = torch.einsum("ls,mnls->mn", P, eri)
    K = torch.einsum("ls,mlns->mn", P, eri)
    return H_core + J - 0.5 * K


def _diis_extrapolate_gpu(focks: list, errors: list) -> "torch.Tensor":
    """DIIS extrapolation on GPU tensors."""
    n = len(focks)
    B = torch.zeros((n + 1, n + 1), dtype=focks[0].dtype, device=focks[0].device)
    for i in range(n):
        for j in range(n):
            B[i, j] = torch.dot(errors[i].ravel(), errors[j].ravel())
    B[n, :] = -1.0
    B[:, n] = -1.0
    B[n, n] = 0.0
    rhs = torch.zeros(n + 1, dtype=focks[0].dtype, device=focks[0].device)
    rhs[n] = -1.0
    try:
        c = torch.linalg.solve(B, rhs)
    except Exception:
        return focks[-1]
    F_diis = torch.zeros_like(focks[0])
    for i in range(n):
        F_diis += c[i] * focks[i]
    return F_diis


def rhf_scf_gpu(molecule, basis,
                max_iter: int = 100,
                e_conv: float = 1e-10,
                d_conv: float = 1e-8,
                diis_start: int = 2,
                diis_size: int = 8,
                device: str = "cuda",
                verbose: bool = True) -> dict:
    """
    RHF SCF on GPU (PyTorch).  Returns dict with energy and timing.

    Integrals are computed on CPU with the existing code, then transferred
    to GPU for all subsequent linear algebra.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install torch.")

    dev = torch.device(device)
    dtype = torch.float64

    from .integrals import build_core_hamiltonian, build_overlap
    from .eri import build_eri

    n_occ = molecule.n_alpha

    # ---- CPU: integrals ------------------------------------------------
    t0 = time.perf_counter()
    S_np   = build_overlap(basis, molecule)
    H_np   = build_core_hamiltonian(basis, molecule)
    t_hcore = time.perf_counter() - t0

    t0 = time.perf_counter()
    eri_np = build_eri(basis, molecule)
    t_eri = time.perf_counter() - t0

    E_nuc = molecule.nuclear_repulsion_energy()

    # ---- Transfer to GPU ------------------------------------------------
    t0 = time.perf_counter()
    S   = torch.tensor(S_np,   dtype=dtype, device=dev)
    H   = torch.tensor(H_np,   dtype=dtype, device=dev)
    eri = torch.tensor(eri_np, dtype=dtype, device=dev)
    if device == "cuda":
        torch.cuda.synchronize()
    t_transfer = time.perf_counter() - t0

    # ---- Orthogonaliser X = S^{-1/2} on GPU ----------------------------
    vals, vecs = torch.linalg.eigh(S)
    X  = vecs @ torch.diag(vals ** -0.5) @ vecs.T
    Xt = X.T

    # ---- Initial guess: diagonalise H_core ------------------------------
    Fp   = Xt @ H @ X
    _, Cp = torch.linalg.eigh(Fp)
    C    = X @ Cp
    P    = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    energy   = 0.0
    converged = False
    n_iter   = 0
    diis_focks  = []
    diis_errors = []

    t_scf_start = time.perf_counter()

    for it in range(1, max_iter + 1):
        n_iter = it
        F = _fock_gpu(H, P, eri)

        # Electronic energy
        E_elec = 0.5 * torch.sum(P * (H + F))
        E_total = float(E_elec) + E_nuc
        dE = E_total - energy
        energy = E_total

        # DIIS error
        FPS = F @ P @ S
        err_mat = FPS - FPS.T
        if it >= diis_start:
            diis_focks.append(F.clone())
            diis_errors.append(err_mat.clone())
            if len(diis_focks) > diis_size:
                diis_focks.pop(0); diis_errors.pop(0)
            F_scf = _diis_extrapolate_gpu(diis_focks, diis_errors)
        else:
            F_scf = F

        Fp_s = Xt @ F_scf @ X
        eps, Cp_s = torch.linalg.eigh(Fp_s)
        C_new = X @ Cp_s
        P_new = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T
        dP = float(torch.max(torch.abs(P_new - P)))
        P = P_new

        if abs(dE) < e_conv and dP < d_conv and it > 1:
            converged = True
            break

    if device == "cuda":
        torch.cuda.synchronize()
    t_scf = time.perf_counter() - t_scf_start

    if verbose:
        print(f"  GPU SCF {'converged' if converged else 'NOT converged'} "
              f"in {n_iter} iterations: E = {energy:.10f} Ha")
        print(f"  Timings: transfer={t_transfer*1000:.1f} ms  "
              f"SCF={t_scf*1000:.1f} ms")

    return {
        "energy": energy,
        "n_iter": n_iter,
        "converged": converged,
        "t_hcore": t_hcore,
        "t_eri": t_eri,
        "t_transfer": t_transfer,
        "t_scf": t_scf,
    }


# ---------------------------------------------------------------------------
# CPU baseline (same logic, numpy)
# ---------------------------------------------------------------------------

def rhf_scf_cpu_timed(molecule, basis,
                      max_iter: int = 100,
                      e_conv: float = 1e-10,
                      d_conv: float = 1e-8,
                      diis_start: int = 2,
                      diis_size: int = 8) -> dict:
    """RHF SCF on CPU with timing breakdown."""
    from .integrals import build_core_hamiltonian, build_overlap
    from .eri import build_eri

    n_occ = molecule.n_alpha
    E_nuc = molecule.nuclear_repulsion_energy()

    t0 = time.perf_counter()
    S  = build_overlap(basis, molecule)
    H  = build_core_hamiltonian(basis, molecule)
    t_hcore = time.perf_counter() - t0

    t0 = time.perf_counter()
    eri = build_eri(basis, molecule)
    t_eri = time.perf_counter() - t0

    vals, vecs = np.linalg.eigh(S)
    X  = vecs @ np.diag(vals ** -0.5) @ vecs.T
    Xt = X.T

    Fp   = Xt @ H @ X
    _, Cp = np.linalg.eigh(Fp)
    C    = X @ Cp
    P    = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    energy = 0.0
    converged = False
    n_iter = 0
    diis_focks = []
    diis_errors = []

    t_scf_start = time.perf_counter()

    for it in range(1, max_iter + 1):
        n_iter = it
        J = np.einsum("ls,mnls->mn", P, eri)
        K = np.einsum("ls,mlns->mn", P, eri)
        F = H + J - 0.5 * K

        E_elec = 0.5 * np.einsum("mn,mn->", P, H + F)
        E_total = float(E_elec) + E_nuc
        dE = E_total - energy
        energy = E_total

        FPS = F @ P @ S
        err_mat = FPS - FPS.T
        if it >= diis_start:
            diis_focks.append(F.copy())
            diis_errors.append(err_mat.copy())
            if len(diis_focks) > diis_size:
                diis_focks.pop(0); diis_errors.pop(0)
            n = len(diis_focks)
            B = np.zeros((n+1, n+1))
            for i in range(n):
                for j in range(n):
                    B[i, j] = np.dot(diis_errors[i].ravel(), diis_errors[j].ravel())
            B[n, :] = -1.0; B[:, n] = -1.0; B[n, n] = 0.0
            rhs = np.zeros(n+1); rhs[n] = -1.0
            try:
                c = np.linalg.solve(B, rhs)
                F_scf = sum(c[i]*diis_focks[i] for i in range(n))
            except Exception:
                F_scf = F
        else:
            F_scf = F

        Fp_s = Xt @ F_scf @ X
        eps, Cp_s = np.linalg.eigh(Fp_s)
        C_new = X @ Cp_s
        P_new = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T
        dP = float(np.max(np.abs(P_new - P)))
        P = P_new

        if abs(dE) < e_conv and dP < d_conv and it > 1:
            converged = True
            break

    t_scf = time.perf_counter() - t_scf_start

    return {
        "energy": energy,
        "n_iter": n_iter,
        "converged": converged,
        "t_hcore": t_hcore,
        "t_eri": t_eri,
        "t_scf": t_scf,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(molecule, basis, molecule_name: str = "mol",
                  n_repeat: int = 3, verbose: bool = True) -> GPUBenchResult:
    """
    Run CPU vs GPU timing comparison for RHF SCF.

    Repeats n_repeat times and takes the minimum (best-case) timing.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")

    n_basis = len(basis.basis_functions(molecule))

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {molecule_name}  "
              f"(n_basis={n_basis}, n_elec={molecule.n_electrons})")
        print(f"{'='*60}")

    # ---- CPU runs -------------------------------------------------------
    if verbose:
        print("\n  [CPU]")
    cpu_times = []
    cpu_energy = None
    cpu_iters = 0
    for r in range(n_repeat):
        res = rhf_scf_cpu_timed(molecule, basis)
        cpu_times.append(res)
        if verbose:
            print(f"    run {r+1}: ERI={res['t_eri']*1000:.1f}ms  "
                  f"SCF={res['t_scf']*1000:.1f}ms  E={res['energy']:.8f}")
    # best run = min SCF time
    best_cpu = min(cpu_times, key=lambda x: x["t_scf"])
    cpu_energy = best_cpu["energy"]
    cpu_iters  = best_cpu["n_iter"]

    # ---- GPU runs -------------------------------------------------------
    if verbose:
        print("\n  [GPU]")
    gpu_times = []
    gpu_energy = None
    gpu_iters  = 0
    # warm-up
    rhf_scf_gpu(molecule, basis, verbose=False)
    for r in range(n_repeat):
        res = rhf_scf_gpu(molecule, basis, verbose=False)
        gpu_times.append(res)
        if verbose:
            print(f"    run {r+1}: transfer={res['t_transfer']*1000:.1f}ms  "
                  f"SCF={res['t_scf']*1000:.1f}ms  E={res['energy']:.8f}")
    best_gpu = min(gpu_times, key=lambda x: x["t_scf"])
    gpu_energy = best_gpu["energy"]
    gpu_iters  = best_gpu["n_iter"]

    speedup = best_cpu["t_scf"] / best_gpu["t_scf"] if best_gpu["t_scf"] > 0 else float("inf")

    result = GPUBenchResult(
        molecule_name=molecule_name,
        n_basis=n_basis,
        n_electrons=molecule.n_electrons,
        cpu_eri_time=best_cpu["t_eri"],
        cpu_hcore_time=best_cpu["t_hcore"],
        cpu_scf_time=best_cpu["t_scf"],
        cpu_total_time=best_cpu["t_eri"] + best_cpu["t_hcore"] + best_cpu["t_scf"],
        cpu_energy=cpu_energy,
        gpu_transfer_time=best_gpu["t_transfer"],
        gpu_scf_time=best_gpu["t_scf"],
        gpu_total_time=best_gpu["t_transfer"] + best_gpu["t_scf"],
        gpu_energy=gpu_energy,
        speedup_scf=speedup,
        energy_diff=abs(cpu_energy - gpu_energy),
        n_iter_cpu=cpu_iters,
        n_iter_gpu=gpu_iters,
    )

    if verbose:
        print(f"\n  SUMMARY")
        print(f"  {'':20s}  {'CPU':>12}  {'GPU':>12}")
        print(f"  {'SCF time':20s}  {best_cpu['t_scf']*1000:>10.1f}ms  {best_gpu['t_scf']*1000:>10.1f}ms")
        print(f"  {'Transfer':20s}  {'—':>12}  {best_gpu['t_transfer']*1000:>10.1f}ms")
        print(f"  {'Speedup (SCF)':20s}  {speedup:>22.1f}×")
        print(f"  {'Energy diff':20s}  {result.energy_diff:>22.2e} Ha")

    return result
