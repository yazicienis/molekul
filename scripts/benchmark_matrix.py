"""
CPU vs GPU matris işlemi benchmark'ı.

ERI olmadan, doğrudan SCF'in içindeki matris işlemlerini ölçer:
  - Fock build:  J = einsum("ls,mnls->mn", P, eri)
  - Matris çarpımı: F @ P @ S
  - Özdeğer: eigh(F)

n_basis = 7 (H2O/STO-3G) ... 500 (büyük sistem simülasyonu)

Kullanım:
    conda run -n ai python scripts/benchmark_matrix.py
"""
import sys, os, json, datetime, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

dev = torch.device("cuda")
dtype = torch.float64

def fock_cpu(H, P, eri):
    J = np.einsum("ls,mnls->mn", P, eri)
    K = np.einsum("ls,mlns->mn", P, eri)
    return H + J - 0.5 * K

def fock_gpu(H, P, eri):
    J = torch.einsum("ls,mnls->mn", P, eri)
    K = torch.einsum("ls,mlns->mn", P, eri)
    return H + J - 0.5 * K

def time_fn(fn, *args, n_repeat=5):
    """Warm up + min timing."""
    fn(*args)  # warm-up
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn(*args)
        if hasattr(args[0], 'device') and str(args[0].device) != 'cpu':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)

results = []

print(f"\n{'='*72}")
print(f"  CPU vs GPU — SCF matris işlemleri (RTX 5090)")
print(f"{'='*72}")
print(f"\n  {'n_basis':>8}  {'CPU Fock':>12}  {'GPU Fock':>12}  {'Speedup':>10}  {'CPU eigh':>10}  {'GPU eigh':>10}")
print(f"  {'-'*72}")

for n in [7, 19, 25, 50, 100, 200, 300]:
    rng = np.random.default_rng(42)
    # Random symmetric positive definite matrices
    H_np  = rng.standard_normal((n, n)); H_np = (H_np + H_np.T) / 2
    P_np  = rng.standard_normal((n, n)); P_np = (P_np + P_np.T) / 2
    eri_np = rng.standard_normal((n, n, n, n))
    # Symmetrize ERI
    eri_np = 0.5 * (eri_np + eri_np.transpose(1, 0, 2, 3))
    eri_np = 0.5 * (eri_np + eri_np.transpose(0, 1, 3, 2))

    # CPU
    t_fock_cpu = time_fn(fock_cpu, H_np, P_np, eri_np)
    t_eigh_cpu = time_fn(np.linalg.eigh, H_np)

    # GPU
    H_g   = torch.tensor(H_np,   dtype=dtype, device=dev)
    P_g   = torch.tensor(P_np,   dtype=dtype, device=dev)
    eri_g = torch.tensor(eri_np, dtype=dtype, device=dev)
    t_fock_gpu = time_fn(fock_gpu, H_g, P_g, eri_g)
    t_eigh_gpu = time_fn(torch.linalg.eigh, H_g)

    speedup = t_fock_cpu / t_fock_gpu
    print(f"  {n:>8}  {t_fock_cpu*1000:>10.2f}ms  {t_fock_gpu*1000:>10.2f}ms  "
          f"{speedup:>9.1f}×  {t_eigh_cpu*1000:>8.2f}ms  {t_eigh_gpu*1000:>8.2f}ms")

    results.append({
        "n_basis": n,
        "cpu_fock_ms": round(t_fock_cpu * 1000, 3),
        "gpu_fock_ms": round(t_fock_gpu * 1000, 3),
        "speedup_fock": round(speedup, 2),
        "cpu_eigh_ms": round(t_eigh_cpu * 1000, 3),
        "gpu_eigh_ms": round(t_eigh_gpu * 1000, 3),
    })

print(f"\n  Not: n_basis <= 25 küçük sistem (H2O). GPU avantajı n_basis >= 100'de başlar.")

log = {
    "date": datetime.date.today().isoformat(),
    "hardware": {"gpu": "NVIDIA RTX 5090", "cuda": "12.8"},
    "description": "CPU vs GPU SCF matris işlemleri (sentetik, rasgele matrisler)",
    "results": results,
}
with open("outputs/logs/gpu_matrix_benchmark.json", "w") as f:
    json.dump(log, f, indent=2)
print("\n  Kaydedildi: outputs/logs/gpu_matrix_benchmark.json")
