"""
Plot CPU vs GPU speedup trend from gpu_benchmark.json.

Usage:
    python scripts/plot_gpu_benchmark.py
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log_path = "outputs/logs/gpu_benchmark.json"
if not os.path.exists(log_path):
    sys.exit(f"Not found: {log_path}. Run benchmark_gpu.py first.")

with open(log_path) as f:
    data = json.load(f)

results = data["results"]
labels   = [f"{r['molecule']}\n{r['basis']}" for r in results]
n_basis  = [r["n_basis"]       for r in results]
speedup  = [r["speedup_scf"]   for r in results]
cpu_ms   = [r["cpu_scf_ms"]    for r in results]
gpu_ms   = [r["gpu_scf_ms"]    for r in results]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"CPU vs GPU (RTX 5090) — RHF SCF  [{data['date']}]", fontsize=13)

# --- Plot 1: Speedup vs N_basis ---
ax = axes[0]
colors = ["#e74c3c" if s < 1.0 else "#2ecc71" for s in speedup]
bars = ax.bar(range(len(labels)), speedup, color=colors, edgecolor="white", width=0.6)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Breakeven (1×)")
for i, (b, s) in enumerate(zip(bars, speedup)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f"{s:.2f}×", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("SCF Speedup (CPU time / GPU time)")
ax.set_title("Speedup vs Molecule/Basis")
ax.legend()
ax.set_ylim(0, max(speedup) * 1.25 + 0.2)

# --- Plot 2: SCF time vs N_basis ---
ax = axes[1]
x = np.array(n_basis)
order = np.argsort(x)
xs = x[order]
cpu_s = np.array(cpu_ms)[order]
gpu_s = np.array(gpu_ms)[order]
lbl_s = np.array(labels)[order]

ax.plot(xs, cpu_s, "o-", color="#3498db", label="CPU SCF", linewidth=2, markersize=7)
ax.plot(xs, gpu_s, "s-", color="#e67e22", label="GPU SCF", linewidth=2, markersize=7)
for xi, ci, gi, lb in zip(xs, cpu_s, gpu_s, lbl_s):
    ax.annotate(lb.replace("\n", "/"), (xi, max(ci, gi)),
                textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.set_xlabel("Number of basis functions (N)")
ax.set_ylabel("SCF time (ms)")
ax.set_title("SCF Time vs Basis Size")
ax.legend()
ax.set_yscale("log")

plt.tight_layout()
out = "outputs/gpu_speedup.png"
os.makedirs("outputs", exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
