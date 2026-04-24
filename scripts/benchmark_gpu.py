"""
GPU vs CPU benchmark for RHF SCF.

Tests multiple molecules/basis sets and logs results.

Usage:
    conda run -n ai python scripts/benchmark_gpu.py
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.basis_631gstar import get_631gstar
from molekul.basis_ccpvdz import get_ccpvdz
from molekul.gpu import run_benchmark

BOHR = 1.0 / 0.529177

# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------

def make_h2():
    return Molecule([Atom("H", [0, 0, 0]), Atom("H", [0, 0, 0.74 * BOHR])], 0, 1)

def make_h2o():
    b = 0.9584 * BOHR
    import numpy as np
    ang = 104.52 * np.pi / 180
    return Molecule([
        Atom("O", [0, 0, 0]),
        Atom("H", [ b * np.sin(ang/2), 0,  b * np.cos(ang/2)]),
        Atom("H", [-b * np.sin(ang/2), 0,  b * np.cos(ang/2)]),
    ], 0, 1)

def make_ethanol():
    # Ethanol C2H5OH (experimental geometry, Angstrom → bohr)
    def A(sym, x, y, z):
        return Atom(sym, [x * BOHR, y * BOHR, z * BOHR])
    return Molecule([
        A("C",  0.000,  0.000,  0.000),
        A("C",  1.522,  0.000,  0.000),
        A("O",  2.068,  1.177,  0.000),
        A("H", -0.390,  1.020,  0.000),
        A("H", -0.390, -0.509,  0.884),
        A("H", -0.390, -0.509, -0.884),
        A("H",  1.888, -0.509,  0.884),
        A("H",  1.888, -0.509, -0.884),
        A("H",  2.975,  1.177,  0.000),
    ], 0, 1)

def make_butane():
    # n-Butane C4H10, STO-3G → 30 basis functions
    def A(sym, x, y, z):
        return Atom(sym, [x * BOHR, y * BOHR, z * BOHR])
    return Molecule([
        A("C",  0.000,  0.000,  0.000),
        A("C",  1.540,  0.000,  0.000),
        A("C",  2.060,  1.453,  0.000),
        A("C",  3.600,  1.453,  0.000),
        A("H", -0.390,  1.027,  0.000),
        A("H", -0.390, -0.513,  0.890),
        A("H", -0.390, -0.513, -0.890),
        A("H",  1.930, -0.513,  0.890),
        A("H",  1.930, -0.513, -0.890),
        A("H",  1.670,  1.966,  0.890),
        A("H",  1.670,  1.966, -0.890),
        A("H",  3.990,  0.427,  0.000),
        A("H",  3.990,  1.966,  0.890),
        A("H",  3.990,  1.966, -0.890),
    ], 0, 1)

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

results = []

cases = [
    ("H2",      make_h2(),      STO3G,          "STO-3G"),
    ("H2O",     make_h2o(),     STO3G,          "STO-3G"),
    ("H2O",     make_h2o(),     get_631gstar(), "6-31G*"),
    ("H2O",     make_h2o(),     get_ccpvdz(),   "cc-pVDZ"),
    ("Ethanol", make_ethanol(), STO3G,          "STO-3G"),
    ("Butane",  make_butane(),  STO3G,          "STO-3G"),
]

print("\nMOLEKUL GPU Benchmark — RTX 5090 vs CPU")
print("="*60)

for mol_name, mol, basis_cls, basis_name in cases:
    label = f"{mol_name}/{basis_name}"
    try:
        r = run_benchmark(mol, basis_cls, molecule_name=label, n_repeat=3, verbose=True)
        results.append({
            "molecule": mol_name,
            "basis": basis_name,
            "n_basis": r.n_basis,
            "n_electrons": r.n_electrons,
            "cpu_eri_ms": round(r.cpu_eri_time * 1000, 2),
            "cpu_scf_ms": round(r.cpu_scf_time * 1000, 2),
            "gpu_transfer_ms": round(r.gpu_transfer_time * 1000, 2),
            "gpu_scf_ms": round(r.gpu_scf_time * 1000, 2),
            "speedup_scf": round(r.speedup_scf, 2),
            "cpu_energy": r.cpu_energy,
            "gpu_energy": r.gpu_energy,
            "energy_diff": r.energy_diff,
        })
    except Exception as e:
        print(f"  ERROR for {label}: {e}")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print("\n\nFINAL SUMMARY TABLE")
print("="*80)
print(f"{'Molecule/Basis':25s}  {'N_bas':>5}  {'CPU SCF':>10}  {'GPU SCF':>10}  {'Speedup':>8}  {'ΔE':>10}")
print("-"*80)
for r in results:
    print(f"{r['molecule']+'/' + r['basis']:25s}  {r['n_basis']:>5}  "
          f"{r['cpu_scf_ms']:>8.1f}ms  {r['gpu_scf_ms']:>8.1f}ms  "
          f"{r['speedup_scf']:>7.1f}×  {r['energy_diff']:>10.2e}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

log = {
    "phase": "GPU_benchmark",
    "date": datetime.date.today().isoformat(),
    "hardware": {"gpu": "NVIDIA RTX 5090", "cuda": "12.8"},
    "results": results,
}
os.makedirs("outputs/logs", exist_ok=True)
with open("outputs/logs/gpu_benchmark.json", "w") as f:
    json.dump(log, f, indent=2)
print("\nResults saved to outputs/logs/gpu_benchmark.json")
