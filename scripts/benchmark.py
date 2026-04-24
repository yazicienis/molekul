#!/usr/bin/env python3
"""
scripts/benchmark.py — Phase 8: CPU runtime benchmarking and scaling analysis.

Measures wall-clock time and memory estimates for every MOLEKUL computational
phase across six molecules spanning 2–36 basis functions (STO-3G).

Phases timed
------------
  integrals   : build_overlap + build_kinetic + build_nuclear   (one-electron)
  eri         : build_eri                                        (two-electron)
  scf         : rhf_scf (with ERI pre-built, so only SCF cost)
  grad        : numerical_gradient (6·N_atoms SCF calls via finite diff)
  opt         : optimize_geometry  (up to max_steps iterations)

Scope decisions
---------------
  gradient   — skipped for molecules with n_basis > 15 (too many SCF calls)
  opt        — only H2, HeH+, H2O (small, well-behaved)
  benzene    — ERI + SCF measured; gradient/opt skipped

Usage
-----
  python scripts/benchmark.py            # run all
  python scripts/benchmark.py --skip-large  # skip benzene ERI/SCF

Outputs
-------
  outputs/phase8_benchmark.txt
  outputs/logs/phase8_benchmark.json
  outputs/logs/phase8_benchmark.txt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from molekul.atoms       import Atom
from molekul.molecule    import Molecule
from molekul.basis_sto3g import STO3G
from molekul.integrals   import build_overlap, build_kinetic, build_nuclear
from molekul.eri         import build_eri
from molekul.rhf         import rhf_scf
from molekul.grad        import numerical_gradient
from molekul.optimizer   import optimize_geometry
from molekul.logging_utils import ExperimentLogger

REPO = Path(__file__).resolve().parents[1]
SEP  = "─" * 72

BOHR = 1.8897259886   # Å → bohr

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MolBench:
    name:          str
    n_atoms:       int
    n_electrons:   int
    n_basis:       int
    charge:        int

    # runtimes in seconds (None = skipped)
    t_integrals_s:  Optional[float] = None
    t_eri_s:        Optional[float] = None
    t_scf_s:        Optional[float] = None
    t_grad_s:       Optional[float] = None
    t_opt_s:        Optional[float] = None

    # SCF metadata
    scf_converged:  Optional[bool]  = None
    scf_iters:      Optional[int]   = None
    scf_energy_ha:  Optional[float] = None

    # gradient metadata
    grad_norm:      Optional[float] = None   # Hartree/bohr (max component)
    grad_n_calls:   Optional[int]   = None   # number of SCF calls for gradient

    # opt metadata
    opt_converged:  Optional[bool]  = None
    opt_steps:      Optional[int]   = None

    # memory estimates (bytes)
    mem_eri_bytes:  int = 0   # n^4 * 8  (full ERI array)
    mem_1e_bytes:   int = 0   # 3 * n^2 * 8  (S, T, V)

    # skip flags
    eri_skipped:  bool = False
    grad_skipped: bool = False
    opt_skipped:  bool = False

    def eri_mb(self) -> float:
        return self.mem_eri_bytes / 1e6

    def n4_count(self) -> int:
        return self.n_basis ** 4


# ---------------------------------------------------------------------------
# Molecule constructors
# ---------------------------------------------------------------------------

def _h2() -> Molecule:
    return Molecule([Atom("H",[0,0,0]), Atom("H",[0,0,1.4])],
                    charge=0, multiplicity=1, name="H2")

def _heh() -> Molecule:
    return Molecule([Atom("He",[0,0,0]), Atom("H",[0,0,1.4632])],
                    charge=1, multiplicity=1, name="HeH+")

def _h2o() -> Molecule:
    R, th = 1.870, math.radians(50.0)
    Hy = R * math.sin(th);  Hz = -R * math.cos(th)
    return Molecule([Atom("O",[0,0,0]),
                     Atom("H",[0, Hy, Hz]),
                     Atom("H",[0,-Hy, Hz])],
                    charge=0, multiplicity=1, name="H2O")

def _nh3() -> Molecule:
    # Experimental geometry: N-H = 1.012 Å, H-N-H = 107.8°
    return Molecule([
        Atom.from_angstrom("N",  0.000,  0.000,  0.116),
        Atom.from_angstrom("H",  0.000,  0.940, -0.272),
        Atom.from_angstrom("H",  0.814, -0.470, -0.272),
        Atom.from_angstrom("H", -0.814, -0.470, -0.272),
    ], charge=0, multiplicity=1, name="NH3")

def _ch4() -> Molecule:
    # Tetrahedral: C-H = 1.089 Å → each H displaced ±d along each axis
    d = 1.089 / math.sqrt(3) * BOHR   # bohr
    return Molecule([
        Atom("C", [0, 0, 0]),
        Atom("H", [ d,  d,  d]),
        Atom("H", [-d, -d,  d]),
        Atom("H", [-d,  d, -d]),
        Atom("H", [ d, -d, -d]),
    ], charge=0, multiplicity=1, name="CH4")

def _benzene() -> Molecule:
    # D6h: C-C = 1.397 Å, C-H = 1.085 Å, all in xy-plane
    R_CC, R_CH = 1.397, 1.085
    atoms: List[Atom] = []
    for i in range(6):
        a = math.radians(i * 60)
        atoms.append(Atom.from_angstrom("C",
            R_CC * math.cos(a), R_CC * math.sin(a), 0.0))
        atoms.append(Atom.from_angstrom("H",
            (R_CC + R_CH) * math.cos(a), (R_CC + R_CH) * math.sin(a), 0.0))
    return Molecule(atoms, charge=0, multiplicity=1, name="C6H6")


MOLECULES = [
    ("H2",   _h2,      True),    # (name, factory, run_opt)
    ("HeH+", _heh,     True),
    ("H2O",  _h2o,     True),
    ("NH3",  _nh3,     False),
    ("CH4",  _ch4,     False),
    ("C6H6", _benzene, False),   # large — ERI/SCF only
]

# Skip gradient if too many basis functions (gradient needs 6*N_atoms SCF calls)
GRAD_N_BASIS_LIMIT = 15


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def benchmark_molecule(mol: Molecule, run_opt: bool,
                       skip_large: bool, large_threshold: int = 15) -> MolBench:
    bfs    = STO3G.basis_functions(mol)
    n      = len(bfs)
    is_large = n > large_threshold

    bench = MolBench(
        name        = mol.name,
        n_atoms     = mol.n_atoms,
        n_electrons = mol.n_electrons,
        n_basis     = n,
        charge      = mol.charge,
        mem_eri_bytes = n**4 * 8,
        mem_1e_bytes  = 3 * n**2 * 8,
    )

    # ── One-electron integrals ─────────────────────────────────────────────
    _, bench.t_integrals_s = _timed(lambda: (
        build_overlap(STO3G, mol),
        build_kinetic(STO3G, mol),
        build_nuclear(STO3G, mol),
    ))

    # ── ERI ───────────────────────────────────────────────────────────────
    if is_large and skip_large:
        bench.eri_skipped = True
        print(f"    ERI: SKIPPED (n_basis={n} > {large_threshold}, --skip-large)")
        eri = None
    else:
        eri, bench.t_eri_s = _timed(build_eri, STO3G, mol)
        print(f"    ERI: {bench.t_eri_s:.3f}s")

    # ── SCF ───────────────────────────────────────────────────────────────
    if eri is None and is_large and skip_large:
        bench.eri_skipped = True
    else:
        scf_result, bench.t_scf_s = _timed(rhf_scf, mol, STO3G, verbose=False)
        bench.scf_converged = scf_result.converged
        bench.scf_iters     = scf_result.n_iter
        bench.scf_energy_ha = scf_result.energy_total
        print(f"    SCF: {bench.t_scf_s:.3f}s  ({bench.scf_iters} iter, "
              f"E={bench.scf_energy_ha:.8f} Ha, converged={bench.scf_converged})")

    # ── Numerical gradient ────────────────────────────────────────────────
    if n > GRAD_N_BASIS_LIMIT:
        bench.grad_skipped = True
        print(f"    Grad: SKIPPED (n_basis={n} > {GRAD_N_BASIS_LIMIT})")
    else:
        bench.grad_n_calls = 2 * 3 * mol.n_atoms   # central differences
        grad, bench.t_grad_s = _timed(numerical_gradient, mol, STO3G)
        bench.grad_norm = float(np.max(np.abs(grad)))
        print(f"    Grad: {bench.t_grad_s:.3f}s  "
              f"({bench.grad_n_calls} SCF calls, max|∂E/∂R|={bench.grad_norm:.4e})")

    # ── Geometry optimisation ─────────────────────────────────────────────
    if not run_opt or n > GRAD_N_BASIS_LIMIT:
        bench.opt_skipped = True
        if not run_opt:
            print(f"    Opt:  SKIPPED (not requested for this molecule)")
        else:
            print(f"    Opt:  SKIPPED (n_basis too large)")
    else:
        opt_result, bench.t_opt_s = _timed(
            optimize_geometry, mol, STO3G, verbose=False, max_steps=50,
        )
        bench.opt_converged = opt_result.converged
        bench.opt_steps     = opt_result.n_steps
        print(f"    Opt:  {bench.t_opt_s:.3f}s  "
              f"({bench.opt_steps} steps, converged={bench.opt_converged})")

    return bench


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_t(t: Optional[float]) -> str:
    if t is None:
        return "skipped"
    if t < 0.001:
        return f"{t*1000:.2f}ms"
    if t < 1.0:
        return f"{t*1000:.1f}ms"
    if t < 60:
        return f"{t:.2f}s"
    return f"{t/60:.1f}min"


def print_summary(results: List[MolBench]) -> None:
    print(f"\n{'='*72}")
    print("  SUMMARY TABLE")
    print(f"{'='*72}")

    # Header
    hdr = (f"{'Molecule':<8}  {'n_bas':>5}  {'n_elec':>6}  "
           f"{'1e-int':>8}  {'ERI':>8}  {'SCF':>8}  {'iter':>4}  "
           f"{'Grad':>8}  {'Opt':>8}  {'ERI_MB':>7}")
    print(hdr)
    print("─" * 72)

    for b in results:
        eri_mb = f"{b.eri_mb():.2f}" if not b.eri_skipped else "—"
        row = (f"{b.name:<8}  {b.n_basis:>5}  {b.n_electrons:>6}  "
               f"{_fmt_t(b.t_integrals_s):>8}  "
               f"{_fmt_t(b.t_eri_s):>8}  "
               f"{_fmt_t(b.t_scf_s):>8}  "
               f"{str(b.scf_iters or '—'):>4}  "
               f"{_fmt_t(b.t_grad_s):>8}  "
               f"{_fmt_t(b.t_opt_s):>8}  "
               f"{eri_mb:>7}")
        print(row)

    print(f"\n  Notes:")
    print(f"  • Gradient skipped for n_basis > {GRAD_N_BASIS_LIMIT}"
          f"  (6·N_atoms SCF calls per gradient)")
    print(f"  • Opt limited to H2, HeH+, H2O  (well-characterised test cases)")
    print(f"  • ERI scales as O(n⁴);  SCF scales as O(n⁴·N_iter)")
    print(f"  • All timings: wall-clock, single CPU thread, Python 3")


def scaling_analysis(results: List[MolBench]) -> None:
    """Fit log-log slope to ERI and SCF timings vs n_basis."""
    print(f"\n{SEP}")
    print("  SCALING ANALYSIS")
    print(SEP)

    eri_data = [(b.n_basis, b.t_eri_s) for b in results
                if b.t_eri_s is not None and b.n_basis >= 2]
    scf_data = [(b.n_basis, b.t_scf_s) for b in results
                if b.t_scf_s is not None and b.n_basis >= 2]
    grad_data = [(b.n_basis, b.t_grad_s) for b in results
                 if b.t_grad_s is not None and b.n_basis >= 2]

    for label, data in [("ERI", eri_data), ("SCF", scf_data), ("Grad", grad_data)]:
        if len(data) < 2:
            print(f"  {label}: insufficient data for slope fit")
            continue
        ns = np.array([d[0] for d in data], dtype=float)
        ts = np.array([d[1] for d in data], dtype=float)
        # log-log fit
        log_n = np.log(ns)
        log_t = np.log(ts)
        slope, intercept = np.polyfit(log_n, log_t, 1)
        print(f"  {label}: empirical scaling ~ n^{slope:.2f}  "
              f"(theoretical: ERI=n^4, Grad=6·N_atoms·n^4)")
        for n, t in data:
            print(f"    n={n:2d}  t={_fmt_t(t)}")


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _bench_to_dict(b: MolBench) -> dict:
    d = asdict(b)
    d["eri_mb"] = b.eri_mb()
    d["n4_count"] = b.n4_count()
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MOLEKUL CPU benchmark")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip ERI and SCF for molecules with n_basis > 15")
    args = parser.parse_args()

    log = ExperimentLogger("phase8", "benchmark")

    print("=" * 72)
    print("  Phase 8: CPU Performance Benchmark  —  MOLEKUL / STO-3G")
    print("=" * 72)
    print(f"  skip_large = {args.skip_large}")
    print()

    results: List[MolBench] = []
    t_wall_start = time.perf_counter()

    for name, factory, run_opt in MOLECULES:
        mol = factory()
        bfs = STO3G.basis_functions(mol)
        n   = len(bfs)
        print(f"{SEP}")
        print(f"  {name}  (n_basis={n}, n_elec={mol.n_electrons}, "
              f"charge={mol.charge})")
        print(SEP)

        bench = benchmark_molecule(mol, run_opt=run_opt,
                                   skip_large=args.skip_large)
        results.append(bench)
        log.metric(f"{name.lower().replace('+','p')}_n_basis",  n)
        if bench.t_integrals_s is not None:
            log.metric(f"{name.lower().replace('+','p')}_t_integrals_s", bench.t_integrals_s)
        if bench.t_eri_s is not None:
            log.metric(f"{name.lower().replace('+','p')}_t_eri_s",   bench.t_eri_s)
        if bench.t_scf_s is not None:
            log.metric(f"{name.lower().replace('+','p')}_t_scf_s",   bench.t_scf_s)
        if bench.t_grad_s is not None:
            log.metric(f"{name.lower().replace('+','p')}_t_grad_s",  bench.t_grad_s)
        if bench.t_opt_s is not None:
            log.metric(f"{name.lower().replace('+','p')}_t_opt_s",   bench.t_opt_s)
        if bench.scf_energy_ha is not None:
            log.metric(f"{name.lower().replace('+','p')}_energy_ha", bench.scf_energy_ha)

    t_total = time.perf_counter() - t_wall_start

    print_summary(results)
    scaling_analysis(results)

    # ── Known bottlenecks ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  BOTTLENECKS AND LIMITATIONS")
    print(SEP)
    print("""\
  1. ERI is O(n⁴): the dominant cost for all molecules above H2O.
     Pure Python loops — no vectorisation, no screening, no symmetry reuse.
  2. Gradient is O(6·N_atoms · n⁴): finite differences require 2·3·N SCF
     calls each building the full ERI. Unfeasible for benzene.
  3. No integral screening: no Cauchy-Schwarz or density-based cutoffs.
     All (μν|λσ) pairs are computed regardless of magnitude.
  4. No memory layout optimisation: ERI stored as dense n×n×n×n float64.
     For benzene: 36⁴ × 8 B ≈ 13 MB — small now, but grows as n⁸ would
     if the full 2-index transform were stored.
  5. SCF DIIS works well but the Fock build inside SCF reuses the stored ERI
     (fast); the bottleneck is the initial ERI construction, not SCF itself.
  6. No GPU acceleration (Phase 9+).
  7. All timings are single-threaded, no OpenMP/MKL parallelism.""")

    log.metric("total_wall_s", t_total)

    # ── Save structured log ───────────────────────────────────────────────
    import json
    for b in results:
        log.metric(f"bench_{b.name}", _bench_to_dict(b))

    txt_path, json_path = log.save()
    print(f"\n  Log: {txt_path.name},  {json_path.name}")

    # ── Save human-readable table to outputs/ ─────────────────────────────
    out_txt = REPO / "outputs" / "phase8_benchmark.txt"
    _write_text_report(results, out_txt, t_total, args.skip_large)
    print(f"  Report: outputs/phase8_benchmark.txt")

    print(f"\n  Total wall time: {_fmt_t(t_total)}")


def _write_text_report(results: List[MolBench], path: Path,
                       total_s: float, skip_large: bool) -> None:
    lines = [
        "MOLEKUL Phase 8 — CPU Performance Benchmark",
        f"Date: 2026-04-02",
        f"Basis: STO-3G",
        f"skip_large: {skip_large}",
        "",
        f"{'Molecule':<8}  {'n_bas':>5}  {'n_elec':>6}  "
        f"{'n_basis^4':>10}  {'ERI_MB':>7}  "
        f"{'1e-int':>8}  {'ERI':>10}  {'SCF':>10}  {'iter':>4}  "
        f"{'Grad':>10}  {'Opt':>10}",
        "─" * 100,
    ]
    for b in results:
        eri_mb  = f"{b.eri_mb():.3f}" if not b.eri_skipped else "—"
        lines.append(
            f"{b.name:<8}  {b.n_basis:>5}  {b.n_electrons:>6}  "
            f"{b.n4_count():>10}  {eri_mb:>7}  "
            f"{_fmt_t(b.t_integrals_s):>8}  "
            f"{_fmt_t(b.t_eri_s):>10}  "
            f"{_fmt_t(b.t_scf_s):>10}  "
            f"{str(b.scf_iters or '—'):>4}  "
            f"{_fmt_t(b.t_grad_s):>10}  "
            f"{_fmt_t(b.t_opt_s):>10}"
        )
    lines += [
        "",
        "Scaling notes:",
        "  ERI  ~ O(n^4)        — dominant cost",
        "  SCF  ~ O(n^4 * N_iter) — Fock build uses stored ERI",
        "  Grad ~ O(6*N_atoms * n^4) — finite differences, many SCF calls",
        "",
        f"Total benchmark wall time: {_fmt_t(total_s)}",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
