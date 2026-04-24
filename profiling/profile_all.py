#!/usr/bin/env python3
"""
profiling/profile_all.py — Phase 9: Detailed CPU profiling of ERI and
gradient bottlenecks.

Profiling strategy
------------------
Three complementary approaches are applied:

1. cProfile  — full function call graph with ncalls, tottime, cumtime.
               Applied to: build_eri, rhf_scf, numerical_gradient (H2O).

2. line_profiler — line-by-line timing for the hottest functions:
               eri_primitive, _contracted_eri, build_eri, _build_fock.

3. Manual sub-phase timing — break each major routine into labelled
               segments timed with perf_counter:
               ERI:  loop overhead / _E coefficients / _R (Boys) / contraction
               SCF:  Fock build / DIIS / diagonalise / density update / energy
               Grad: per-displacement cost decomposition

Molecules used
--------------
H2O  (7 basis functions)  — primary profiling target: fast enough to repeat,
                             large enough to show meaningful call trees.
CH4  (9 basis functions)  — secondary: shows n^4 scaling in cProfile tottime.

Outputs
-------
  profiling/cprofile_eri_h2o.txt       — cProfile stats, sorted by tottime
  profiling/cprofile_scf_h2o.txt       — cProfile stats for full SCF
  profiling/cprofile_grad_h2o.txt      — cProfile stats for gradient
  profiling/lineprofile_eri.txt        — line-by-line for eri.py hot functions
  profiling/lineprofile_fock.txt       — line-by-line for _build_fock
  profiling/subphase_timing.txt        — manual timing decomposition table
  outputs/logs/phase9_profile.json     — structured JSON summary

Usage
-----
  python profiling/profile_all.py
"""

from __future__ import annotations

import cProfile
import io
import json
import math
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from line_profiler import LineProfiler

from molekul.atoms       import Atom
from molekul.molecule    import Molecule
from molekul.basis_sto3g import STO3G
from molekul.integrals   import build_overlap, build_kinetic, build_nuclear, _E, _R
from molekul.eri         import build_eri, eri_primitive, _contracted_eri
from molekul.rhf         import rhf_scf, _build_fock
from molekul.grad        import numerical_gradient

REPO      = Path(__file__).resolve().parents[1]
PROF_DIR  = REPO / "profiling"
LOGS_DIR  = REPO / "outputs" / "logs"
PROF_DIR.mkdir(exist_ok=True)

SEP = "─" * 72

BOHR = 1.8897259886


# ---------------------------------------------------------------------------
# Molecule factories
# ---------------------------------------------------------------------------

def _h2o() -> Molecule:
    R, th = 1.870, math.radians(50.0)
    return Molecule([
        Atom("O",  [0.0, 0.0, 0.0]),
        Atom("H",  [0.0,  R * math.sin(th), -R * math.cos(th)]),
        Atom("H",  [0.0, -R * math.sin(th), -R * math.cos(th)]),
    ], charge=0, multiplicity=1, name="H2O")


def _ch4() -> Molecule:
    d = 1.089 / math.sqrt(3) * BOHR
    return Molecule([
        Atom("C",  [0, 0, 0]),
        Atom("H",  [ d,  d,  d]),
        Atom("H",  [-d, -d,  d]),
        Atom("H",  [-d,  d, -d]),
        Atom("H",  [ d, -d, -d]),
    ], charge=0, multiplicity=1, name="CH4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"  Saved: {path.relative_to(REPO)}")


def _cprofile_text(pr: cProfile.Profile, n_top: int = 40,
                   sort_by: str = "tottime") -> str:
    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf)
    ps.sort_stats(sort_by)
    ps.print_stats(n_top)
    return buf.getvalue()


def _cprofile_top(pr: cProfile.Profile, n: int = 20) -> List[Dict]:
    """Return top-n functions as list of dicts (for JSON)."""
    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf)
    ps.sort_stats("tottime")
    ps.print_stats(n)
    rows = []
    for func, (cc, nc, tt, ct, callers) in ps.stats.items():
        rows.append({
            "function": f"{func[2]}:{func[1]}({func[0]})",
            "ncalls": nc,
            "tottime_s": round(tt, 6),
            "cumtime_s": round(ct, 6),
            "tottime_per_call_us": round(tt / nc * 1e6, 3) if nc else 0,
        })
    rows.sort(key=lambda r: r["tottime_s"], reverse=True)
    return rows[:n]


def _lineprofile_text(lp: LineProfiler) -> str:
    buf = io.StringIO()
    lp.print_stats(stream=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. cProfile — ERI construction
# ---------------------------------------------------------------------------

def section_cprofile_eri(mol_h2o: Molecule, mol_ch4: Molecule) -> Dict[str, Any]:
    print(f"\n{SEP}")
    print("  [1] cProfile: build_eri (H2O and CH4)")
    print(SEP)

    results = {}

    for mol in [mol_h2o, mol_ch4]:
        pr = cProfile.Profile()
        pr.enable()
        eri = build_eri(STO3G, mol)
        pr.disable()

        n   = len(STO3G.basis_functions(mol))
        txt = (f"cProfile: build_eri({mol.name})  n_basis={n}\n"
               + "=" * 72 + "\n"
               + _cprofile_text(pr, n_top=30, sort_by="tottime"))

        fname = f"cprofile_eri_{mol.name.lower().replace('+','p')}.txt"
        _save(PROF_DIR / fname, txt)
        top = _cprofile_top(pr, 15)
        results[mol.name] = {
            "n_basis": n,
            "n_unique_eri": int(n*(n+1)//2 * (n*(n+1)//2 + 1) // 2),
            "top_functions": top,
        }

        # Print brief summary
        for row in top[:8]:
            print(f"    {row['tottime_s']:7.3f}s  {row['ncalls']:>9,}  {row['function']}")

    return results


# ---------------------------------------------------------------------------
# 2. cProfile — full SCF
# ---------------------------------------------------------------------------

def section_cprofile_scf(mol: Molecule) -> Dict[str, Any]:
    print(f"\n{SEP}")
    print("  [2] cProfile: rhf_scf (H2O)")
    print(SEP)

    pr = cProfile.Profile()
    pr.enable()
    result = rhf_scf(mol, STO3G, verbose=False)
    pr.disable()

    n   = len(STO3G.basis_functions(mol))
    txt = (f"cProfile: rhf_scf(H2O)  n_basis={n}  iters={result.n_iter}\n"
           + "=" * 72 + "\n"
           + _cprofile_text(pr, n_top=35, sort_by="tottime"))

    _save(PROF_DIR / "cprofile_scf_h2o.txt", txt)
    top = _cprofile_top(pr, 20)

    for row in top[:10]:
        print(f"    {row['tottime_s']:7.3f}s  {row['ncalls']:>9,}  {row['function']}")

    return {"n_basis": n, "n_iter": result.n_iter, "top_functions": top}


# ---------------------------------------------------------------------------
# 3. cProfile — numerical gradient
# ---------------------------------------------------------------------------

def section_cprofile_grad(mol: Molecule) -> Dict[str, Any]:
    print(f"\n{SEP}")
    print("  [3] cProfile: numerical_gradient (H2O)")
    print(SEP)

    n_scf_calls = 2 * 3 * mol.n_atoms
    print(f"  Expected SCF calls: {n_scf_calls}")

    pr = cProfile.Profile()
    pr.enable()
    grad = numerical_gradient(mol, STO3G)
    pr.disable()

    n   = len(STO3G.basis_functions(mol))
    txt = (f"cProfile: numerical_gradient(H2O)  n_basis={n}  "
           f"n_scf_calls={n_scf_calls}\n"
           + "=" * 72 + "\n"
           + _cprofile_text(pr, n_top=30, sort_by="tottime"))

    _save(PROF_DIR / "cprofile_grad_h2o.txt", txt)
    top = _cprofile_top(pr, 15)

    for row in top[:8]:
        print(f"    {row['tottime_s']:7.3f}s  {row['ncalls']:>9,}  {row['function']}")

    return {
        "n_basis": n,
        "n_scf_calls": n_scf_calls,
        "grad_max_ha_bohr": float(np.max(np.abs(grad))),
        "top_functions": top,
    }


# ---------------------------------------------------------------------------
# 4. line_profiler — ERI hot path
# ---------------------------------------------------------------------------

def section_lineprofile_eri(mol: Molecule) -> None:
    print(f"\n{SEP}")
    print("  [4] line_profiler: eri_primitive, _contracted_eri, build_eri (H2O)")
    print(SEP)

    lp = LineProfiler()
    lp.add_function(eri_primitive)
    lp.add_function(_contracted_eri)
    lp.add_function(build_eri)

    lp_wrapper = lp(build_eri)
    lp_wrapper(STO3G, mol)

    txt = ("line_profiler: ERI hot path (H2O)\n"
           + "=" * 72 + "\n"
           + _lineprofile_text(lp))
    _save(PROF_DIR / "lineprofile_eri.txt", txt)

    print("  (saved — see profiling/lineprofile_eri.txt for line-level breakdown)")


# ---------------------------------------------------------------------------
# 5. line_profiler — Fock build
# ---------------------------------------------------------------------------

def section_lineprofile_fock(mol: Molecule) -> None:
    print(f"\n{SEP}")
    print("  [5] line_profiler: _build_fock (H2O, all SCF iterations)")
    print(SEP)

    # Instrument the SCF to capture _build_fock line times across all iters
    eri  = build_eri(STO3G, mol)
    S    = build_overlap(STO3G, mol)
    H    = build_kinetic(STO3G, mol) + build_nuclear(STO3G, mol)
    n    = S.shape[0]
    n_occ = mol.n_alpha

    from molekul.rhf import _symmetric_orthogonalizer, _diis_extrapolate
    X = _symmetric_orthogonalizer(S)
    Fp = X.T @ H @ X
    _, Cp = np.linalg.eigh(Fp)
    C = X @ Cp
    P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    lp = LineProfiler()
    lp.add_function(_build_fock)
    _build_fock_lp = lp(_build_fock)

    # Run 10 Fock builds (simulate SCF iterations)
    for _ in range(10):
        F = _build_fock_lp(H, P, eri)

    txt = ("line_profiler: _build_fock (10 calls, H2O)\n"
           + "=" * 72 + "\n"
           + _lineprofile_text(lp))
    _save(PROF_DIR / "lineprofile_fock.txt", txt)

    print("  (saved — see profiling/lineprofile_fock.txt)")


# ---------------------------------------------------------------------------
# 6. Manual sub-phase timing decomposition
# ---------------------------------------------------------------------------

def _count_calls() -> Dict[str, int]:
    """Count _E and _R calls during one build_eri on H2O by monkey-patching."""
    import molekul.integrals as _int_mod
    import molekul.eri as _eri_mod

    counts = {"E_calls": 0, "R_calls": 0, "eri_prim_calls": 0,
              "contracted_calls": 0}

    orig_E   = _int_mod._E.__wrapped__ if hasattr(_int_mod._E, "__wrapped__") else None
    orig_R   = _int_mod._R
    orig_ep  = _eri_mod.eri_primitive
    orig_ce  = _eri_mod._contracted_eri

    # Temporarily disable lru_cache on _E to count raw calls
    import functools
    _E_nocache = _int_mod._E.__wrapped__ if hasattr(_int_mod._E, "__wrapped__") else None

    # Instead of disabling cache, count contracted_eri and eri_primitive calls
    # which are NOT cached.
    def counted_ep(*args, **kwargs):
        counts["eri_prim_calls"] += 1
        return orig_ep(*args, **kwargs)

    def counted_ce(*args, **kwargs):
        counts["contracted_calls"] += 1
        return orig_ce(*args, **kwargs)

    _eri_mod.eri_primitive   = counted_ep
    _eri_mod._contracted_eri = counted_ce

    # Also patch build_eri to use local references
    mol = _h2o()
    bfs    = STO3G.basis_functions(mol)
    coords = np.array([a.coords for a in mol.atoms])
    n      = len(bfs)
    eri_out = np.zeros((n, n, n, n))

    for i in range(n):
        ai, lx1, ly1, lz1, sh1 = bfs[i]
        A = coords[ai]
        for j in range(i + 1):
            aj, lx2, ly2, lz2, sh2 = bfs[j]
            B = coords[aj]
            ij = i * (i + 1) // 2 + j
            for k in range(n):
                ak, lx3, ly3, lz3, sh3 = bfs[k]
                C = coords[ak]
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:
                        continue
                    al, lx4, ly4, lz4, sh4 = bfs[l]
                    D = coords[al]
                    counts["contracted_calls"] += 1
                    val = counted_ep(
                        lx1, ly1, lz1, A, list(sh1.exponents)[0],
                        lx2, ly2, lz2, B, list(sh2.exponents)[0],
                        lx3, ly3, lz3, C, list(sh3.exponents)[0],
                        lx4, ly4, lz4, D, list(sh4.exponents)[0],
                    )

    # Restore
    _eri_mod.eri_primitive   = orig_ep
    _eri_mod._contracted_eri = orig_ce

    # Count unique integrals analytically
    n_ij = n * (n + 1) // 2
    counts["unique_eri"]    = n_ij * (n_ij + 1) // 2
    counts["contracted_calls"] = counts["unique_eri"]       # each contracted once
    counts["eri_prim_calls"] = counts["unique_eri"] * (3**4) # 3 prims per function, 4 functions
    # eri_primitive is called once per primitive quartet inside contracted_eri
    # STO-3G: 3 primitives per contracted function → 3^4 = 81 prim per contracted

    return counts


def section_subphase_timing(mol_h2o: Molecule) -> Dict[str, Any]:
    print(f"\n{SEP}")
    print("  [6] Manual sub-phase timing decomposition (H2O)")
    print(SEP)

    n    = len(STO3G.basis_functions(mol_h2o))
    n_ij = n * (n + 1) // 2
    n_unique = n_ij * (n_ij + 1) // 2
    n_prim_per_contracted = 3**4   # STO-3G: 3 primitives × 4 functions
    n_prim_total = n_unique * n_prim_per_contracted

    result: Dict[str, Any] = {
        "n_basis": n,
        "n_unique_eri": n_unique,
        "n_primitive_calls": n_prim_total,
        "n_prim_per_contracted": n_prim_per_contracted,
    }

    # ── ERI sub-phases ────────────────────────────────────────────────────
    print(f"\n  ERI build breakdown (H2O, n_basis={n}):")
    print(f"    Unique (ij|kl) integrals : {n_unique:,}")
    print(f"    Primitive calls (3^4=81 per unique): {n_prim_total:,}")

    # Time the _E function in isolation (cached vs uncached)
    import molekul.integrals as _int_mod
    _E_fn = _int_mod._E

    # Clear cache and time a batch of E calls (representative)
    _E_fn.cache_clear()
    n_e_sample = 1000
    t0 = time.perf_counter()
    for _ in range(n_e_sample):
        _E_fn(1, 1, 0, 0.5, 0.3, 0.2)
        _E_fn(1, 1, 1, 0.5, 0.3, 0.2)
        _E_fn(0, 0, 0, 0.5, 0.3, 0.2)
    t_E_per_call_us = (time.perf_counter() - t0) / (n_e_sample * 3) * 1e6
    ci = _E_fn.cache_info()
    print(f"    _E (cold, 3 calls each × {n_e_sample}): {t_E_per_call_us:.3f} µs/call")
    print(f"    _E lru_cache info: {ci}")

    # Time _E with warm cache
    t0 = time.perf_counter()
    for _ in range(n_e_sample):
        _E_fn(1, 1, 0, 0.5, 0.3, 0.2)
        _E_fn(1, 1, 1, 0.5, 0.3, 0.2)
        _E_fn(0, 0, 0, 0.5, 0.3, 0.2)
    t_E_warm_us = (time.perf_counter() - t0) / (n_e_sample * 3) * 1e6
    print(f"    _E (warm cache):  {t_E_warm_us:.3f} µs/call")
    result["t_E_cold_us"] = round(t_E_per_call_us, 4)
    result["t_E_warm_us"] = round(t_E_warm_us, 4)

    # Time _R (Boys via hyp1f1) — no cache
    _R_fn = _int_mod._R
    n_r_sample = 1000
    PQ = np.array([0.5, 0.3, 0.1])
    t0 = time.perf_counter()
    for _ in range(n_r_sample):
        _R_fn(0, 0, 0, 0, 0.5, PQ)
        _R_fn(1, 0, 0, 0, 0.5, PQ)
        _R_fn(0, 1, 0, 0, 0.5, PQ)
    t_R_us = (time.perf_counter() - t0) / (n_r_sample * 3) * 1e6
    print(f"    _R (Boys/hyp1f1): {t_R_us:.3f} µs/call")
    result["t_R_us"] = round(t_R_us, 4)

    # Time one eri_primitive call
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 1.4])
    n_ep_sample = 200
    t0 = time.perf_counter()
    for _ in range(n_ep_sample):
        eri_primitive(0,0,0, A, 0.3, 0,0,0, B, 0.2, 0,0,0, A, 0.5, 0,0,0, B, 0.4)
    t_ep_us = (time.perf_counter() - t0) / n_ep_sample * 1e6
    print(f"    eri_primitive (s-s-s-s): {t_ep_us:.1f} µs/call")
    result["t_eri_prim_ss_us"] = round(t_ep_us, 2)

    # Time one eri_primitive call with p-functions (more _E/_R calls)
    t0 = time.perf_counter()
    for _ in range(n_ep_sample):
        eri_primitive(1,0,0, A, 0.3, 1,0,0, B, 0.2, 1,0,0, A, 0.5, 1,0,0, B, 0.4)
    t_ep_p_us = (time.perf_counter() - t0) / n_ep_sample * 1e6
    print(f"    eri_primitive (p-p-p-p): {t_ep_p_us:.1f} µs/call")
    result["t_eri_prim_pp_us"] = round(t_ep_p_us, 2)

    # Estimate total ERI time from primitives
    t_eri_est_s = n_prim_total * t_ep_us * 1e-6
    print(f"\n    Predicted ERI time from prim cost × count:")
    print(f"      {n_prim_total:,} calls × {t_ep_us:.1f} µs = {t_eri_est_s:.2f}s")
    result["t_eri_estimated_s"] = round(t_eri_est_s, 3)

    # Actual ERI time for comparison
    t0 = time.perf_counter()
    build_eri(STO3G, mol_h2o)
    t_eri_actual_s = time.perf_counter() - t0
    print(f"      Actual measured ERI time: {t_eri_actual_s:.3f}s")
    overhead_pct = (t_eri_actual_s - t_eri_est_s) / t_eri_actual_s * 100
    print(f"      Loop/Python overhead: {overhead_pct:.1f}%")
    result["t_eri_actual_s"]   = round(t_eri_actual_s, 4)
    result["eri_overhead_pct"] = round(overhead_pct, 1)

    # ── SCF sub-phases ────────────────────────────────────────────────────
    print(f"\n  SCF iteration breakdown (H2O):")

    eri = build_eri(STO3G, mol_h2o)
    S   = build_overlap(STO3G, mol_h2o)
    H   = build_kinetic(STO3G, mol_h2o) + build_nuclear(STO3G, mol_h2o)
    n_occ = mol_h2o.n_alpha

    from molekul.rhf import _symmetric_orthogonalizer, _diis_extrapolate
    X = _symmetric_orthogonalizer(S)
    Fp = X.T @ H @ X
    _, Cp = np.linalg.eigh(Fp)
    C = X @ Cp
    P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    n_rep = 20
    timings: Dict[str, float] = {
        "fock_build": 0.0, "energy_einsum": 0.0,
        "diis_error": 0.0, "diis_extrap": 0.0,
        "diagonalise": 0.0, "density_update": 0.0,
    }

    fock_list, err_list = [], []
    for it in range(n_rep):
        t0 = time.perf_counter();  F = _build_fock(H, P, eri)
        timings["fock_build"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = 0.5 * np.einsum("mn,mn->", P, H + F)
        timings["energy_einsum"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        err = F @ P @ S - S @ P @ F
        timings["diis_error"] += time.perf_counter() - t0

        fock_list.append(F.copy()); err_list.append(err.copy())
        if len(fock_list) > 8: fock_list.pop(0); err_list.pop(0)

        t0 = time.perf_counter()
        F_d = _diis_extrapolate(fock_list, err_list)
        timings["diis_extrap"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        Fp2 = X.T @ F_d @ X
        _, Cp2 = np.linalg.eigh(Fp2)
        C2 = X @ Cp2
        timings["diagonalise"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        P = 2.0 * C2[:, :n_occ] @ C2[:, :n_occ].T
        timings["density_update"] += time.perf_counter() - t0

    total_iter_t = sum(timings.values())
    print(f"  {'Phase':<20} {'time/iter (ms)':>16}  {'%':>6}")
    print(f"  {'─'*20}  {'─'*16}  {'─'*6}")
    for k, v in timings.items():
        per_iter_ms = v / n_rep * 1000
        pct = v / total_iter_t * 100
        print(f"  {k:<20}  {per_iter_ms:>14.3f}ms  {pct:>5.1f}%")

    result["scf_iter_timings_ms"] = {
        k: round(v / n_rep * 1000, 4) for k, v in timings.items()
    }

    # ── Gradient sub-phases ───────────────────────────────────────────────
    print(f"\n  Gradient breakdown (H2O, finite diff):")
    n_displacements = 2 * 3 * mol_h2o.n_atoms
    t0 = time.perf_counter()
    build_eri(STO3G, mol_h2o)
    t_eri_single = time.perf_counter() - t0

    t0 = time.perf_counter()
    rhf_scf(mol_h2o, STO3G, verbose=False)
    t_scf_single = time.perf_counter() - t0

    # Fraction of gradient that is ERI
    t_grad_eri_est  = n_displacements * t_eri_single
    t_grad_scf_est  = n_displacements * t_scf_single
    print(f"  Displacements (2·3·N): {n_displacements}")
    print(f"  Per-displacement ERI  : {t_eri_single*1000:.1f}ms")
    print(f"  Per-displacement SCF  : {t_scf_single*1000:.1f}ms")
    print(f"  Est. gradient ERI total: {t_grad_eri_est:.2f}s")
    print(f"  Est. gradient total    : {t_grad_scf_est:.2f}s")
    print(f"  ERI fraction of grad   : {t_eri_single/t_scf_single*100:.1f}%")

    result["grad_n_displacements"] = n_displacements
    result["grad_t_eri_per_disp_ms"] = round(t_eri_single * 1000, 2)
    result["grad_t_scf_per_disp_ms"] = round(t_scf_single * 1000, 2)
    result["grad_eri_fraction_pct"]  = round(t_eri_single / t_scf_single * 100, 1)

    # Build and save sub-phase timing report
    lines = [
        "MOLEKUL Phase 9 — Sub-phase timing decomposition",
        f"Molecule: H2O,  n_basis={n},  STO-3G",
        "",
        "ERI PRIMITIVES",
        f"  Unique (ij|kl) integrals : {n_unique:,}",
        f"  Primitive quartets (×81) : {n_prim_total:,}",
        f"  _E call cost (cold)       : {t_E_per_call_us:.3f} µs",
        f"  _E call cost (warm cache) : {t_E_warm_us:.3f} µs",
        f"  _R call cost (Boys fn)    : {t_R_us:.3f} µs",
        f"  eri_primitive (s-s-s-s)  : {t_ep_us:.1f} µs",
        f"  eri_primitive (p-p-p-p)  : {t_ep_p_us:.1f} µs",
        f"  Predicted ERI from prim  : {t_eri_est_s:.3f}s",
        f"  Actual ERI time           : {t_eri_actual_s:.3f}s",
        f"  Python loop overhead      : {overhead_pct:.1f}%",
        "",
        "SCF ITERATION (averaged over 20 reps)",
    ]
    for k, v in timings.items():
        per_iter_ms = v / n_rep * 1000
        pct = v / total_iter_t * 100
        lines.append(f"  {k:<20}  {per_iter_ms:>10.3f}ms  {pct:>5.1f}%")
    lines += [
        "",
        "GRADIENT (finite differences)",
        f"  Displacements        : {n_displacements}",
        f"  ERI per displacement : {t_eri_single*1000:.1f}ms",
        f"  SCF per displacement : {t_scf_single*1000:.1f}ms",
        f"  ERI fraction of grad : {t_eri_single/t_scf_single*100:.1f}%",
    ]
    _save(PROF_DIR / "subphase_timing.txt", "\n".join(lines))

    return result


# ---------------------------------------------------------------------------
# 7. lru_cache analysis for _E
# ---------------------------------------------------------------------------

def section_cache_analysis(mol: Molecule) -> Dict[str, Any]:
    print(f"\n{SEP}")
    print("  [7] lru_cache analysis for _E (McMurchie-Davidson coefficients)")
    print(SEP)

    import molekul.integrals as _int_mod
    _E_fn = _int_mod._E

    _E_fn.cache_clear()
    build_eri(STO3G, mol)
    ci = _E_fn.cache_info()
    hit_rate = ci.hits / (ci.hits + ci.misses) * 100 if (ci.hits + ci.misses) > 0 else 0

    print(f"  After build_eri(H2O):")
    print(f"    Cache hits   : {ci.hits:,}")
    print(f"    Cache misses : {ci.misses:,}")
    print(f"    Cache size   : {ci.currsize:,}")
    print(f"    Hit rate     : {hit_rate:.1f}%")
    print(f"  → {hit_rate:.0f}% of _E lookups saved by caching.")

    return {
        "cache_hits":   ci.hits,
        "cache_misses": ci.misses,
        "cache_size":   ci.currsize,
        "hit_rate_pct": round(hit_rate, 1),
    }


# ---------------------------------------------------------------------------
# JSON logger (minimal, no external dependency on ExperimentLogger)
# ---------------------------------------------------------------------------

def _save_json(data: Dict[str, Any]) -> Path:
    import datetime, subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sha = None

    doc = {
        "timestamp": datetime.datetime.now().isoformat(),
        "status":    "PASS",
        "git_sha":   sha,
        "elapsed_s": data.pop("elapsed_s", None),
        "n_passed":  0,
        "n_failed":  0,
        "metrics":   data,   # status.py reads log["metrics"]
    }

    path = LOGS_DIR / "phase9_profile.json"
    path.write_text(json.dumps(doc, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall = time.perf_counter()

    print("=" * 72)
    print("  Phase 9: Detailed CPU Profiling — ERI & Gradient Bottlenecks")
    print("  MOLEKUL / RHF / STO-3G")
    print("=" * 72)
    print(f"  Output dir: {PROF_DIR.relative_to(REPO)}/")

    mol_h2o = _h2o()
    mol_ch4 = _ch4()

    summary: Dict[str, Any] = {}

    summary["cprofile_eri"]   = section_cprofile_eri(mol_h2o, mol_ch4)
    summary["cprofile_scf"]   = section_cprofile_scf(mol_h2o)
    summary["cprofile_grad"]  = section_cprofile_grad(mol_h2o)
    section_lineprofile_eri(mol_h2o)
    section_lineprofile_fock(mol_h2o)
    summary["subphase"]       = section_subphase_timing(mol_h2o)
    summary["cache_analysis"] = section_cache_analysis(mol_h2o)

    elapsed = time.perf_counter() - t_wall
    summary["elapsed_s"] = round(elapsed, 2)

    json_path = _save_json(summary)

    print(f"\n{SEP}")
    print("  FILES WRITTEN")
    print(SEP)
    for f in sorted(PROF_DIR.iterdir()):
        if f.is_file():
            print(f"    profiling/{f.name}  ({f.stat().st_size // 1024 + 1} KB)")
    print(f"    outputs/logs/phase9_profile.json")

    print(f"\n  Total profiling time: {elapsed:.1f}s")
    print("\n  Done.")


if __name__ == "__main__":
    main()
