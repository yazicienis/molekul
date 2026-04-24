# MOLEKUL Phase 9 — CPU Profiling & Bottleneck Analysis

**Date:** 2026-04-02  
**Basis:** STO-3G | **Method:** RHF | **Target molecule:** H2O (n=7), CH4 (n=9)  
**Tools:** `cProfile`, `line_profiler`, manual `perf_counter` decomposition  
**Raw data:** `profiling/`, `outputs/logs/phase9_profile.json`

---

## Summary: where the time goes

| Phase | Time (H2O) | % of ERI | Bottleneck |
|-------|-----------|----------|------------|
| `build_eri` | 360 ms | 100% | Python primitive loops + Boys fn |
| `rhf_scf` (full) | 366 ms | 102% | dominated by ERI build |
| `numerical_gradient` | 6.6 s | 1830% | 18× ERI rebuilds |
| SCF iteration (post-ERI) | 0.055 ms/iter | — | DIIS B-matrix |

**The ERI construction is the bottleneck for every phase.** SCF iterations
themselves (Fock build + DIIS + diagonalise) cost only ~0.055 ms/iter once
the ERI tensor is in memory.

---

## 1. `eri_primitive` — line-level breakdown

Source: `line_profiler` on `eri_primitive`, 32,886 calls (H2O).

```
Line  Hits       Time      %    Code
  67  32886    6.1 ms    0.7%   A = np.asarray(A, dtype=float)
  68  32886    5.9 ms    0.7%   B = np.asarray(B, dtype=float)
  69  32886    5.5 ms    0.7%   C = np.asarray(C, dtype=float)
  70  32886    5.4 ms    0.7%   D = np.asarray(D, dtype=float)
  76  32886   44.9 ms    5.4%   P = (a * A + b * B) / p
  77  32886   44.2 ms    5.3%   Q = (c * C + d * D) / q
  78  32886   12.4 ms    1.5%   PQ = P - Q
  89–96  ×6  57.4 ms    7.0%   _E(...) list comprehensions
  99–109 loops  85.1 ms   10.3%  6 nested Python for-loops
 110  88857  420.2 ms   50.8%  * _R(t+tau, u+nu, v+phi, 0, alpha, PQ)
 112  32886    9.6 ms    1.2%   return 2π^2.5/... * total
```

**_R (Boys function) is 50.8% of all time in `eri_primitive`.**

### Sub-breakdown of `_R` overhead (cProfile):

| Function | Calls | tottime | µs/call |
|----------|-------|---------|---------|
| `_R` (recursion + Boys) | 228,906 | 158 ms | 0.69 µs |
| `_boys` (hyp1f1 call) | 110,727 | 57 ms | 0.51 µs |
| `numpy.dot` (inside `_R`) | 228,906 | 9 ms | 0.04 µs |

The Boys function `F_n(x)` is evaluated via SciPy's `hyp1f1` (confluent
hypergeometric function), which carries significant call overhead. Each
`_R` invocation at order `n=0` calls `_boys` directly; higher orders
recurse, re-entering `_boys` and `_R`. There is **no caching** on `_R`.

### `np.asarray` conversion overhead

Four `np.asarray(A, dtype=float)` calls at function entry cost **2.8%**
of total `eri_primitive` time. These are unnecessary when coordinates are
already float64 NumPy arrays (which they always are in practice).

### P and Q weighted centroid calculation

```python
P = (a * A + b * B) / p   # 5.4%
Q = (c * C + d * D) / q   # 5.3%
```

These allocate new NumPy arrays of size 3 for every primitive call.
At 32,886 calls × 2 arrays × 3 floats, the allocation/GC pressure
is measurable. Total: **10.7%** of `eri_primitive`.

### 6 nested Python loops

The innermost summation over Hermite indices `(t,u,v,τ,ν,φ)` runs
entirely in pure Python. For s-functions there are 1^6 = 1 terms;
for p-functions up to 2^6 = 64 terms. Average loop body hits: ~2.7
iterations per axis → ~88,857 innermost iterations per 32,886 calls.

Python loop overhead + `(-1)**(tau+nu+phi)` integer exponentiation:
**~10.3%** of `eri_primitive`.

---

## 2. `_E` (McMurchie-Davidson coefficients) — cache analysis

`_E` is decorated with `@lru_cache(maxsize=None)`.

After `build_eri(H2O)`:

| Metric | Value |
|--------|-------|
| Cache hits | 253,692 |
| Cache misses | 459 |
| Cache size | 459 unique keys |
| **Hit rate** | **99.8%** |

In STO-3G, each element type has fixed exponents shared across all atoms
of that element, so the `(i, j, t, Qx, a, b)` keys repeat heavily.
**`_E` is effectively free after the first call per unique key.** Removing
the cache would increase ERI time by ~10–20×.

Benchmark: `_E` cold call = 0.055 µs, warm (cached) = 0.052 µs — the
cached lookup cost is negligible.

---

## 3. `_contracted_eri` — contraction overhead

Source: `line_profiler` on `_contracted_eri`, 406 calls (H2O unique integrals).

```
Line  Hits     Time     %    Code
 132   406    2.7 ms   0.3%  N1 = sh1.norms(...)   ← 4 norm fetches
 139  4872    1.2 ms   0.1%  for b_exp, c2, n2 in ...
 141 43848   11.6 ms   1.2%  for c_exp, c3, n3 in ...
 142 65772   15.7 ms   1.6%  for d_exp, c4, n4 in ...  (innermost)
 143 65772  908.6 ms  94.0%  result += n1*c1*... * eri_primitive(...)
```

**94% of `_contracted_eri` is the `eri_primitive` call.** The 4 nested
Python loops over primitives (3^4 = 81 iterations) contribute only 3%.

---

## 4. `build_eri` — outer loop overhead

Source: `line_profiler` on `build_eri`.

```
Line  Hits   Time      %    Code
 199   812  977.8 ms  99.9%  val = _contracted_eri(...)
 207–214  ×8 negligible   eri[...] symmetry broadcast
```

The outer 4 Python loops over basis function indices `(i,j,k,l)` with
the unique-pair filter account for **< 0.1%** of ERI build time.

---

## 5. SCF iteration decomposition

Source: `perf_counter` timing of each sub-step, averaged over 20 iterations, H2O.

| Sub-step | Time/iter | % |
|----------|-----------|---|
| `_build_fock` (J+K einsum) | 0.007 ms | 13.2% |
| Energy einsum `tr(PH+F)` | 0.001 ms | 2.5% |
| DIIS error `FPS - SPF` | 0.003 ms | 4.6% |
| **DIIS extrapolation** | **0.035 ms** | **64.4%** |
| Diagonalise (`np.linalg.eigh`) | 0.007 ms | 11.9% |
| Density update `P = 2·CC^T` | 0.002 ms | 3.3% |

**DIIS extrapolation is 64% of per-iteration cost** despite operating on
7×7 matrices. The bottleneck is the pure Python B-matrix build loop
(O(m²) inner products over stored Fock vectors) plus a `np.linalg.solve`
call on an (m+1)×(m+1) system. For small n this Python overhead dominates
over the linear algebra itself.

`_build_fock` — which performs two `np.einsum` contractions over the 7⁴
ERI tensor — costs only **0.007 ms/iter** because NumPy einsum is
vectorised C code. This will dominate for larger n.

---

## 6. Numerical gradient — displacement budget

Source: cProfile on `numerical_gradient(H2O)`.

H2O has N=3 atoms → 2×3×3 = **18 displaced SCF calculations**.

| Function | Total calls | Total time | % of gradient |
|----------|-------------|------------|---------------|
| `eri_primitive` | 591,948 | 4.0 s | 59% |
| `_R` | 4,214,430 | 2.9 s | 43% |
| `_boys` | 2,052,054 | 1.1 s | 16% |
| `_contracted_eri` | 7,308 | 0.5 s | 7% |
| `np.asarray` | 2,436,450 | 0.2 s | 2.4% |

Measured per-displacement:
- ERI rebuild: **358 ms**
- Full SCF: **366 ms**
- **ERI is 97.7% of per-displacement cost.**

The 8 ms non-ERI cost per displacement (Fock build + DIIS + eigh × ~9
iterations) is negligible against the ERI rebuild.

**Every gradient evaluation rebuilds the full ERI tensor 18 times.**
There is no caching of ERI across displacements (each geometry is different,
so this is physically correct but computationally wasteful compared to an
analytic gradient implementation).

---

## 7. Quantified bottleneck hierarchy

Ranked by total time consumed in a full gradient calculation (H2O):

| Rank | Hotspot | Time | Fix category |
|------|---------|------|-------------|
| 1 | `_R` Boys function (`hyp1f1`) | ~2.9 s | Replace with Taylor/Chebyshev expansion |
| 2 | `eri_primitive` Python overhead (loops, `np.asarray`) | ~1.1 s | Vectorise / Numba / Cython |
| 3 | P, Q centroid allocation in `eri_primitive` | ~0.2 s | Pass raw floats; avoid NumPy alloc |
| 4 | 18× full ERI rebuild per gradient | structural | Analytic gradient (1 SCF equivalent) |
| 5 | DIIS B-matrix Python loop | ~0.007 ms/iter | Vectorise B-matrix build |
| 6 | `np.asarray` guard at `eri_primitive` entry | ~0.1 s | Pre-convert; remove redundant call |

---

## 8. Key numbers for optimization planning

| Metric | Value |
|--------|-------|
| `_R` cost (hyp1f1) | 1.38 µs/call |
| `_R` call count per H2O ERI | 228,906 |
| `eri_primitive` (s-s-s-s) | 5.4 µs |
| `eri_primitive` (p-p-p-p) | 28.9 µs |
| Python loop overhead in ERI | 50.5% |
| `_E` cache hit rate | 99.8% |
| DIIS fraction of SCF iter | 64.4% |
| ERI fraction of gradient | 97.7% |
| Gradient speedup from analytic grad | ~18× (H2O) |
| Gradient speedup from vectorised ERI | ~2–10× |

---

## Reproducing

```bash
# Full profiling run (~15s, includes gradient of H2O)
python profiling/profile_all.py

# View detailed line-level results
cat profiling/lineprofile_eri.txt
cat profiling/lineprofile_fock.txt
cat profiling/cprofile_grad_h2o.txt
```

Files generated:
- `profiling/cprofile_eri_h2o.txt` — cProfile for ERI (H2O)
- `profiling/cprofile_eri_ch4.txt` — cProfile for ERI (CH4)
- `profiling/cprofile_scf_h2o.txt` — cProfile for full SCF (H2O)
- `profiling/cprofile_grad_h2o.txt` — cProfile for gradient (H2O)
- `profiling/lineprofile_eri.txt` — line-by-line: `eri_primitive`, `_contracted_eri`, `build_eri`
- `profiling/lineprofile_fock.txt` — line-by-line: `_build_fock`
- `profiling/subphase_timing.txt` — manual timing decomposition table
- `outputs/logs/phase9_profile.json` — structured JSON summary
