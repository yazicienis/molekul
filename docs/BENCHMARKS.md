# MOLEKUL Phase 8 — CPU Performance Benchmarks

**Date:** 2026-04-02  
**Basis:** STO-3G  
**Method:** RHF  
**Hardware:** Single CPU thread, pure Python (no GPU, no OpenMP, no MKL)

---

## Results

| Molecule | n_basis | n_elec | 1e-integrals | ERI     | SCF     | SCF iter | Gradient | Opt     | ERI mem |
|----------|---------|--------|--------------|---------|---------|----------|----------|---------|---------|
| H2       | 2       | 2      | 0.78 ms      | 3.3 ms  | 3.9 ms  | 2        | 44 ms    | 247 ms  | <0.01 MB |
| HeH+     | 2       | 2      | 0.64 ms      | 3.1 ms  | 4.2 ms  | 12       | 53 ms    | 350 ms  | <0.01 MB |
| H2O      | 7       | 10     | 10.5 ms      | 364 ms  | 372 ms  | 9        | 6.8 s    | 21.6 s  | 0.02 MB  |
| NH3      | 8       | 10     | 14.0 ms      | 593 ms  | 602 ms  | 9        | 14.4 s   | —       | 0.03 MB  |
| CH4      | 9       | 10     | 19.6 ms      | 884 ms  | 903 ms  | 10       | 27.2 s   | —       | 0.05 MB  |
| C6H6     | 36      | 42     | 698 ms       | 4.2 min | 4.2 min | 10       | skipped  | —       | 13.4 MB  |

**Gradient** skipped for n_basis > 15 (requires 6·N_atoms SCF calls via finite differences).  
**Opt** measured only for H2, HeH+, H2O.  
**Benzene** gradient would require ~72 SCF calls × ~25 s each ≈ 30 minutes — not run.

---

## Scaling

Empirical log-log fit against n_basis:

| Phase    | Empirical exponent | Theoretical |
|----------|--------------------|-------------|
| ERI      | n^3.88             | O(n^4)      |
| SCF      | n^3.78             | O(n^4)      |
| Gradient | n^4.12             | O(6·N · n^4) |

The slight deviation from n^4 is expected: the two-atom molecules (n=2) carry fixed overhead that depresses the low-end slope; the benzene point pulls the fit back toward 4.

### ERI time by molecule (log scale)

```
n=2  │ ▏ 3 ms
n=7  │ ████████████ 364 ms
n=8  │ ████████████████████ 593 ms
n=9  │ ████████████████████████████████ 884 ms
n=36 │ ██████████████████████████████████████████████ 251 s
```

---

## Bottlenecks

1. **ERI is O(n⁴)** — the dominant cost for all molecules with n_basis ≥ 7.  
   Implemented as pure Python nested loops with no screening, no symmetry
   reuse beyond 8-fold permutation symmetry, no vectorisation.

2. **Gradient is O(6·N_atoms · n⁴)** — finite differences require two full
   ERI builds per displaced coordinate. For CH4 (5 atoms, 9 basis functions)
   this means 30 ERI builds × 0.9 s ≈ 27 s.

3. **No integral screening** — all (μν|λσ) pairs computed regardless of
   magnitude. Cauchy-Schwarz would eliminate many near-zero integrals,
   especially for diffuse functions or distant atom pairs.

4. **ERI stored as dense n⁴ tensor** — adequate for small molecules but
   grows rapidly. Benzene: 36⁴ × 8 bytes ≈ 13 MB. For n=100 this would
   be 800 MB (barely feasible); for n=200, 128 GB (infeasible).

5. **SCF Fock build is fast once ERI is stored** — the per-iteration cost
   (matrix multiply + Fock update) is negligible compared to the one-time
   ERI construction. DIIS convergence (typically 9–12 iterations) does not
   contribute meaningfully to total time.

6. **One-electron integrals are cheap** — even for benzene (n=36) these
   take < 700 ms. Not a bottleneck.

---

## Key Observations

- **H2O is the practical boundary** for interactive use: full ERI + SCF in
  ~370 ms. NH3 and CH4 are one SCF cycle away from crossing 1 s.
- **Benzene is the stress test**: 4.2 minutes for ERI alone. Any molecule
  above ~20 basis functions requires either screening, vectorisation, or
  GPU offload to be practical.
- **HeH+ takes 12 SCF iterations** despite having only 2 basis functions —
  DIIS convergence is slower for charged open-ish systems near the minimal
  basis limit.
- **H2O gradient (6.8 s)** is dominated by 18 × 370 ms SCF calls. An
  analytic gradient implementation would reduce this to a single-pass cost
  comparable to one SCF cycle.

---

## Next Steps (not yet implemented)

| Priority | Action | Expected speedup |
|----------|--------|-----------------|
| High | Vectorise ERI inner loop with NumPy | 10–50× |
| High | Cauchy-Schwarz integral screening | 2–10× (larger basis) |
| Medium | Analytic gradient | ~18× for H2O gradient |
| Medium | Exploit 8-fold ERI permutation symmetry in storage | 8× memory |
| Low | GPU offload (CuPy or custom CUDA) | 100–1000× for ERI |
| Low | Density fitting / RI approximation | n^3 vs n^4 |

These are Phase 9+ goals. Phase 8 scope is measurement only.

---

## Reproducing

```bash
# Full benchmark (includes benzene, ~10 min)
python scripts/benchmark.py

# Skip benzene ERI/SCF
python scripts/benchmark.py --skip-large
```

Raw data: `outputs/logs/phase8_benchmark.json`, `outputs/phase8_benchmark.txt`
