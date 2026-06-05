# MOLEKUL Teaching Notebooks

A guided, A-to-Z tour of a working ab initio quantum chemistry code — from a single
atom to band structures and phonons. Every notebook pairs the **theory** (equations and
standard references) with the **actual MOLEKUL code** that implements it, and validates
the result against PySCF where possible.

**Audience:** advanced undergraduates and MSc/PhD students who want to see how a real
electronic-structure program is built, not just how to drive one.

---

## 1. Setup

From the repository root:

```bash
# 1. create/activate an environment (Python >= 3.10), then:
pip install -e ".[notebooks]"     # installs molekul + matplotlib + pyscf + py3Dmol + jupyter

# 2. launch
jupyter lab            # or: jupyter notebook
```

Every notebook begins with a small bootstrap cell that makes `molekul` importable even
if you skipped `pip install` — it locates the in-repo `src/` automatically. So the
notebooks also run straight from a fresh clone.

**Optional pieces** (the notebooks degrade gracefully if these are missing):
- `pyscf` — only the "compare against a reference code" cells use it.
- `py3Dmol` — only the inline 3D orbital/molecule viewer in notebook 06 uses it.

A 30-second environment self-check lives in **`00_index.ipynb`** — run it first.

## 2. Prerequisites

| You should be comfortable with | Used for |
|--------------------------------|----------|
| Linear algebra (eigenproblems, matrices) | Roothaan-Hall, every SCF |
| Basic quantum mechanics (operators, expectation values) | all theory sections |
| Python + NumPy (arrays, `einsum`, slicing) | reading the code cells |
| (Helpful) a first course in physical chemistry | chemical interpretation |

No prior quantum-chemistry course is assumed — concepts are built up from notebook 01.

## 3. Learning path

Work top to bottom. After the core (01–04) the branches are fairly independent, so you
can follow your interest. Runtimes are rough estimates on a typical laptop, CPU-only.

| # | Notebook | Phase(s) | What you build | Depends on | ~Time | Level |
|---|----------|----------|----------------|------------|-------|-------|
| 00 | `index` | — | environment check + this roadmap | — | <1 s | ⭐ |
| 01 | `atoms_molecules_constants` | 1 | `Atom`, `Molecule`, units, nuclear repulsion | — | ~10 s | ⭐ |
| 02 | `one_electron_integrals` | 2 | S, T, V matrices; Boys; McMurchie-Davidson | 01 | ~30 s | ⭐⭐ |
| 03 | `basis_sets` | 3, 9 | STO-3G / 6-31G\* / cc-pVDZ | 01, 02 | ~1 min | ⭐⭐ |
| 04 | `hartree_fock_scf` | 4 | RHF, DIIS, SAD guess, energy components | 02, 03 | ~30 s | ⭐⭐⭐ |
| 05 | `geometry_optimization` | 5 | PES, gradients, BFGS optimiser | 04 | ~2 min | ⭐⭐ |
| 06 | `visualization_population` | 6, 7 | cube files, Mulliken charges, dipole | 04 | ~1 min | ⭐⭐ |
| 07 | `mp2_harmonic_frequencies` | 8, 10 | MP2 correlation, Hessian, IR spectrum | 04 | ~2–3 min | ⭐⭐⭐ |
| 08 | `coupled_cluster` | 11, 14 | CCSD and CCSD(T) | 04, 07 | ~1–2 min | ⭐⭐⭐⭐ |
| 09 | `density_functional_theory` | 12 | KS-DFT, LDA/PBE, Becke grids | 04 | ~1–2 min | ⭐⭐⭐ |
| 10 | `excited_states` | 13, 15–17 | UHF, CIS, TD-DFT, EOM-CCSD | 04, 08, 09 | ~2–3 min | ⭐⭐⭐⭐ |
| 11 | `gradients_gpu` | 18, 19 | semi-numerical gradient, CuPy backend | 04, 05 | ~1 min | ⭐⭐⭐ |
| 12 | `periodic_systems_phonons` | 20–23 | Bloch sums, bands, DOS, phonons | 04 | ~2–3 min | ⭐⭐⭐⭐ |
| 13 | `full_workflow` (capstone) | all | one molecule, end to end | 04–10 | ~3–5 min | ⭐⭐⭐ |

```
        01 ── 02 ── 03 ── 04 ─┬─ 05 ─── 11
                              ├─ 06
                              ├─ 07 ─── 08 ─┐
                              ├─ 09 ────────┼─ 10
                              ├─ 12         │
                              └─────────────┴─ 13 (capstone)
```

## 4. How to read a notebook

Each notebook follows the same rhythm:

1. **What you will learn** — the goals up front.
2. **Theory** — equations with citations (Szabo-Ostlund, Helgaker, Pulay, Stanton, …).
3. **Code** — the real `molekul` functions, run on small molecules (H₂, H₂O, N₂, …).
4. **Validation** — comparison to PySCF or experiment, with honest error sizes.
5. **Exercises** — try these; expected answers are collected in `solutions.md`.
6. **Summary** — a one-table recap.

## 5. Scope and honesty

MOLEKUL is a *teaching* code: clarity over speed. Concretely:
- Dense N⁴ integral storage → practical limit ≈ 50 basis functions.
- Elements: calculations are limited by basis coverage (STO-3G: H–Ne; 6-31G\*/cc-pVDZ:
  H, He, C, N, O, F). See notebook 01.
- A few methods are explicitly **experimental** (PBE / PBE TD-DFT on grids); the
  notebooks flag these where they appear. Trust LDA, CCSD, RHF, MP2 quantitatively.

For production work you would reach for PySCF, Psi4, ORCA, or VASP — the capstone
notebook closes with a short "where to go next" pointer.
