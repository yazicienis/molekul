# Prompt: Audit Cleanup (code/repo-side)

**Origin:** 2026-05-29 full-repo audit (Opus supervisor). Core science is sound
(RHF/MP2, gradient expression, PySCF validations all correct). These are
metadata / documentation / hygiene fixes only — no algorithm changes.

**Scheduling:** QUEUED to run AFTER Phase 19 GPU-backend review is accepted.
Do not start until Phase 19 is closed in HANDOFF.md.

**Scope rule:** This is cleanup, not a phase. Touch only what is listed. Run the
full test suite afterward; nothing here should change any numerical result.

---

## Tasks (code/repo-side — land on `main` now)

### 1. Version string is inconsistent (and lower than the released archive)
- `src/molekul/__init__.py` has `__version__ = "0.1.0"`.
- Everything else (paper, README, CITATION.cff, Zenodo) says **v0.1.2**, and
  HEAD is actually well beyond v0.1.2 (Phases 11–19 added since the archive).
- Fix: set a single coherent value. Recommended: `__version__ = "0.2.0.dev0"`
  for HEAD (clearly post-v0.1.2 dev). If supervisor prefers the released tag,
  use `"0.1.2"` — but never leave it at `0.1.0` (lower than what shipped).
- Confirm no other file hardcodes a conflicting version (`grep -rn "0\.1\.0"`).

### 2. README quick-start energy is stale / contradicts the paper
- README quick-start comment shows H2O RHF `-74.96258854 Eh`.
- The paper example and HANDOFF (`-74.96294667` / `-74.962946665868`) and
  `benchmark_14mol.json` disagree — README's number looks outdated.
- Fix: recompute the exact geometry in the README snippet and replace the
  comment with the verified value. Make README, paper, and log mutually
  consistent for the same geometry.

### 3. "Analytic gradient" label is overstated
- `STATUS.md` Phase 18 row says "Analytic RHF gradient ✅". The actual code
  (`grad.py`) takes integral derivatives by **finite difference**, not true
  recurrence-based analytic integral derivatives. The code docstring and
  SCIENCE.md are already honest; only the headline label overstates.
- Fix: relabel as "Semi-numerical RHF gradient (analytic energy expression +
  FD integral derivatives)" in STATUS.md (and anywhere else that says
  "analytic"). Do NOT let "analytic gradient" leak into any future paper text.

### 4. Repo hygiene — submission zips and LaTeX build artifacts
- Root has 6 `softwarex_submission*.zip` (v3/v4/v5/v6 + 2 unversioned) plus
  LaTeX build artifacts (`.aux`, `.log`, `.bbl`, `.blg`, `.out`, empty `.spl`).
- Ambiguous which zip is the actual submission → archival risk.
- Fix: (a) keep only the canonical submitted zip (confirm with supervisor which
  one), move the rest to an untracked `archive/` or delete; (b) add LaTeX build
  artifacts and `softwarex_submission*.zip` to `.gitignore`. Do not delete the
  `.tex`, `.bib`, `.pdf`, or `pes_n2.*` figure sources.

## Paper-side (.tex) — NOT in this task
Findings touching `softwarex_paper.tex` (benchmark number 4.9e-8→5.8e-8/HF,
`harmonic.py`→`freqs.py`, SciPy in metadata) stay in
`paper_corrections_pending.txt` and are applied in the journal **revision
round**, not pushed to the editor unilaterally. Supervisor decides timing.

## Done criteria
- `pytest tests/ -x` still green (no numerical change expected).
- README / paper / `benchmark_14mol.json` agree on the H2O RHF energy.
- One coherent version string repo-wide.
- STATUS.md no longer claims a fully "analytic" gradient.
- Clean repo root; updated `.gitignore`.
- Update HANDOFF.md / CHANGELOG_AGENT.md and hand back to supervisor.
