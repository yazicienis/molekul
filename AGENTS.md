# AGENTS.md — instructions for Codex on MOLEKUL

You (Codex) are the **IMPLEMENTER**. Claude Code is the supervisor/auditor who reviews
your output and independently re-checks every number. Write code and prose; do not
decide scope — that is set in `prompts/jose_resubmission.md`.

## Project
MOLEKUL is a pure-Python (NumPy-core) ab initio quantum chemistry **teaching** code
with a 14-notebook curriculum (`notebooks/` 00–13). It is **not** a production
alternative to PySCF/Psi4 — never write that it is.

## Current task source
The SoftwareX paper was rejected; we are repositioning for **JOSE**. Your tasks are the
`→ CODEX` blocks in **`prompts/jose_resubmission.md`** (① validation table, ③ JOSE
package, ⑤ finalize). Do them in order; stop at each gate for supervisor review.
Known-good anchor numbers: **`benchmarks/seed_numbers.json`**.

## Standing rules (do not violate)
1. Frame every method as pedagogical coverage **backed by a validation number**, never
   as a capability boast.
2. Freeze scope: add no new methods, optimize no performance.
3. No half-validated claims. If a method/basis/element cell fails or is unsupported,
   mark it **experimental** or **unsupported** with a reason — never fake or silently
   drop it. PBE / PBE-TDDFT are experimental (~1e-3 Ha grid error).
4. Invent no results. Cite only numbers your own committed scripts produce.
5. Before writing any claim about a competitor (eChem, Psi4NumPy, DQC, pyfock,
   SlowQuant, PyQuante2), verify it — the SoftwareX rejection faulted exactly this.
6. Write in plain, direct prose. Avoid "delve / showcase / leverage", hedging, and
   three-item padding.

## Envelope (keep claims honest)
Atom objects: H–Ar. STO-3G: H–Ne. 6-31G\* & cc-pVDZ: H, He, C, N, O, F. Masses
(frequencies/phonons): H–Ne. MOLEKUL uses 6 Cartesian d-functions (cc-pVDZ H₂O = 25,
N₂ = 30).

## Environment & commands
Single shared conda env (`base`). Already installed: `pip install -e ".[notebooks]"`.
```bash
pytest tests/                                   # test suite
python benchmarks/validation_table.py           # (you create this) regenerates tables
jupyter nbconvert --to notebook --execute --inplace notebooks/NN_*.ipynb
```
Notebooks are gitignored (working files). Commit code, scripts, paper.md, and the
benchmark tables; do not commit notebook outputs unless asked.
