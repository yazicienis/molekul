# CLAUDE.md — working context for MOLEKUL

Auto-loaded every session. Keep it short; it costs context on every turn.

## What this project is
MOLEKUL is a pure-Python (NumPy-core) ab initio quantum chemistry **teaching** code:
constants → integrals → SCF → MP2/CC → DFT → excited states → gradients →
periodic/phonons, plus a 14-notebook curriculum in `notebooks/` (00–13). It is **not** a
production alternative to PySCF/Psi4 — never pitch it as one.

## Current focus — JOSE (DECIDED, do not reopen)
The SoftwareX submission was **rejected** (niche already served by PySCF, Psi4NumPy,
eChem, SlowQuant, DQC, pyfock). On 2026-06-04 the user **considered and rejected**
the IF-bearing alternatives (JCE, AJP/EJP, and Elsevier/Wiley/Springer education
journals — Education for Chem. Engineers, CAEE, Heliyon, J. Sci. Educ. & Tech.) and
**confirmed the JOSE path** (Journal of Open Source Education). Do not re-litigate the
venue unless the user reopens it.

Plan + sequenced agent prompts: **`prompts/jose_resubmission.md`**.
Verified anchor numbers: **`benchmarks/seed_numbers.json`**.

**M1 PASSED (2026-06-04).** Validation tables committed. 23 cells ok, 7 experimental (DFT/TD-DFT non-STO-3G). cc-pVDZ DFT is broken (|ΔE|≈1.5 Ha, labeled in table). CCSD(T) tested only on H2 — (T) correction is zero for 2-electron system; tables note this. **Immediate next action: M2** — user gives Codex prompt ③.

## Two-agent workflow
Codex = implementer (writes code/prose). **Claude (me) = supervisor / auditor**: I
direct scope and independently re-verify every number and claim. This is the default
mode for the JOSE effort — but if the user asks me to implement something directly
(as with the notebook work), I just do it. The role is a default, not a cage.

## Standing rules (enforce unless told otherwise)
1. Never frame breadth as capability ("it even does phonons"). Frame each method as
   **pedagogical coverage backed by a validation number**.
2. Freeze scope: add no methods, optimize no performance, unless explicitly asked.
3. No half-validated claims. If a method/basis cell fails or is undefined, mark it
   **experimental** in docs or remove it — never claim it. (PBE / PBE-TDDFT are
   experimental: ~1e-3 Ha grid error.)
4. Trust no un-re-verified number. (A wrong PySCF reference was already found in NB09;
   the LDA/RHF/PBE anchors are in `benchmarks/seed_numbers.json`.)
5. Watch for AI-tells in any prose (hedging, "delve/showcase/leverage", tricolon
   padding) and rewrite into the author's plain, direct voice.

## Environment & commands
Single shared conda env (`base`, `/home/yazicie/miniconda3`) — install once, not per
agent. Notebooks need the extra:
```bash
pip install -e ".[notebooks]"          # molekul + jupyter + matplotlib + pyscf + py3Dmol
pytest tests/                          # full test suite
jupyter nbconvert --to notebook --execute --inplace notebooks/NN_*.ipynb   # headless run
python /tmp/...  # ad-hoc numeric checks: import from src/, compare to PySCF
```
Notebooks are gitignored (`.gitignore: notebooks/`) — they are local working files.

## Element / basis envelope (so claims stay honest)
Atom objects: H–Ar. STO-3G calculations: H–Ne. 6-31G\* and cc-pVDZ: H, He, C, N, O, F.
Atomic masses (frequencies/phonons): H–Ne. MOLEKUL uses **6 Cartesian** d-functions
(cc-pVDZ H₂O = 25, N₂ = 30 — not the 24/28 spherical counts).
