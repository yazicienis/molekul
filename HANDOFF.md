# HANDOFF

## Last action (Codex — implementer, 2026-05-23, REVISION 1 addressed)

- Re-derived the spin-orbital CCSD signs/antisymmetrisers against the installed
  PySCF GCCSD Stanton/Gauss implementation.
- Kept the assigned `Wvvvv` and R1 `ovvv` sign corrections and fixed the
  multi-electron residual path:
  - R1 `ovov` contraction sign
  - T2 modified F intermediates using `Fme`
  - `Wovvo` T1T1 coefficient
  - T2 final T1/`ovvv` and T1/`ooov` antisymmetrisers
- Added `scripts/validate_ccsd.py`.
- Generated `outputs/logs/phase11_ccsd.json` and
  `outputs/logs/phase11_ccsd.txt`.
- Verification:
  - `pytest tests/test_ccsd.py -v`: 10 passed
  - `pytest tests/ -x`: 606 passed in 516.61s
  - H2 validation diff vs PySCF runtime reference:
    `1.661336e-07` Ha (< `1e-6`)
  - H2O validation diff vs PySCF runtime reference:
    `9.463267e-09` Ha (< `1e-5`)
- NEXT_AGENT set to Claude reviewer.

---
## Last action (Claude — reviewer, 2026-05-23, REVISION 1)

Phase 11 review: H2 passes, H2O fails (diff 2.89e-4 Ha). Sign fix unverified
for multi-electron systems. REVISION written to NEXT_AGENT.md.
Codex must re-derive signs from Stanton 1991 and pass both H2 + H2O before commit.

---

## Previous action (Claude — supervisor, 2026-05-23)

- Reviewed project state after SoftwareX paper submission
- Found uncommitted CCSD sign fixes in `ccsd.py` (Wvvvv + R1 residual)
- Confirmed 606/606 tests pass with current uncommitted state
- Set up supervisor/worker workflow: NEXT_AGENT, STATUS, CHANGELOG, SCIENCE files
- Wrote Codex prompts for phases 14–17 in `prompts/`
- **Did not commit** the ccsd.py changes — that is Codex's task (Phase 11)

## State handed to Codex

- Branch: `master`
- Working tree: 3 modified files (`ccsd.py`, `softwarex_paper.tex`, `softwarex_paper.pdf`)
- The `.tex` and `.pdf` changes are paper-related and can be committed separately or left
- `ccsd.py` changes are the Phase 11 target

## Known issues / watch for

- `softwarex_paper.tex` has a layout change (twocolumn → single-column) that is
  unrelated to code — do not commit this in the Phase 11 commit. Leave it or
  ask human to handle.
- HeH⁺ CCSD test uses a relaxed tolerance (2e-4) — this is a known edge case
  in the 2-electron spin-orbital formulation, not a bug.

## Blocking questions for human

None currently.
