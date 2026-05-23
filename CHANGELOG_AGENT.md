# CHANGELOG — Agent Actions

Format: `YYYY-MM-DD | Agent | Phase | Action | Commit`

---

## 2026-05-23 | Claude | setup | Workflow bootstrap

- Created NEXT_AGENT.md, HANDOFF.md, STATUS.md, CHANGELOG_AGENT.md, SCIENCE.md
- Created PHASES.md (phase roadmap), WORKFLOW.md (protocol)
- Created `prompts/` with Codex prompts for phases 11, 14, 15, 16, 17
- No commits made — setup only

---

_Earlier history (pre-workflow) reconstructed from git log:_

| Date | Phase | Action | Commit |
|------|-------|--------|--------|
| 2026-04-18 | 13 | CIS implementation | — |
| 2026-04 | 12 | KS-DFT LDA/PBE | — |
| 2026-04 | 11 | CCSD spin-orbital (sign bug pending) | — |
| 2026-04 | 10 | Harmonic frequencies | — |
| 2026-04 | 1–9 | Core methods | — |
| 2026-05 | paper | SoftwareX V6 submission (human, local) | 4eb8de9 |

---

## 2026-05-23 | Codex | Phase 11 | CCSD sign fix validation + commit

- Confirmed and revised spin-orbital CCSD sign/antisymmetriser corrections in
  `_make_intermediates_so`, `_t1_residual_so`, and `_t2_residual_so`
- Added `scripts/validate_ccsd.py`
- Generated `outputs/logs/phase11_ccsd.json` and `.txt`
- Ran `pytest tests/test_ccsd.py -v`: 10 passed
- Ran `pytest tests/ -x`: 606 passed
- H2 validation diff vs PySCF runtime reference: `1.661336e-07` Ha
- H2O validation diff vs PySCF runtime reference: `9.463267e-09` Ha
- Commit: `PENDING`
