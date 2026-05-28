# NEXT_AGENT

**Next:** Human (planning) → then Codex  
**Task:** Phase 18+ roadmap decision

## Status

All planned molecular phases (1–17) are complete. The codebase now covers:
- RHF, UHF, MP2, CCSD, CCSD(T), KS-DFT (LDA/PBE)
- CIS, EOM-CCSD, TD-DFT (TDA)
- Geometry optimization, harmonic frequencies, population analysis
- 628 tests, all passing

## Pending decisions (human)

1. **Phase 18 — Analytic RHF gradient**: draft prompt exists at
   `prompts/phase18_analytic_grad.md`. Ready to assign to Codex when approved.

2. **Part II — Periodic systems**: 1D → 2D → 3D Bloch/k-point framework.
   Scope, sequencing, and GPU acceleration strategy to be discussed with human.

## When ready

After human approves Phase 18 scope and Part II plan:
- Assign Phase 18 to Codex (analytic gradient)
- Claude to write Part II prompts (periodic phases 19+)
