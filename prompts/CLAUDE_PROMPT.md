# Claude Prompt (sabit — her seferinde bu)

---

Your turn.

Read first:
- NEXT_AGENT.md
- HANDOFF.md
- STATUS.md
- CHANGELOG_AGENT.md
- `git diff HEAD~1` (or the range since last Claude review)

If the review involves a new numerical parameter or a change to an existing
one, also read SCIENCE.md and verify the citation and justification.

Then review only the changes Codex just made per HANDOFF.md.

Rules:
- No broad repo scan unless a specific cross-file issue is suspected.
- No scope expansion. No refactoring beyond what blocks acceptance.
- Report blocking issues first, minor notes after.
- Flag specifically: wrong signs in quantum-chemical equations, missing
  PySCF cross-validation, hardcoded magic numbers without SCIENCE.md entry,
  test tolerances too loose (> 1e-3 Ha without documented reason),
  uncommitted validation outputs.
- If acceptable: update NEXT_AGENT.md with the next phase task, update
  STATUS.md, append to CHANGELOG_AGENT.md. Set NEXT_AGENT → Codex.
- If not acceptable: write a REVISION block at the bottom of NEXT_AGENT.md
  explaining exactly what to fix. Set NEXT_AGENT → Codex.
