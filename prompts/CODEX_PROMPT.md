# Codex Prompt (sabit — her seferinde bu)

---

Your turn.

Read first:
- NEXT_AGENT.md
- HANDOFF.md
- STATUS.md
- CHANGELOG_AGENT.md
- `git status` and `git diff` if relevant

If the assigned task introduces or changes any numerical parameter
(convergence thresholds, grid sizes, step sizes, basis choices, iteration
limits), also read SCIENCE.md.

Then implement only the assigned task per NEXT_AGENT.md.

Rules:
- Keep scope minimal. Do not touch unrelated files.
- Run only the checks listed in the task's acceptance self-check.
- Never invent numerical parameters. Use `TODO(human):` markers if unsure.
- All new code must pass `pytest tests/ -x` before you hand off.
- If you set any numerical parameter, append a justified entry to SCIENCE.md.
- When done, update HANDOFF.md, STATUS.md, CHANGELOG_AGENT.md, and NEXT_AGENT.md.
- Set NEXT_AGENT → Claude (reviewer).
