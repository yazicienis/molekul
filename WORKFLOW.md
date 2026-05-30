# MOLEKUL — Supervisor / Worker Workflow

## Roles

**Supervisor: Claude**
- Reads completed phase logs and test output
- Decides next phase scope and sequencing
- Writes the Codex prompt (`prompts/phaseNN_<name>.md`)
- Reviews Codex output: checks log values against PySCF, reads diff, approves commit
- Updates `PHASES.md` after each phase lands

**Worker: Codex**
- Receives a single prompt file, no other context needed
- Implements the module, tests, and validation script
- Runs the validation script, captures output to `outputs/logs/`
- Commits with message: `Phase NN: <title>`

---

## Phase Lifecycle

```
1. Supervisor writes prompts/phaseNN_<name>.md
2. Codex reads prompt → implements → runs validation → commits
3. Supervisor reviews:
     git diff HEAD~1
     cat outputs/logs/phaseNN_<name>.txt
     pytest tests/test_<module>.py -v
4. If accepted: Supervisor updates PHASES.md (⚠️ → ✅)
5. If rejected: Supervisor annotates prompt with "REVISION N:" block, Codex re-runs
```

---

## Prompt File Format

Every prompt in `prompts/` follows this template:

```
# Phase NN: <Title>

## Context
What is already implemented. Which modules Codex may import.

## Objective
Exactly what to build. One paragraph.

## Theory
Key equations with notation matching the codebase (spin-orbital, physicist bra-ket, etc.).
Reference: Author Year, eq. N.

## Implementation
- File: src/molekul/<module>.py
- Public API: function signatures, return dataclass fields
- Constraints: pure NumPy only, no SciPy, closed-shell only (unless stated)

## Tests
File: tests/test_<module>.py
Required test cases with PySCF reference values and tolerances.

## Validation Script
File: scripts/validate_<module>.py
Must write outputs/logs/phaseNN_<name>.json and outputs/logs/phaseNN_<name>.txt

## Log Schema
{
  "phase": NN,
  "description": "...",
  "date": "YYYY-MM-DD",
  "results": [...],
  "notes": "..."
}

## Acceptance Criteria
- All listed pytest cases pass
- Log diff vs PySCF within stated tolerance
- No new test regressions (pytest tests/ -x passes)
```

---

## Communication Protocol

When Codex finishes a phase, it posts:
```
PHASE NN COMPLETE
log: outputs/logs/phaseNN_<name>.txt
tests: N passed
max_error: X Ha vs PySCF
commit: <hash>
```

Supervisor responds with either:
- `ACCEPTED — update PHASES.md, proceed to phase NN+1`
- `REVISION: <specific issue> — see prompt REVISION 1 block`
