# HANDOFF

## Last action (Claude — reviewer/completer, 2026-05-27, ACCEPTED)

Phase 14 CCSD(T) complete. Computer had shut down mid-session; Codex had
written all code but hadn't run validation or committed.

- Fixed syntax error in `scripts/validate_ccsdt.py` (mangled `\n` escape chars)
- Ran `pytest tests/test_ccsdt.py -v`: 5/5 passed
- Ran `pytest tests/ -x`: 611/611 passed (606 existing + 5 new)
- Ran `scripts/validate_ccsdt.py`:
  - H2: diff 1.66e-7 Ha (< 1e-6) ✓
  - H2O: diff 9.43e-9 Ha (< 1e-5) ✓
- Generated `outputs/logs/phase14_ccsdt.json` and `.txt`
- NEXT_AGENT → Codex, Phase 15 EOM-CCSD

---

## Previous action (Codex — implementer, 2026-05-23, INTERRUPTED)

Phase 14 CCSD(T) implementation written but not validated/committed before
computer shutdown:
- `ccsdt_energy()` and `_ccsdt_correction_so()` added to `src/molekul/ccsd.py`
- `tests/test_ccsdt.py` created
- `scripts/validate_ccsdt.py` created

---

## Previous action (Claude — reviewer, 2026-05-23, ACCEPTED)

Phase 11 review passed. Commit c114255 accepted.
- All 5 sign/antisymmetriser corrections verified numerically.
- H2: 1.66e-7 Ha from PySCF; H2O: 9.46e-9 Ha from PySCF. Both pass.
- 606 tests pass. No paper files in commit.
- NEXT_AGENT → Codex, Phase 14 (CCSD(T)).
