# NEXT_AGENT

**Next:** Codex  
**Task:** (1) Commit all pending phases, then (2) Phase 20c — Ewald + 3D periodic HF

## Step 1 — Commit first (required before 20c)

The working tree has accepted but uncommitted changes spanning Phase 18,
Phase 19, audit cleanup, Phase 20a, and Phase 20b. Make a single well-described
commit (or a series of logical commits) covering all of them before starting
Phase 20c. All 655 tests pass (2 CuPy-gated skips) and the working tree is clean
except for the new phase files.

## Step 2 — Implement Phase 20c

Read and implement `prompts/phase20c_periodic_hf_3d.md` in full.

Phase 20b (periodic HF 1D) is accepted. The PySCF comparison was deferred from
20b because `low_dim_ft_type=inf_vacuum` fails in PySCF 2.12.1. With Ewald in
Phase 20c, a proper 3D PySCF comparison (standard PBC, no 1D vacuum settings)
becomes possible. Choose a 3D test system that works cleanly with the available
PySCF version.

Proceed per the standard Codex protocol: implement → test → validate → log →
commit → update HANDOFF/CHANGELOG/STATUS → set NEXT_AGENT → Claude.
