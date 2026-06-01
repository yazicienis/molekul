# NEXT_AGENT

**Next:** Codex
**Task:** Phase 21 — Band structure (H chain + LiH, native)

## What to implement

Read and implement `prompts/phase21_periodic_dft.md` in full.

Scope is now corrected: band structure via H_core(k) diagonalization,
no PySCF delegation. Everything is native and traceable.

Key additions:
- `BandStructureResult` dataclass
- `kpath()` helper (lineer interpolasyon between special points)
- `band_structure()` function — uses `bloch_hcore()` + `bloch_overlap()` + `_generalized_eigh()`
- Tests: 1D H chain dispersiyon, 3D LiH shape/n_occ
- Validation script + log

Proceed per the standard protocol: implement → test → validate → log →
commit → update HANDOFF/CHANGELOG/STATUS → NEXT_AGENT → Claude.
