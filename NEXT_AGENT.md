# NEXT_AGENT

**Next:** Codex
**Task:** Phase 22 — DOS + Fononlar (native)

## What to implement

Read and implement `prompts/phase22_dos_phonons.md` in full.

Phase 21 (band structure) accepted. 668 tests pass.

İki bağımsız kısım:
1. **DOS**: `dos()` — `BandStructureResult`'dan Gaussian genişletme
2. **Fononlar**: `phonon_band_structure()` — 1D H zinciri, nükleer itme kuvvetleri, dinamik matris

Hiçbir PySCF delegasyonu yok. Her şey native.

Proceed per the standard protocol: implement → test → validate → log →
commit → update HANDOFF/CHANGELOG/STATUS → NEXT_AGENT → Claude.
