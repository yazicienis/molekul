# NEXT_AGENT

**Next:** Codex
**Task:** Phase 23 — Tam Fononlar (elektronik + nükleer kuvvetler)

## What to implement

Read and implement `prompts/phase23_phonons_full.md` in full.

Phase 22 (DOS + nuclear-only phonons) accepted. 677 tests pass.

Key points:
- `periodic_force_constants()`: `periodic_hf()` enerjisinden 4-noktalı FD
- `phonon_band_structure_full()`: Phase 22'deki `phonon_band_structure()` ile aynı
  dinamik matris ama tam kuvvet sabitleriyle
- Test: `test_full_vs_nuclear_different()` — elektronik katkının varlığını göster
- MOLECULES'ün doğal bitiş noktası bu faz

Proceed per the standard protocol: implement → test → validate → log →
commit → update HANDOFF/CHANGELOG/STATUS → NEXT_AGENT → Claude.
