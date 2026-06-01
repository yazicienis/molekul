# NEXT_AGENT

**Next:** Claude
**Task:** Review Phase 21 — native band structure

Codex implemented `prompts/phase21_periodic_dft.md`.

Review focus:
- `src/molekul/periodic.py`: `BandStructureResult`, `kpath()`, `band_structure()`
- `tests/test_band_structure.py`
- `scripts/validate_band_structure.py`
- `outputs/logs/phase21_band_structure.json/.txt`
- `SCIENCE.md` entry for `n_points=50` k-path sampling

Validation already run by Codex:
- `pytest tests/test_band_structure.py -q`: 7 passed
- `python scripts/validate_band_structure.py`: PASS
- `pytest tests/test_periodic_hf_3d.py tests/test_periodic_infrastructure.py -q`: 14 passed
- `pytest tests/ -x`: 668 passed, 2 skipped, 2 warnings in 619.55s

Reviewer note: this is a one-electron tight-binding band structure from `H_core(k)` and `S(k)`, not an HF/DFT quasiparticle band structure. The LiH logged one-electron gap is negative and should be reviewed only as a native-path regression value.
