# NEXT_AGENT

**Next:** Claude
**Task:** Review Phase 23 — full periodic-HF phonons

Codex implemented `prompts/phase23_phonons_full.md`.

Review focus:
- `src/molekul/periodic.py`: `periodic_force_constants()` and `phonon_band_structure_full()`
- `tests/test_phonons_full.py`
- `scripts/validate_phonons_full.py`
- `outputs/logs/phase23_phonons_full.json/.txt`
- `SCIENCE.md` entry for full periodic phonon finite-difference controls

Validation already run by Codex:
- `pytest tests/test_phonons_full.py -q`: 5 passed
- `python scripts/validate_phonons_full.py`: PASS
- `pytest tests/test_phonons.py tests/test_phonons_full.py -q`: 9 passed
- `pytest tests/ -x`: 682 passed, 2 skipped, 2 warnings in 634.59s

Reviewer note: because the current periodic HF engine is one-cell/1D, relative image-cell displacement is represented as a small lattice-vector perturbation before SCF energy evaluation. This keeps Phase 23 native and energy-based, but it is still an educational force-constant model, not a production supercell phonon implementation.
