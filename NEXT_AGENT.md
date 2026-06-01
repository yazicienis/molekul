# NEXT_AGENT

**Next:** Claude
**Task:** Review Phase 22 — DOS + nuclear-only phonons

Codex implemented `prompts/phase22_dos_phonons.md`.

Review focus:
- `src/molekul/periodic.py`: `DOSResult`, `dos()`, `PhononResult`, `phonon_band_structure()`
- `tests/test_dos.py`
- `tests/test_phonons.py`
- `scripts/validate_dos_phonons.py`
- `outputs/logs/phase22_dos_phonons.json/.txt`
- `SCIENCE.md` entries for DOS broadening and nuclear-only phonon finite-difference controls

Validation already run by Codex:
- `pytest tests/test_dos.py tests/test_phonons.py -q`: 9 passed
- `python scripts/validate_dos_phonons.py`: PASS
- `pytest tests/test_band_structure.py tests/test_periodic_hf_1d.py -q`: 13 passed, 2 existing PySCF warnings
- `pytest tests/ -x`: 677 passed, 2 skipped, 2 warnings in 618.56s

Reviewer note: phonons intentionally include only nuclear repulsion force constants. Electronic Hellmann-Feynman/Pulay force constants are omitted, so the resulting frequencies are educational infrastructure values, not physical production phonons.
