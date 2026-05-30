# NEXT_AGENT

**Next:** Claude
**Task:** Review Phase 20c — Ewald + 3D periodic HF

Codex implemented `prompts/phase20c_periodic_hf_3d.md` and committed the accepted Phase 18–20b stack first.

Review focus:
- `src/molekul/periodic.py`: `ewald_energy()`, `ewald_hcore()`, and the new 3D `periodic_hf(..., use_ewald=True)` path
- `tests/test_periodic_hf_3d.py`
- `scripts/validate_periodic_hf_3d.py`
- `outputs/logs/phase20c_periodic_hf_3d.json/.txt`
- `SCIENCE.md` entries for Ewald eta/truncation and the fast PySCF PBC reference grid

Validation already run by Codex:
- `pytest tests/test_periodic_hf_3d.py -q`: 6 passed
- `python scripts/validate_periodic_hf_3d.py`: PASS
- `pytest tests/test_periodic_hf_1d.py -q`: 6 passed, 2 existing PySCF warnings
- `pytest tests/ -x`: 661 passed, 2 skipped, 2 warnings in 615.64s

Reviewer note: the Phase 20c 3D energy path delegates to PySCF when available rather than implementing native production-quality 3D periodic ERI/Ewald J/K. Decide whether this is acceptable for Phase 20c or should be split into a stricter native implementation phase before Phase 21.
