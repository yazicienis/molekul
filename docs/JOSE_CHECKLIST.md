# JOSE Submission Checklist

- [x] Open license: MIT (`LICENSE`).
- [x] Generated validation tables: `benchmarks/validation_table.md` and `benchmarks/periodic_validation.md`.
- [x] Paper draft cites only M1 validation summaries; experimental/broken cells are not claimed as validated.
- [x] Notebook entry guide mirrors `notebooks/README.md` in the top-level README.
- [x] Headless notebook execution gate: all 14 notebooks (00–13) exit 0 with `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800`. NB05 (geometry optimization, ~8 min for 6-31G* cell) and NB12 (phonon dispersion, ~2 min) require the 1800 s per-cell timeout. One fix applied: NB06 dipole cell used stale attribute name `mu_au`/`mu_debye` → corrected to `total_au`/`total_debye`.
- [x] Full test suite gate: 682 passed, 2 skipped, 2 harmless PySCF warnings (electron/spin in H-chain periodic tests), exit 0.
- [x] Zenodo version DOI: `10.5281/zenodo.20558145` (v0.2.0 archive published).
