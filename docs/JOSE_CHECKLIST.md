# JOSE Submission Checklist

- [x] Open license: MIT (`LICENSE`).
- [x] Generated validation tables: `benchmarks/validation_table.md` and `benchmarks/periodic_validation.md`.
- [x] Paper draft cites only M1 validation summaries; experimental/broken cells are not claimed as validated.
- [x] Notebook entry guide mirrors `notebooks/README.md` in the top-level README.
- [ ] Headless notebook execution gate: run all notebooks 00-13 with `jupyter nbconvert --execute` in the M2 audit.
- [ ] Full test suite gate: run `pytest tests/` in the M2 audit/final pass.
- [ ] Zenodo DOI bump: mint a new archive during final JOSE submission assembly, not in this draft step.
