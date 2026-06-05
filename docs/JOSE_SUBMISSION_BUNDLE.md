# JOSE Submission Bundle

Prepared: 2026-06-05
Target release: v0.2.0

## Included submission files

- `paper.md` — JOSE manuscript draft.
- `paper.bib` — bibliography used by `paper.md`.
- `README.md` — repository entry point and notebook course map.
- `CITATION.cff` — citation metadata for v0.2.0.
- `LICENSE` — MIT license.
- `benchmarks/validation_table.md` and `benchmarks/validation_table.json` — M1 molecular validation table.
- `benchmarks/periodic_validation.md` and `benchmarks/periodic_validation.json` — M1 periodic mini-validation table.
- `docs/JOSE_CHECKLIST.md` — gate/checklist status.

## Repository state

- Package version: `0.2.0` in `pyproject.toml` and `src/molekul/__init__.py`.
- Local release tag: `v0.2.0` (created for this finalize state; push it with the commit).
- M1 gate commit: `e68a085 M1 gate: audit fixes to validation tables`.
- M2 gate commit: `3e5bac2 M2 gate: audit fixes — notebooks, CITATION.cff, checklist`.
- Finalize commit: this commit (`Finalize JOSE v0.2.0 bundle`; see `git log -1`).

## Validation gates already completed

- M1 validation table audit passed.
- M2 audit fixed notebook/CITATION/checklist issues.
- Headless notebook gate: all notebooks 00-13 executed successfully with `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800`.
- Full test suite gate: `682 passed, 2 skipped`, with two harmless PySCF periodic warnings.

## Zenodo archive

- Existing Zenodo concept DOI: `10.5281/zenodo.19763107`.
- Version-specific v0.2.0 DOI: **must be minted in Zenodo after publishing the v0.2.0 archive/release**.
- Do not replace this with a guessed DOI. The DOI is assigned by Zenodo only after upload/release.

## Final external action

After this commit is pushed, push tag `v0.2.0`, publish the GitHub/Zenodo archive, copy the version-specific DOI into `CITATION.cff`, `README.md`, `paper.md`, and this bundle file if JOSE requires the version DOI rather than the concept DOI.
