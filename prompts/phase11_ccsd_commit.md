# Phase 11: CCSD — Commit Sign Fix + Log

## Context

`src/molekul/ccsd.py` is fully implemented. Two sign errors were found and
corrected but not yet committed:

1. `_make_intermediates_so`: `Wvvvv` intermediate — the P̂(ab) antisymmetriser
   signs were flipped.
2. `_t1_residual_so`: the `+½ Σmef t_{im}^{ef} <ma||ef>` term must be
   negative (`-½`), confirmed numerically against PySCF.

All 606 tests pass with the corrected signs.

## Objective

1. Commit the two sign fixes already present in `src/molekul/ccsd.py`.
2. Write a validation script that cross-checks CCSD correlation energies
   against PySCF for H₂ and H₂O.
3. Produce `outputs/logs/phase11_ccsd.json` and `outputs/logs/phase11_ccsd.txt`.

## Validation Script

File: `scripts/validate_ccsd.py`

Molecules to test:

| Molecule | Geometry (Å) | Basis | PySCF E_corr (Ha) | Tolerance |
|----------|-------------|-------|-------------------|-----------|
| H₂ | r=0.74 | STO-3G | -0.020524691 | 1e-6 |
| H₂O | O at origin, H at ±0.757/−0.469 Å | STO-3G | TBD — run PySCF | 1e-5 |

If PySCF is available (`import pyscf`), compute references on the fly.
If not, use the hardcoded values from `tests/test_ccsd.py`.

## Log Schema

```json
{
  "phase": 11,
  "description": "CCSD spin-orbital implementation",
  "date": "YYYY-MM-DD",
  "results": [
    {
      "molecule": "H2",
      "basis": "STO-3G",
      "E_corr_Ha": -0.020524691,
      "ref_pyscf_Ha": -0.020524691,
      "diff": 0.0
    }
  ],
  "sign_fixes": [
    "Wvvvv P̂(ab) signs corrected in _make_intermediates_so",
    "R1 ovvv contraction sign corrected in _t1_residual_so"
  ],
  "notes": "..."
}
```

## Commit Message

```
Phase 11: CCSD sign fixes in Wvvvv and R1 residual

Correct P̂(ab) antisymmetriser signs in _make_intermediates_so and
the ovvv contraction sign in _t1_residual_so. Both confirmed
numerically against PySCF. All 606 tests pass.
```

## Acceptance Criteria

- `pytest tests/test_ccsd.py -v` — all cases pass
- `pytest tests/ -x` — no regressions
- Log file present at `outputs/logs/phase11_ccsd.json`
- H₂ E_corr diff < 1e-6 Ha
