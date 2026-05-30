# Phase 14: CCSD(T) — Perturbative Triples

## Context

CCSD is implemented in `src/molekul/ccsd.py` and exports:
- `ccsd_energy(mol, basis_fn, rhf_result, verbose) -> CCSDResult`
- `CCSDResult`: fields `t1, t2, energy_ccsd, energy_hf, energy_mp2, energy_total,
  converged, n_iter, n_occ, n_virt, n_basis`

The spin-orbital two-electron integrals and orbital energies are built inside
`ccsd_energy`. You will need to expose them or refactor to a helper so CCSD(T)
can reuse them without repeating the integral transformation.

Available imports: `numpy`, all modules in `src/molekul/`.

## Objective

Implement the (T) perturbative triples correction on top of converged CCSD
amplitudes. Add `ccsdt_energy()` to `src/molekul/ccsd.py` and extend
`CCSDResult` (or return a new `CCSDTResult`) with the triples correction.

## Theory

Raghavachari et al. (1989), J. Chem. Phys. 91, 1007.
Stanton et al. spin-orbital form (same paper, Eq. A1–A5):

```
E_(T) = (1/36) Σ_{ijk,abc} t3_{ijk}^{abc} * W_{ijk}^{abc}
```

where the disconnected triples amplitudes are:

```
t3_{ijk}^{abc} = P̂(ijk/abc) [ t2_{ij}^{ab} * f_{kc} / D_{ijk}^{abc} ]
               + P̂(ijk/abc) [ <ij||ak> * t1_c / D_{ijk}^{abc} ]
```

and `D_{ijk}^{abc} = ε_i + ε_j + ε_k - ε_a - ε_b - ε_c`.

The full expression uses the "W" intermediates (see Stanton 1989 Appendix).
Use the spin-orbital formulation throughout; restrict to closed-shell.

Key reference: Szabo & Ostlund §6; Crawford & Schaefer, Rev. Comp. Chem. 14.

## Implementation

File: `src/molekul/ccsd.py` (extend existing file)

```python
@dataclass
class CCSDTResult:
    energy_ccsdt: float       # triples correction only
    energy_total: float       # RHF + CCSD + (T)
    energy_hf: float
    energy_ccsd: float        # CCSD correlation (from CCSDResult)
    energy_mp2: float
    n_occ: int
    n_virt: int
    n_basis: int

def ccsdt_energy(
    mol: Molecule,
    basis_fn,
    rhf_result,
    verbose: bool = False,
) -> CCSDTResult:
    ...
```

- Reuse the CCSD amplitudes by calling `ccsd_energy()` internally.
- The triples loop is O(N^7). For N ≤ 10 basis functions this is acceptable.
  Do NOT add batching or optimization — clarity over speed.
- Use `np.einsum` or explicit index loops; document the index mapping.

## Tests

File: `tests/test_ccsdt.py`

PySCF reference values (compute with `pyscf.cc.CCSD(mf).run()`,
then `ccsd.ccsd_t()` for the (T) correction):

| Molecule | Basis | E(T) Ha | Tolerance |
|----------|-------|---------|-----------|
| H₂ r=0.74Å | STO-3G | ~0.0 (2-electron, triples vanish) | < 1e-10 |
| H₂O std geom | STO-3G | TBD from PySCF | 1e-5 |
| HF r=0.917Å | STO-3G | TBD from PySCF | 1e-5 |

Required tests:
- `test_h2_triples_zero` — H₂ (T) correction must be < 1e-10 (only 2 electrons)
- `test_h2o_ccsdt_ecorr` — total CCSD(T) corr within 1e-5 of PySCF
- `test_hf_ccsdt_ecorr` — same for HF
- `test_ccsdt_result_fields` — all dataclass fields present
- `test_ccsdt_lower_than_ccsd` — E_total(CCSD(T)) ≤ E_total(CCSD)

## Validation Script

File: `scripts/validate_ccsdt.py`

Output: `outputs/logs/phase14_ccsdt.json` and `outputs/logs/phase14_ccsdt.txt`

Log schema:
```json
{
  "phase": 14,
  "description": "CCSD(T) perturbative triples",
  "date": "YYYY-MM-DD",
  "results": [
    {
      "molecule": "H2O",
      "basis": "STO-3G",
      "E_ccsdt_corr_Ha": -0.XXXXX,
      "ref_pyscf_Ha": -0.XXXXX,
      "diff": 0.0000X
    }
  ],
  "notes": "..."
}
```

## Acceptance Criteria

- `pytest tests/test_ccsdt.py -v` — all cases pass
- `pytest tests/ -x` — no regressions
- H₂O E(T) diff < 1e-5 Ha vs PySCF
- Log file committed in `outputs/logs/phase14_ccsdt.json`
