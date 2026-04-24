# MOLEKUL Experiment Logging

Every validation script produces two output files under `outputs/logs/`:

| File | Purpose |
|------|---------|
| `outputs/logs/phaseN_name.txt` | Human-readable summary for review |
| `outputs/logs/phaseN_name.json` | Machine-readable metrics for status tracking |

Bulkier artifacts (trajectories, integral dumps, checkpoints) go in:

| Directory | Contents |
|-----------|---------|
| `outputs/` | Legacy flat outputs; intermediate files |
| `outputs/logs/` | Phase report pairs (`.txt` + `.json`) |
| `outputs/checkpoints/` | Serialised state for inter-phase continuity |

---

## Phase numbering

| Phase | Module(s) | Validation script | Log prefix |
|-------|-----------|-------------------|------------|
| 1 | `io_xyz`, `molecule`, `atoms` | `scripts/run_example.py` | `phase1_molecules` |
| 2 | `integrals`, `basis`, `basis_sto3g` | `scripts/validate_phase2.py` | `phase2_integrals` |
| 3 | `eri` | pytest `tests/test_eri.py` | *(test suite only)* |
| 4 | `rhf` | `scripts/validate_rhf.py` | `phase4_rhf` |
| 5 | `grad`, `optimizer` | `scripts/validate_optimizer.py` | `phase5_optimizer` |

Phase 3 (ERI) validation is covered entirely by the pytest suite (`tests/test_eri.py`,
44 tests) and the optional `scripts/compare_pyscf.py`. No standalone log file is
generated because the ERI tensor is intermediate state; its correctness is verified
through the RHF energy (Phase 4) which uses it.

---

## JSON log schema

Every `outputs/logs/phaseN_name.json` file has the following top-level keys:

```json
{
  "phase":     "phase4",
  "name":      "rhf",
  "timestamp": "2026-04-01T12:00:00",
  "git_sha":   "abcdef1",
  "status":    "PASS",
  "n_passed":  5,
  "n_failed":  0,
  "elapsed_s": 3.14,
  "metrics":   { ... },
  "artifacts": [ ... ],
  "details":   { ... }
}
```

### `status` values

| Value | Meaning |
|-------|---------|
| `PASS` | All recorded checks passed |
| `FAIL` | One or more checks failed |
| `PARTIAL` | Some checks passed, some failed |

### `metrics` — key scalars per phase

**Phase 1 (`phase1_molecules.json`)**
- `n_molecules` — number of XYZ files processed
- `{mol}_n_electrons` — electron count per molecule
- `{mol}_enuc_ha` — nuclear repulsion energy (Hartree)
- `roundtrip_max_diff_bohr` — XYZ write/re-read coordinate error

**Phase 2 (`phase2_integrals.json`)**
- `h2_S12`, `h2_T11`, `h2_T12`, `h2_H11`, `h2_H12` — integral values at R=1.4 bohr
- `h2_S12_ref_so`, `h2_T11_ref_so` — Szabo & Ostlund reference values
- `h2_S12_err`, `h2_T11_err` — absolute errors vs reference
- `n_checks_run`, `n_checks_failed`

**Phase 4 (`phase4_rhf.json`)**
- `{mol}_energy_ha` — total RHF/STO-3G energy (Hartree)
- `{mol}_enuc_ha` — nuclear repulsion
- `{mol}_converged` — SCF convergence flag
- `{mol}_n_iter` — SCF iteration count
- `{mol}_pyscf_ref_ha` — PySCF reference energy
- `{mol}_pyscf_diff_ha` — deviation from PySCF

**Phase 5 (`phase5_optimizer.json`)**
- `h2_start_R_bohr`, `h2_start_energy_ha` — initial geometry
- `h2_opt_R_bohr`, `h2_opt_R_ang` — optimised bond length
- `h2_opt_energy_ha` — optimised total energy
- `h2_opt_grad_max_ha_bohr` — final max gradient component
- `h2_opt_n_steps` — BFGS step count
- `pyscf_ref_R_bohr`, `pyscf_ref_energy_ha` — PySCF equilibrium reference
- `delta_R_vs_pyscf_bohr`, `delta_E_vs_pyscf_ha` — deviations from PySCF
- `grad_h_bohr` — finite-difference step size used

---

## Naming convention

```
outputs/logs/{phase}_{name}.{ext}
```

- `phase` — always `phaseN` (e.g. `phase4`)
- `name`  — short slug for the experiment (e.g. `rhf`, `integrals`, `optimizer`)
- `ext`   — `txt` for human-readable, `json` for machine-readable

Checkpoints follow:

```
outputs/checkpoints/{phase}_{name}_{molecule}.{ext}
```

e.g. `outputs/checkpoints/phase5_optimizer_h2_eq.xyz`

---

## How to run all validations

```bash
# Phase 1
python scripts/run_example.py

# Phase 2
python scripts/validate_phase2.py

# Phase 3 — test suite (ERI)
python -m pytest tests/test_eri.py -v

# Phase 4
python scripts/validate_rhf.py

# Phase 5
python scripts/validate_optimizer.py

# Overall status
python scripts/status.py
```

---

## How to read logs programmatically

```python
import json
from pathlib import Path

log = json.loads(Path("outputs/logs/phase4_rhf.json").read_text())
print(log["status"])                    # "PASS"
print(log["metrics"]["h2_energy_ha"])   # -1.1167143190
```

---

## Adding logging to a new script

```python
from molekul.logging_utils import ExperimentLogger

exp = ExperimentLogger("phase6", "my_experiment")

# Record scalar metrics
exp.metric("some_energy_ha", result.energy)

# Record pass/fail checks
exp.check("Energy below zero", result.energy < 0)

# Record output files
exp.artifact("outputs/my_output.xyz")

# Add human-readable lines
exp.line("Method: RHF/STO-3G")
exp.line(f"Final energy: {result.energy:.10f} Ha")

# Write both files
txt_path, json_path = exp.save()
```

The logger automatically records timestamp, git SHA, elapsed time, and
pass/fail summary. No boilerplate needed beyond the calls above.

---

## Reproducibility notes

- All scripts use **atomic units (bohr, Hartree)** internally
- Geometries are stored in bohr in memory; XYZ files use Angstrom
- Reference values are from **PySCF 2.x at identical geometries** unless noted
- Git SHA is recorded in every log; run `git log --oneline -5` to match
- The `outputs/` directory is in `.gitignore`; logs must be regenerated from source
