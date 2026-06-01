# MOLEKUL — Phase Roadmap

## Workflow Summary

| Role | Agent | Responsibility |
|------|-------|----------------|
| Supervisor | Claude | Phase planning, prompt authoring, output review, PHASES.md updates |
| Worker | Codex | Implementation, tests, log generation, commits |

Each phase follows the lifecycle: **Plan → Prompt → Implement → Validate → Log → Commit → Update here**

Codex prompts live in `prompts/phaseNN_<name>.md`.  
Validation logs live in `outputs/logs/phaseNN_<name>.{json,txt}`.

---

## Part I — Molecular Quantum Chemistry (COMPLETE)

| # | Name | Module(s) | Status |
|---|------|-----------|--------|
| 1 | Molecules & Atoms | `atoms.py`, `molecule.py`, `constants.py` | ✅ |
| 2 | One-electron integrals | `integrals.py` | ✅ |
| 3 | Basis sets (STO-3G) | `basis_sto3g.py`, `basis.py` | ✅ |
| 4 | RHF SCF (DIIS + SAD) | `rhf.py` | ✅ |
| 5 | Geometry optimizer (numerical grad) | `optimizer.py` | ✅ |
| 6 | Cube file export | `cube.py` | ✅ |
| 7 | Population analysis | `population.py` | ✅ |
| 8 | MP2 correlation | `mp2.py` | ✅ |
| 9 | Basis sets (6-31G*, cc-pVDZ) + profiling | `basis_631gstar.py`, `basis_ccpvdz.py` | ✅ |
| 10 | Harmonic frequencies & IR | `freqs.py` | ✅ |
| 11 | CCSD spin-orbital | `ccsd.py` | ✅ |
| 12 | KS-DFT (LDA + PBE) | `dft.py` | ✅ |
| 13 | CIS excited states | `cis.py` | ✅ |
| 14 | CCSD(T) perturbative triples | `ccsd.py` | ✅ |
| 15 | EOM-CCSD excited states | `eom_ccsd.py` | ✅ |
| 16 | UHF (unrestricted HF) | `uhf.py` | ✅ |
| 17 | TD-DFT (Casida TDA) | `tddft.py` | ✅ |

677 tests passing as of 2026-06-01.

---

## Part II — Periodic Systems & GPU (PLANNED)

### Design decisions (agreed 2026-05-29)

- **GPU:** Optional CuPy backend (`xp = cupy if use_gpu else numpy`); NumPy API preserved.
  CPU default, GPU opt-in. Tests always run on CPU.
- **Periodic approach:** Gaussian-type orbitals with Bloch theorem (like CRYSTAL code).
  Real-space cutoff first (pedagogical), Ewald summation added for accuracy.
- **Validation:** PySCF periodic module (`pyscf.pbc`) as reference.
- **1D test systems:** H chain (1D), then 3D cubic systems (LiH rock-salt etc.)
- **Phonons:** finite-difference from periodic forces.

| # | Name | Depends on | Prompt | Status |
|---|------|-----------|--------|--------|
| 18 | Semi-numerical RHF gradient (analytic expression + FD integral derivatives) | 4 | `prompts/phase18_analytic_grad.md` | ✅ complete |
| 19 | GPU backend (optional CuPy) | 4, 11 | `prompts/phase19_gpu_backend.md` | ✅ complete |
| 20a | Periodic infrastructure (Crystal, lattice, Bloch S+H) | 19 | `prompts/phase20a_periodic_infrastructure.md` | ✅ complete |
| 20b | Periodic HF 1D (H chain, real-space cutoff) | 20a | `prompts/phase20b_periodic_hf_1d.md` | ✅ complete |
| 20c | Periodic HF 3D (LiH, Ewald, k-mesh) | 20b | `prompts/phase20c_periodic_hf_3d.md` | ✅ complete |
| 21 | Band structure (H chain + LiH, native tight-binding) | 20c | `prompts/phase21_periodic_dft.md` | ✅ complete |
| 22 | DOS + nuclear-only phonons | 21 | `prompts/phase22_dos_phonons.md` | 🔶 ready for review |

---

## Long-term Vision

- **Target audience:** undergraduate, MSc, and PhD students in computational
  materials science. Even PhD students often use codes as black boxes — MOLEKUL
  is designed to break that cycle.
- **Platform goal:** elective-course-level teaching platform. Every algorithm is
  transparent; every step can be traced.
- **Documentation roadmap:**
  1. All phases complete first ("see the full picture")
  2. Jupyter notebooks — one per phase, narrative + running code + exercises
  3. Course book — distilled from notebooks, authored text only (cannot be
     agent-written; this is the PI's voice)
- **Scientific usability** is a secondary goal — SoftwareX paper already
  demonstrates this. Accuracy benchmarks against PySCF in every phase.

---

## Supervisor Notes

- All reference values must be cross-checked with PySCF at identical geometries before logging.
- Each phase produces at minimum: one `.json` + one `.txt` log in `outputs/logs/`.
- Tests go in `tests/test_<module>.py`; reference values hardcoded, tolerance in docstring.
- Do not start a new phase until the previous phase's log is committed.
- Part II prompts for phases 21–23 will be written after Phase 20 lands (periodic
  integral infrastructure must be understood before scoping downstream phases).
