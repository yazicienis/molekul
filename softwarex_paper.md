# MOLEKUL: A Pure-Python Ab Initio Quantum Chemistry Platform for Education and Reproducible Benchmarking

**Enis Yazici**  
SRH University of Applied Sciences Heidelberg, School of Engineering and Architecture, Heidelberg, Germany  
ORCID: 0000-0001-7573-6344  
Email: yazicienis@gmail.com

---

## Code Metadata

| Nr. | Code metadata description | Details |
|-----|--------------------------|---------|
| C1 | Current code version | v0.1.1 |
| C2 | Permanent link to code/repository | https://github.com/yazicienis/molekul |
| C3 | Permanent link to Reproducible Capsule | https://doi.org/10.5281/zenodo.19743306 |
| C4 | Legal code license | MIT |
| C5 | Code versioning system | git |
| C6 | Software languages and tools | Python ≥ 3.10, NumPy |
| C7 | Dependencies | numpy, pytest (dev) |
| C8 | Developer documentation | https://github.com/yazicienis/molekul |
| C9 | Support contact | yazicienis@gmail.com |

---

## Abstract

MOLEKUL is an open-source, pure-Python implementation of Restricted Hartree–Fock (RHF) self-consistent field theory and second-order Møller–Plesset perturbation theory (MP2) for closed-shell molecules. Starting from an XYZ-format molecular geometry and a contracted Gaussian basis set, MOLEKUL evaluates all one- and two-electron integrals analytically, solves the Roothaan–Hall equations iteratively, and reports the total RHF and MP2 energies. Built-in basis data cover hydrogen through fluorine for STO-3G, 6-31G*, and cc-pVDZ. Additional modules provide a geometry optimizer, Mulliken and Löwdin population analysis, experimental excited-state infrastructure, and numerical Hessian-based harmonic frequency workflows. Validated against PySCF on a 14-molecule benchmark suite, MOLEKUL achieves RHF/STO-3G total-energy agreement within 5×10⁻⁸ hartree and MP2 correlation-energy agreement within 2×10⁻⁷ hartree. The codebase comprises 606 automated tests and is designed for advanced undergraduate and graduate courses in quantum chemistry, as well as for reproducible hardware benchmarking.

**Keywords:** quantum chemistry, Hartree-Fock, MP2, ab initio, Gaussian basis sets, Python, education, benchmarking

---

## 1. Motivation and Significance

Production electronic-structure packages such as PySCF [1], Psi4 [2], and ORCA [3] provide broad functionality and optimized implementations, but their internal complexity can obscure the connection between textbook equations and executable algorithms. Earlier educational codes such as PyQuante [4] demonstrated the value of readable implementations, but they are no longer as actively maintained or systematically validated against modern reference packages.

MOLEKUL occupies the space between textbook equations and production code. Every algorithmic step — from primitive integral evaluation to Fock matrix diagonalization — traces directly to a named Python function with clear correspondence to standard quantum-chemistry references [5,6]. The 606-test suite and deterministic NumPy implementation ensure that every numerical claim is automatically reproducible.

Beyond pedagogy, the Python/NumPy implementation provides a self-contained baseline for hardware benchmarking, because quantum-chemistry algorithms are implemented explicitly rather than delegated to compiled electronic-structure kernels, allowing clean isolation of NumPy/BLAS performance characteristics.

---

## 2. Software Description

### 2.1 Architecture

MOLEKUL is organized as an installable Python package (`src/molekul/`) with the following main modules:

| Module | Responsibility |
|--------|---------------|
| `molecule.py`, `atoms.py` | Molecular geometry and atomic data |
| `basis_sto3g.py`, `basis_631gstar.py`, `basis_ccpvdz.py` | Contracted Gaussian basis sets |
| `integrals.py` | One-electron integrals (overlap, kinetic, nuclear attraction, dipole) |
| `eri.py` | Two-electron repulsion integrals |
| `rhf.py` | RHF SCF driver with DIIS and SAD initial guess |
| `mp2.py` | MP2 correlation energy |
| `population.py` | Mulliken and Löwdin population analysis |
| `optimizer.py` | Geometry optimizer (numerical gradients) |
| `harmonic.py` | Harmonic frequency analysis (numerical Hessian) |

Installation: `pip install -e ".[dev]"`. Tests: `pytest tests/`.

### 2.2 Integral Engine

One-electron and two-electron integrals are evaluated analytically using the McMurchie–Davidson (MD) scheme [7]. The MD scheme expresses products of Cartesian Gaussians as a linear combination of Hermite Gaussians via expansion coefficients, reducing all integral evaluation to a Boys-function call F_n(x) = ∫₀¹ t²ⁿ exp(−xt²) dt [8].

The Boys function is evaluated using a pure-Python combination of `math.erf` for the analytic F₀ expression, recurrence relations, and small-x Taylor expansions, with the branch selected to maintain numerical stability over the argument range used in the validation suite. No SciPy dependency is required; a dedicated test confirms relative errors below 5×10⁻¹⁴ for the n and x values encountered in STO-3G through cc-pVDZ bases.

Two-electron repulsion integrals (ERIs) are stored in a full N⁴_AO double-precision array. Dense storage is intentional — it keeps the Fock-build loop transparent and inspectable. At N_AO = 100 this requires approximately 800 MB; practical use is therefore limited to small molecules and teaching-scale basis sets. Integral screening, density fitting, and integral-direct algorithms are documented extension points.

Contracted Gaussian shells are explicitly renormalized to unit overlap, correcting for cross-primitive overlap contributions that are commonly omitted in introductory treatments.

### 2.3 SCF Driver

The RHF SCF driver [9,10] implements three key convergence features:

- **SAD initial guess**: a Superposition of Atomic Densities density matrix with spherically-averaged neutral-atom occupations, avoiding wrong-state convergence traps that arise from the core-Hamiltonian guess for symmetric molecules.
- **DIIS acceleration**: Pulay's Direct Inversion in the Iterative Subspace [11,12] using the error vector e = FPS − SPF.
- **Level shift**: an optional virtual-orbital energy shift [13] that stabilizes convergence during early iterations and is disabled once the DIIS subspace is populated.

Default convergence thresholds: |ΔE| < 10⁻¹⁰ Eh and max|ΔP| < 10⁻⁸.

### 2.4 Post-HF and Analysis

The MP2 [14] correlation energy is computed from the converged RHF molecular orbitals via a four-index integral transformation. Mulliken and Löwdin [15] population analysis, electric dipole moments, and a geometry optimizer based on numerical nuclear gradients are also implemented. Excited-state and harmonic frequency analysis modules are available as experimental pedagogical infrastructure.

### 2.5 Basis Sets

Built-in basis data cover hydrogen through fluorine for three standard basis sets. Exponents and contraction coefficients are taken directly from PySCF's built-in basis-set data to ensure reproducibility.

| Basis | Reference | AO per heavy atom |
|-------|-----------|-------------------|
| STO-3G | Hehre et al. (1969) [16] | 5 |
| 6-31G* | Hehre et al. (1972) [17]; Hariharan & Pople (1973) [18] | 15 |
| cc-pVDZ | Dunning (1989) [19] | 15 |

---

## 3. Illustrative Example

```python
from molekul.molecule import Molecule
from molekul.atoms import Atom
from molekul.basis_sto3g import get_sto3g
from molekul.rhf import rhf_scf
from molekul.mp2 import mp2_energy
import numpy as np

ANG2BOHR = 1.8897259886

mol = Molecule([
    Atom('O', np.array([0.000,  0.000,  0.117]) * ANG2BOHR),
    Atom('H', np.array([0.000,  0.757, -0.469]) * ANG2BOHR),
    Atom('H', np.array([0.000, -0.757, -0.469]) * ANG2BOHR),
])

basis = get_sto3g()
rhf_result = rhf_scf(mol, basis)
print(f"RHF energy: {rhf_result.energy_total:.8f} Eh")   # −74.96258854 Eh

mp2_result = mp2_energy(mol, basis, rhf_result)
print(f"MP2 energy: {mp2_result.energy_total:.8f} Eh")   # −74.99844967 Eh
```

---

## 4. Validation

### 4.1 RHF/STO-3G — 14-Molecule Benchmark

The benchmark script `scripts/benchmark_14mol.py` compares MOLEKUL RHF/STO-3G total energies against PySCF for 14 closed-shell molecules spanning H₂ through formaldehyde (CH₂O). All 14 molecules pass with maximum absolute deviation 4.9×10⁻⁸ Eh (C₂H₂), consistent with floating-point differences in integral evaluation. Complete results are logged in `outputs/logs/benchmark_14mol.json`.

### 4.2 MP2 Correlation Energy

MP2 correlation energies are validated against PySCF for eight STO-3G molecules spanning polar hydrides, homonuclear diatomics, and polyatomic systems:

| Molecule | E_corr (PySCF) / Eh | |ΔE_corr| / Eh |
|----------|--------------------:|-------------:|
| H₂       | −0.01313807         | 8.1×10⁻¹⁰   |
| HF       | −0.01734432         | 2.6×10⁻⁹    |
| H₂O      | −0.03550283         | 1.4×10⁻⁸    |
| N₂       | −0.15419856         | 2.2×10⁻⁸    |
| CO       | −0.12852242         | 1.2×10⁻⁷    |
| NH₃      | −0.04704176         | 5.6×10⁻⁹    |
| CH₄      | −0.05650741         | 6.1×10⁻⁹    |
| HCN      | −0.12997898         | 5.4×10⁻⁹    |

All eight molecules agree within 2×10⁻⁷ Eh.

---

## 5. Known Limitations

- **Memory**: dense N⁴_AO ERI storage limits practical use to small molecules (N_AO ≲ 50).
- **Single-threaded**: no explicit parallelism; single NumPy/BLAS thread by default.
- **Closed-shell RHF only**: UHF and ROHF are not supported.
- **Element coverage**: H–F only; heavier elements are not parametrized.
- **Numerical derivatives**: geometry optimization and harmonic-frequency workflows use finite differences, not analytic gradients.
- **Experimental modules**: excited-state and frequency-analysis workflows are pedagogical infrastructure, not production-validated.
- **No integral screening, ECPs, or relativistic corrections.**

---

## 6. Impact

MOLEKUL serves two complementary purposes. As a teaching tool, all algorithms trace to named functions in readable Python, making it suitable for advanced undergraduate and graduate courses in quantum chemistry and computational physics. As a reproducible benchmarking baseline, the 606-test suite and deterministic NumPy implementation allow hardware and software comparisons without delegating quantum-chemistry algorithms to compiled electronic-structure kernels. Benchmark timings and energies are logged to version-controlled JSON files for full reproducibility.

---

## 7. Software Availability

- **Repository:** https://github.com/yazicienis/molekul
- **Archive DOI:** https://doi.org/10.5281/zenodo.19743306
- **License:** MIT
- **Version:** v0.1.1
- **Installation:** `pip install -e ".[dev]"`
- **Tests:** `pytest tests/`

---

## AI Usage Disclosure

Generative AI tools, including Claude (Anthropic) and ChatGPT (OpenAI), were used during software development for code drafting, refactoring assistance, documentation support, test-generation suggestions, and editorial review of the manuscript. All scientific algorithms, numerical reference values, validation criteria, and reported results were reviewed and verified by the author, who made the primary design decisions and remains responsible for the correctness, originality, and scientific content of the submission. Reference values were not modified to fit the implementation.

---

## Acknowledgements

The author thanks the developers of PySCF, whose reference calculations were used throughout validation, and the EMSL Basis Set Exchange [20] for tabulated basis-set parameters.

---

## References

1. Sun, Q., et al. (2020). Recent developments in the PySCF program package. *J. Chem. Phys.*, 153, 024109. https://doi.org/10.1063/5.0006074
2. Smith, D. G. A., et al. (2020). PSI4 1.4: Open-source software for high-throughput quantum chemistry. *J. Chem. Phys.*, 152, 184108. https://doi.org/10.1063/5.0006002
3. Neese, F., et al. (2020). The ORCA quantum chemistry program package. *J. Chem. Phys.*, 152, 224108. https://doi.org/10.1063/5.0004608
4. Muller, R. P. (2004). PyQuante: Python Quantum Chemistry. http://pyquante.sourceforge.net (accessed 2026-04-24).
5. Szabo, A., & Ostlund, N. S. (1989). *Modern Quantum Chemistry*. McGraw-Hill, New York.
6. Helgaker, T., Jørgensen, P., & Olsen, J. (2000). *Molecular Electronic-Structure Theory*. Wiley, Chichester. https://doi.org/10.1002/9781119019572
7. McMurchie, L. E., & Davidson, E. R. (1978). One- and two-electron integrals over Cartesian Gaussian functions. *J. Comput. Phys.*, 26, 218–231. https://doi.org/10.1016/0021-9991(78)90092-X
8. Boys, S. F. (1950). Electronic wave functions. I. *Proc. R. Soc. Lond. A*, 200, 542–554. https://doi.org/10.1098/rspa.1950.0036
9. Roothaan, C. C. J. (1951). New developments in molecular orbital theory. *Rev. Mod. Phys.*, 23, 69–89. https://doi.org/10.1103/RevModPhys.23.69
10. Hall, G. G. (1951). The molecular orbital theory of chemical valency. VIII. *Proc. R. Soc. Lond. A*, 205, 541–552. https://doi.org/10.1098/rspa.1951.0048
11. Pulay, P. (1980). Convergence acceleration of iterative sequences. *Chem. Phys. Lett.*, 73, 393–398. https://doi.org/10.1016/0009-2614(80)80396-4
12. Pulay, P. (1982). Improved SCF convergence acceleration. *J. Comput. Chem.*, 3, 556–560. https://doi.org/10.1002/jcc.540030413
13. Saunders, V. R., & Hillier, I. H. (1973). A level-shifting method for converging closed shell Hartree-Fock wave functions. *Int. J. Quantum Chem.*, 7, 699–705. https://doi.org/10.1002/qua.560070407
14. Møller, C., & Plesset, M. S. (1934). Note on an approximation treatment for many-electron systems. *Phys. Rev.*, 46, 618–622. https://doi.org/10.1103/PhysRev.46.618
15. Löwdin, P.-O. (1950). On the non-orthogonality problem connected with the use of atomic wave functions in the theory of molecules and crystals. *J. Chem. Phys.*, 18, 365–375. https://doi.org/10.1063/1.1747632
16. Hehre, W. J., Stewart, R. F., & Pople, J. A. (1969). Self-consistent molecular-orbital methods. I. *J. Chem. Phys.*, 51, 2657–2664. https://doi.org/10.1063/1.1672392
17. Hehre, W. J., Ditchfield, R., & Pople, J. A. (1972). Self-consistent molecular orbital methods. XII. *J. Chem. Phys.*, 56, 2257–2261. https://doi.org/10.1063/1.1677527
18. Hariharan, P. C., & Pople, J. A. (1973). The influence of polarization functions on molecular orbital hydrogenation energies. *Theor. Chim. Acta*, 28, 213–222. https://doi.org/10.1007/BF00533485
19. Dunning, T. H. (1989). Gaussian basis sets for use in correlated molecular calculations. I. *J. Chem. Phys.*, 90, 1007–1023. https://doi.org/10.1063/1.456153
20. Pritchard, B. P., et al. (2019). New Basis Set Exchange. *J. Chem. Inf. Model.*, 59, 4814–4820. https://doi.org/10.1021/acs.jcim.9b00725
21. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
