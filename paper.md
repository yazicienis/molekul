---
title: 'MOLEKUL: A Pure-Python Ab Initio Quantum Chemistry Code for Education and Reproducible Benchmarking'
tags:
  - Python
  - quantum chemistry
  - Hartree-Fock
  - MP2
  - electronic structure
  - Gaussian basis sets
  - McMurchie-Davidson
authors:
  - name: Enis Yazici
    orcid: 0000-0001-7573-6344
    affiliation: 1
affiliations:
  - name: SRH University of Applied Sciences Heidelberg, School of Engineering and Architecture, Heidelberg, Germany
    index: 1
date: 24 April 2026
bibliography: paper.bib
---

# Summary

MOLEKUL is an open-source, pure-Python implementation of Restricted Hartree–Fock
(RHF) self-consistent field (SCF) theory for closed-shell molecules.
Starting from a molecular geometry and a contracted Gaussian basis set, MOLEKUL
evaluates all one- and two-electron integrals analytically, solves the
Roothaan–Hall equations [@roothaan1951; @hall1951] iteratively, and reports
the total RHF energy.
Built-in basis data cover H–F for STO-3G [@hehre1969], 6-31G\*
[@hehre1972; @hariharan1973], and cc-pVDZ [@dunning1989].
Additional modules provide a geometry optimizer, Mulliken and Löwdin
[@lowdin1950] population analysis, an MP2 [@moller1934] correlation-energy
layer, experimental CIS infrastructure, and numerical Hessian-based harmonic
frequency workflows.

Validated against PySCF [@sun2020] on a 14-molecule test suite spanning
H$_2$ through formaldehyde (CH$_2$O), MOLEKUL achieves RHF/STO-3G total-energy
agreement within $5 \times 10^{-8}$ hartree for every molecule using fixed
default SCF settings.

# State of the Field

Production electronic-structure packages such as PySCF [@sun2020], Psi4
[@smith2020], and ORCA [@neese2020] provide broad functionality and optimized
implementations.  Their strength is production research, not necessarily
transparent step-by-step pedagogy.  Earlier educational codes such as PyQuante
[@quante2004] demonstrated the value of readable implementations but are no
longer currently maintained or systematically validated against modern
reference packages.  MOLEKUL occupies this intermediate space: small in scope,
pure Python, and continuously tested against PySCF.

# Statement of Need

MOLEKUL is intended for advanced undergraduate and graduate students,
instructors, and researchers who need a readable reference implementation of
closed-shell ab initio methods rather than a production-scale engine.
Szabo and Ostlund's textbook [@szabo1989] provides the theoretical foundation,
yet the gap between a textbook derivation and a program that reproduces
published energies is substantial.  MOLEKUL bridges this gap: every
algorithmic step traces directly to a named Python function documented in the
repository, and every numerical result is compared automatically against PySCF
for the core RHF and MP2 features.

Beyond pedagogy, the pure NumPy [@harris2020] implementation provides a
self-contained baseline for hardware benchmarking, because performance profiles
isolate NumPy/BLAS behaviour without hidden compiled-kernel overhead.

# Software Design

MOLEKUL is intentionally a compact pure-Python/NumPy codebase.  The design
prioritises inspectability and reproducibility over production-scale
performance.  One- and two-electron integrals are evaluated analytically using
McMurchie–Davidson recursions [@mcmurchie1978; @helgaker2000], RHF energies
are obtained from a Roothaan–Hall SCF cycle with DIIS acceleration
[@pulay1980; @pulay1982] and a Superposition of Atomic Densities (SAD) initial
guess, and the MP2 layer builds directly on the same molecular-orbital
quantities.

The Boys function $F_n(x)$ [@boys1950] is evaluated using a pure-Python
combination of `math.erf` for the analytic $F_0$ expression, recurrence
relations, and small-$x$ Taylor expansions, with the branch selected to
maintain numerical stability over the argument range used in the validation
suite.  No SciPy dependency is required; a dedicated test confirms relative
errors below $5 \times 10^{-14}$ for the $n$ and $x$ values encountered in
STO-3G through cc-pVDZ bases.

Two-electron repulsion integrals (ERIs) are stored in a full $N_\text{AO}^4$
double-precision array.  Dense storage is intentional: it keeps the Fock-build
loop transparent and easy to inspect.  At $N_\text{AO} = 100$ this requires
approximately 800 MB; practical use is therefore limited to small molecules and
teaching-scale basis sets.  Integral screening, density fitting, and
integral-direct algorithms are documented extension points.

Contracted Gaussian shells are explicitly renormalised to unit overlap,
correcting for cross-primitive overlap contributions that are tabulated but
commonly omitted in introductory treatments.  An optional level-shift
[@saunders1973] raises virtual-orbital energies during early SCF iterations and
is disabled once DIIS takes over.  Full algorithmic details are provided in the
repository documentation.

# Validation

MOLEKUL is tested with 606 automated tests.  The benchmark script
`scripts/benchmark_14mol.py` compares RHF/STO-3G total energies against PySCF
for 14 closed-shell molecules (H$_2$, HeH$^+$, LiH, HF, N$_2$, CO, H$_2$O,
NH$_3$, CH$_4$, BH$_3$, HCN, C$_2$H$_2$, C$_2$H$_4$, CH$_2$O).  All 14
molecules pass with maximum absolute deviation $4.9 \times 10^{-8}$ Eh
(C$_2$H$_2$), consistent with floating-point differences in integral
evaluation.  Complete per-molecule results are logged in
`outputs/logs/benchmark_14mol.json`.

MP2 [@moller1934] correlation energies are validated against PySCF for eight
STO-3G molecules spanning polar hydrides, homonuclear diatomics, and polyatomic
systems.  A representative subset is shown below; the full table is in the
repository.

| Molecule  | $E_\text{corr}^\text{PySCF}$ / Eh | $|\Delta E_\text{corr}|$ / Eh |
|-----------|----------------------------------:|------------------------------:|
| H$_2$     | $-0.01313807$                     | $8.1 \times 10^{-10}$         |
| H$_2$O    | $-0.03550283$                     | $1.4 \times 10^{-8}$          |
| N$_2$     | $-0.15419856$                     | $2.2 \times 10^{-8}$          |
| CO        | $-0.12852242$                     | $1.2 \times 10^{-7}$          |

All eight molecules agree within $2 \times 10^{-7}$ Eh.

# Known Limitations

MOLEKUL is designed for clarity; several production-grade features are
deliberately absent:

- **Memory scaling**: dense $N_\text{AO}^4$ ERI storage limits practical use
  to small molecules (see Software Design).
- **Single-threaded**: the integral engine and SCF driver use NumPy without
  explicit parallelism.
- **Closed-shell RHF only**: unrestricted (UHF) and restricted open-shell
  (ROHF) references are not supported.
- **Element coverage**: built-in basis data cover H–F only; heavier elements
  are not parametrised.
- **Numerical derivatives**: geometry optimisation and harmonic-frequency
  workflows use finite differences rather than analytic nuclear gradients.
- **Experimental modules**: CIS and frequency analysis are pedagogical
  infrastructure; they are not production-validated.
- **No integral screening, ECPs, or relativistic corrections.**

# Research Impact

MOLEKUL serves two complementary purposes.  As a teaching tool, all algorithms
trace to named functions in readable Python, making it suitable for advanced
undergraduate and graduate courses in quantum chemistry and computational
physics.  As a reproducible benchmarking baseline, the 606-test suite and
deterministic NumPy implementation allow hardware and software comparisons
(e.g.\ evaluating Numba or CuPy back-ends) without hidden compiled-kernel
overhead.  Benchmark timings and energies are logged to version-controlled JSON
files for full reproducibility.

# AI Usage Disclosure

Generative AI tools were used during software development for code drafting,
refactoring assistance, documentation support, and test-generation suggestions.
All scientific algorithms, numerical reference values, and validation criteria
were reviewed by the author.  Reference values were not modified to fit the
implementation; every reported energy difference was computed independently
from PySCF.

# Acknowledgements

The author thanks the developers of PySCF, whose reference calculations
were used throughout validation, and the EMSL Basis Set Exchange [@pritchard2019]
for tabulated basis-set parameters.

# References
