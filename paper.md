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
The library ships with built-in basis data for H–F: STO-3G [@hehre1969], 6-31G\* [@hehre1972; @hariharan1973], and cc-pVDZ [@dunning1989].  The code also
includes a geometry optimizer, Mulliken and Löwdin [@lowdin1950] population
analysis, an MP2 [@moller1934] correlation-energy layer, experimental CIS infrastructure,
and numerical Hessian-based harmonic frequency workflows.

Validated against PySCF [@sun2020] on a 14-molecule test suite spanning H$_2$
through formaldehyde (CH$_2$O), MOLEKUL achieves total-energy agreement within
$5 \times 10^{-8}$ hartree for every molecule using STO-3G, with fixed default SCF settings.

# State of the Field

Production quantum-chemistry packages such as PySCF [@sun2020], Psi4
[@smith2020], and ORCA [@neese2020] provide broad functionality and highly
optimized implementations, but their internal complexity can obscure the
connection between textbook equations and executable algorithms.
PyQuante [@quante2004] and similar educational codes demonstrate the
pedagogical value of readable implementations, but most are no longer
actively maintained or validated against modern reference data.  MOLEKUL
provides a validated, actively tested alternative: every algorithm maps
directly to a named Python function, and every numerical result is checked
automatically against PySCF.

# Statement of Need

Production quantum-chemistry packages such as PySCF [@sun2020], Psi4
[@smith2020], and ORCA [@neese2020] are indispensable research tools, but
their internal complexity can obscure the connection between textbook
equations and executable algorithms.  Szabo and Ostlund's textbook [@szabo1989]
provides the equations, yet the gap between a textbook derivation and a
working SCF program capable of reproducing published numbers is substantial.

MOLEKUL fills this gap.  Every computational step maps directly to a named
function in readable Python:

- `overlap_primitive` / `kinetic_primitive` / `nuclear_primitive` —
  one-electron integrals via McMurchie–Davidson recursion [@mcmurchie1978];
- `eri_primitive` — electron repulsion integrals (ERIs) via the same recursion
  with Boys-function evaluation [@boys1950];
- `rhf_scf` — Roothaan–Hall SCF [@roothaan1951; @hall1951] with DIIS acceleration [@pulay1980; @pulay1982]
  and a Superposition of Atomic Densities (SAD) initial guess;
- `contracted_norm` — explicit renormalization of contracted Gaussian shells,
  a subtlety omitted from many introductory treatments.

Beyond pedagogy, MOLEKUL provides a self-contained Python baseline for
hardware benchmarking.  Because the implementation relies only on Python, NumPy [@harris2020], and
the standard library, performance profiles isolate NumPy/BLAS behaviour cleanly,
making MOLEKUL a useful reference point when evaluating accelerated
back-ends (Numba, CuPy) or heterogeneous hardware.

# Implementation

## Integral Engine

One-electron and two-electron integrals are evaluated analytically using the
McMurchie–Davidson (MD) scheme [@mcmurchie1978; @helgaker2000].  The MD
scheme expresses products of Cartesian Gaussians as a linear combination of
Hermite Gaussians, allowing recurrence relations to reduce every integral to
a Boys-function call $F_n(x)$ [@boys1950].

The Boys function is evaluated using `math.erf` (exact $F_0$), upward
recurrence for $n > 0$, and a finite Taylor series in the small-$x$ regime.
The asymptotic form for large $x$ is
$F_n(x) \approx \tfrac{(2n-1)!!}{2^{n+1}} \sqrt{\pi}\, x^{-(n+1/2)}$.
Validation against high-precision reference values gives relative errors
below $5 \times 10^{-14}$ over the argument range encountered in STO-3G
through cc-pVDZ bases; no dependency on SciPy is required.

Two-electron repulsion integrals (ERIs) are stored in a full $N_\text{AO}^4$
double-precision array for simplicity; at $N_\text{AO} = 100$ this requires
approximately 800 MB before additional working arrays.  Symmetry screening and
integral-direct techniques are left as documented extension points.  For the 14
STO-3G benchmark molecules ($N_\text{AO} \le 14$) wall-clock time per SCF is
under one second on a single CPU core.

## Contracted-Shell Normalisation

STO-nG contraction coefficients are tabulated for individually normalised
primitive Gaussians.  The contracted shell $\phi = \sum_k c_k \chi_k$ is not
itself normalised when cross-primitive overlaps $\langle \chi_i | \chi_j \rangle
\ne 0$.  MOLEKUL computes the exact contracted-shell norm

$$
N_\phi = \bigl\langle \phi \big| \phi \bigr\rangle^{-1/2}
       = \Bigl(\sum_{i,j} c_i c_j N_i N_j
           S(\chi_i,\chi_j)\Bigr)^{-1/2}
$$

and multiplies every one- and two-electron matrix element by the appropriate
product of shell norms.  Without this correction, diagonal overlap matrix
elements deviate from 1 by up to $10^{-5}$ for heavy-atom STO-3G shells.

## SCF Convergence

The SCF driver implements Pulay's Direct Inversion in the Iterative Subspace
(DIIS) [@pulay1980; @pulay1982] using the error vector $\mathbf{e} =
\mathbf{FPS} - \mathbf{SPF}$ [@helgaker2000].

The initial density matrix is constructed from a Superposition of Atomic
Densities (SAD) guess: for each element the STO-3G shells are populated with
spherically averaged neutral-atom occupations (fractional for open-subshell
atoms), and the per-atom density blocks are assembled into the molecular
density.  The SAD guess proved essential for molecules with near-degenerate
frontier orbitals: using a core-Hamiltonian ($\mathbf{H}^\text{core}$) guess
for N$_2$ placed a $\pi_g$ antibonding orbital in the occupied space and
converged to a state 0.73 hartree above the correct ground state.

An optional level-shift parameter raises virtual-orbital energies during
early SCF iterations by adding

$$
\mathbf{F}^\text{LS} = \mathbf{F} + \sigma
   \bigl(\mathbf{S} - \mathbf{S}\tfrac{\mathbf{P}}{2}\mathbf{S}\bigr)
$$

to the Fock matrix [@saunders1973].  The factor $\tfrac{1}{2}$ appears because
the closed-shell AO density matrix satisfies $\mathbf{P} = 2\mathbf{C}_\text{occ}
\mathbf{C}_\text{occ}^T$.  The level-shifted Fock matrix is used during the
initial SCF iterations and disabled once the DIIS subspace is populated.

# Validation

The benchmark script `scripts/benchmark_14mol.py` compares MOLEKUL RHF/STO-3G
total energies against PySCF for a chemically diverse set of 14 closed-shell
molecules: H$_2$, HeH$^+$, LiH, HF, N$_2$, CO, H$_2$O, NH$_3$, CH$_4$,
BH$_3$, HCN, C$_2$H$_2$, C$_2$H$_4$, and CH$_2$O.  All geometries are
taken from NIST CCCBDB or optimised at the RHF/STO-3G level.

| Molecule | $|\Delta E|$ / hartree | SCF iterations |
|----------|----------------------:|---------------:|
| H$_2$    | $6.1 \times 10^{-9}$  | 3              |
| HeH$^+$  | $2.8 \times 10^{-8}$  | 18             |
| LiH      | $1.1 \times 10^{-9}$  | 11             |
| HF       | $5.8 \times 10^{-8}$  | 8              |
| N$_2$    | $2.9 \times 10^{-9}$  | 6              |
| CO       | $6.0 \times 10^{-9}$  | 10             |
| H$_2$O   | $1.9 \times 10^{-8}$  | 9              |
| NH$_3$   | $3.6 \times 10^{-9}$  | 9              |
| CH$_4$   | $1.9 \times 10^{-8}$  | 10             |
| BH$_3$   | $2.4 \times 10^{-8}$  | 9              |
| HCN      | $1.5 \times 10^{-8}$  | 11             |
| C$_2$H$_2$ | $4.9 \times 10^{-8}$ | 9             |
| C$_2$H$_4$ | $3.0 \times 10^{-8}$ | 10            |
| CH$_2$O  | $8.3 \times 10^{-9}$  | 12             |

All 14 molecules pass at a tolerance of $10^{-6}$ hartree.  The largest
discrepancy ($4.9 \times 10^{-8}$ Eh for C$_2$H$_2$) lies well below the
chemical accuracy threshold of 1 kcal mol$^{-1}$ ($1.6 \times 10^{-3}$ Eh)
and is consistent with floating-point and implementation-level differences
in integral evaluation.

## MP2 Correlation Energy

The MP2 layer (`mp2_energy`) is validated against PySCF on eight
closed-shell STO-3G molecules spanning polar hydrides, homonuclear diatomics,
and polyatomic systems.  Correlation energies agree to within
$2 \times 10^{-7}$ hartree in all cases:

| Molecule  | $E_\text{RHF}$ / Eh | $E_\text{corr}^\text{PySCF}$ / Eh | $|\Delta E_\text{corr}|$ / Eh |
|-----------|--------------------:|---------------------------------:|------------------------------:|
| H$_2$     | $-1.11675930$       | $-0.01313807$                    | $8.1 \times 10^{-10}$         |
| HF        | $-98.57078004$      | $-0.01734432$                    | $2.6 \times 10^{-9}$          |
| H$_2$O    | $-74.96294667$      | $-0.03550283$                    | $1.4 \times 10^{-8}$          |
| N$_2$     | $-107.49597501$     | $-0.15419856$                    | $2.2 \times 10^{-8}$          |
| CO        | $-111.22455869$     | $-0.12852242$                    | $1.2 \times 10^{-7}$          |
| NH$_3$    | $-55.45379915$      | $-0.04704176$                    | $5.6 \times 10^{-9}$          |
| CH$_4$    | $-39.72672422$      | $-0.05650741$                    | $6.1 \times 10^{-9}$          |
| HCN       | $-91.65072755$      | $-0.12997898$                    | $5.4 \times 10^{-9}$          |

# Known Limitations

MOLEKUL is designed for clarity and pedagogical use; several production-grade
features are deliberately absent:

- **Memory scaling**: all two-electron repulsion integrals are stored in a
  dense $N_\text{AO}^4$ double-precision array.  At $N_\text{AO} = 100$ this
  requires approximately 800 MB; practical calculations are therefore limited
  to small molecules and teaching-scale basis sets.  Integral-direct or
  density-fitting techniques are required for larger systems.
- **Single-threaded execution**: the integral engine and SCF driver use NumPy
  without explicit parallelism.  Shared-memory or distributed parallelisation
  is left as a future extension.
- **Closed-shell only**: the SCF driver implements RHF; unrestricted
  (UHF) and restricted open-shell (ROHF) references are not supported.
- **Basis set coverage**: only STO-3G, 6-31G\*, and cc-pVDZ are built in.
  Heavier elements (beyond F, atomic number 9) are not parametrised.
- **Post-HF layer**: the MP2 and CIS implementations are single-pass
  reference implementations validated against PySCF; they lack frozen-core
  options, density-fitting acceleration, or excited-state geometry
  optimisation.
- **No integral screening**: Schwarz or density-based screening is not
  applied; every shell quartet is evaluated, which limits practical
  applicability to small molecules.

# Research Impact

MOLEKUL serves two complementary purposes.  First, as a teaching tool: all
algorithms are traceable to named functions in readable Python, making it
suitable for graduate courses in quantum chemistry and computational physics.
Second, as a reproducible benchmarking baseline: the 606-test suite and
deterministic NumPy implementation allow hardware and software comparisons
(e.g.\ evaluating Numba or CuPy acceleration) without hidden compiled-kernel
overhead.  The benchmark suite covers 14 molecules and three basis sets; all
timings and energies are logged to version-controlled JSON files.

# AI Usage Disclosure

Generative AI tools were used during software development for code drafting,
refactoring assistance, documentation support, and test-generation suggestions.
All scientific algorithms, numerical reference values, and validation criteria
were reviewed by the author.  Reference values were not modified to fit the
implementation; every reported energy difference was computed independently
from PySCF.

# Acknowledgements

The author thanks the developers of PySCF, whose intor routines served as the
reference throughout development, and the EMSL Basis Set Exchange [@pritchard2019]
for tabulated STO-3G parameters.

# References
