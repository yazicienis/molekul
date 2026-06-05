# Exercise solutions & hints

Worked answers and pointers for the exercises at the end of each notebook. Numbers
quoted were produced with MOLEKUL itself (STO-3G unless stated); your values should match
to the last digit or two. "Advanced" parts give the *approach* rather than a full
derivation — the point is for you to do the work.

> Tip: almost every exercise is "modify one input and re-run a cell you already have".
> Keep the notebook's results in memory (assign them to a variable) instead of
> recomputing — see the pattern used throughout the notebooks.

---

## 01 — Atoms, Molecules & Constants

1. **N₂ nuclear repulsion (1.098 Å).** `E_nuc = 23.6154 Ha`. It is large because
   $E_\text{nuc}=Z_AZ_B/R$ with $Z=7$: the $Z^2=49$ factor dwarfs H₂'s $Z^2=1$.
2. **Water cation H₂O⁺.** Removing one electron from closed-shell H₂O leaves an odd
   electron → doublet, **multiplicity = 2**. `Molecule(..., charge=1, multiplicity=2)`
   is accepted; `multiplicity=1` would raise (odd electron count).
3. **Inconsistent charge/multiplicity.** H₂ with `charge=0, multiplicity=2` → 2 electrons
   cannot give $S=1/2$. `Molecule.__post_init__` raises `ValueError` (electron count vs
   multiplicity mismatch).
4. **Summary table.** Loop over `mol.atoms`, print `i`, `atom.symbol`, `atom.Z`,
   `atom.coords_angstrom()`.
5. **$Z^2/R$ scaling.** Convert Å→bohr ($\times1.8897$) and check $E_\text{nuc}\approx Z^2/R$:
   H₂ `0.7151` (=1/1.398), N₂ `23.6154` (=49/2.075), F₂ `30.2280` (=81/2.679). The
   computed `nuclear_repulsion_energy()` matches $Z^2/R$ exactly — it *is* that formula.

## 02 — One-Electron Integrals

1. **Boys $F_0$.** $F_0(x)=\tfrac12\sqrt{\pi/x}\,\mathrm{erf}(\sqrt x)$. Values:
   `F0(0.01)=0.996677`, `F0(0.1)=0.967643`, `F0(1)=0.746824`, `F0(5)=0.395712`,
   `F0(20)=0.198166`. The implementation switches from the small-$x$ Taylor branch to the
   `erf` branch around $x\sim$ a few × 0.1; agreement is machine precision throughout.
2. **Largest $T$ diagonal.** The O 1s function — it has the largest exponent (tightest,
   fastest-curving function), and kinetic energy scales with the exponent.
3. **HeH⁺ S, T, V.** All 2×2; `S[0,0]=S[1,1]=1` by normalisation.
4. **Linear-dependence checker.** Flag eigenvalues of **S** below ~$10^{-6}$. At 10 Å H₂ the
   two 1s functions barely overlap, so **S** stays well-conditioned (no near-zero
   eigenvalue); near-degeneracy appears instead when atoms are pushed *together*.
5. **V decomposition (water).** The O nucleus ($Z=8$) dominates the depth of **V** —
   electrons spend most time near oxygen, consistent with its electronegativity.

## 03 — Basis Sets

1. **Counts.** STO-3G: N₂ `10`, CH₄ `9`. 6-31G\*: N₂ `30`, CH₄ `23`. cc-pVDZ: N₂ `30`,
   CH₄ `35`. (MOLEKUL uses **6 Cartesian** d-functions, so d-containing counts exceed the
   spherical-harmonic numbers.) Confirm with `len(basis.basis_functions(mol))`.
2. **O 2s vs 1s radial.** The 2s contracted function has a radial node (sign change) to stay
   orthogonal to 1s, so its profile is more oscillatory.
3. **N₂ RHF energies.** STO-3G `-107.495975`, 6-31G\* `-108.942623`, cc-pVDZ `-108.954693`
   Ha — monotonically lower (better) with the larger basis.
4. **Variational check (H₂).** Each larger basis gives a strictly lower RHF energy.
5. **Advanced — d-exponent sensitivity.** Scaling the O d-exponent by ±50% changes the
   energy by a few mHa; there is a shallow optimum near the standard value.

## 04 — Hartree-Fock SCF

1. **HeH⁺.** 2 electrons → **1 occupied** MO. RHF/STO-3G `E=-2.841838 Ha`,
   HOMO `-1.6328 Ha` → Koopmans IP `≈44.4 eV` (large: tightly bound cation — physically
   reasonable that it is hard to ionise further).
2. **`max_iter=5`.** SCF stops un-converged; the energy history is still decreasing
   (monotone, not yet oscillating) — it simply has not reached the threshold.
3. **DIIS off (`diis_start` huge).** Plain SCF needs many more iterations (tens vs a
   handful) for the same convergence; some systems oscillate without DIIS.
4. **N₂ vs H₂O gap.** N₂ has a much larger HOMO–LUMO gap (strong triple bond, no lone-pair
   HOMO) → chemically inert; H₂O's smaller gap reflects reactive lone pairs.
5. **Advanced — virial theorem.** For a converged atom, $-V/T\to2$, i.e.
   $2T+V\approx0$. Check `E_kin` and `E_pot` for Ne/STO-3G; small basis gives a small
   residual.

## 05 — Geometry Optimization

1. **H₂ PES minimum.** Parabolic fit of the three lowest scan points lands within a few
   mÅ of the gradient optimiser result (~0.71 Å for RHF/STO-3G; exp. 0.741 Å).
2. **N₂ optimisation.** RHF/STO-3G slightly *underestimates* the bond (minimal basis is
   too stiff); larger bases move toward the experimental 1.098 Å.
3. **CO₂ linear gradient.** By symmetry the x/y gradient components are ~0 at the linear
   geometry; only the symmetric stretch direction carries force off-equilibrium.
4. **Already at minimum.** Starting from the optimised H₂, the optimiser terminates in
   ~0–1 steps (gradient already below threshold).
5. **Advanced — $\mathcal{O}(h^2)$.** Plot gradient error vs $h$: it is linear in $h^2$
   until $h\lesssim10^{-4}$, where finite-precision cancellation takes over.

## 06 — Visualization & Population

1. **$\int\rho\,dV$.** Riemann sum (density × voxel volume) approaches $N_e=10$ as the
   step shrinks (0.5→0.1 bohr); coarse grids over/undershoot by a few %.
2. **HOMO of water.** A b₁ oxygen lone pair — perpendicular to the molecular plane, one
   nodal plane through O.
3. **N₂ Mulliken.** Homonuclear → **q = 0** on both atoms by symmetry.
4. **CO dipole.** STO-3G famously gets the *direction* of CO's tiny dipole wrong — a
   classic warning that minimal-basis Mulliken/dipole results are only qualitative.
5. **Advanced — deformation density.** $\Delta\rho=\rho_\text{mol}-\sum_A\rho_A^\text{atom}$
   shows charge piling up in the bonds and lone-pair regions.

## 07 — MP2 & Harmonic Frequencies

1. **HeH⁺ MP2.** `E_corr = -0.007238 Ha` — a small fraction (~0.25 %) of the HF energy
   (only 2 electrons to correlate).
2. **MP2 on the H₂ scan.** MP2 deepens the well and lengthens the equilibrium bond
   slightly relative to RHF (correlation softens the bond).
3. **N₂ modes.** A diatomic has $3N-5=1$ vibrational mode. The N–N stretch is **IR-inactive**
   (homonuclear → no dipole derivative → zero intensity).
4. **ZPE.** $\text{ZPE}=\tfrac12\sum\hbar\omega$. H₂'s very high stretch frequency gives a
   large ZPE; N₂'s heavier atoms and lower frequency give a smaller ZPE.
5. **Advanced — small denominators.** The MP2 pair energy $\sim|(ia|jb)|^2/\Delta\varepsilon$
   blows up as the gap $\to0$; a 0.1 eV gap makes a single pair term enormous → MP2 breaks
   down for near-degenerate (multireference) systems.

## 08 — Coupled Cluster

1. **HeH⁺ CCSD.** `E_corr = -0.009630 Ha`. Smaller than water's because there are fewer
   electron pairs to correlate.
2. **$|E_\text{MP2}|<|E_\text{CCSD}|$.** True for every molecule here — CCSD resums MP2 plus
   higher-order terms (HeH⁺: MP2 `-0.00724` vs CCSD `-0.00963`).
3. **T1 vs bond length (H₂).** T1 grows as the bond stretches; it crosses ~0.02 well before
   3.0 Å, signalling the onset of multireference character (RHF breaks down on dissociation).
4. **Size consistency.** CCSD of two H atoms 20 Å apart ≈ $2\times E_\text{CCSD}(\text H)$
   to high precision — the exponential ansatz is size-consistent (CISD is not).
5. **Advanced — (T) magnitude.** For H₂/STO-3G the (T) correction is **much smaller** than
   the CCSD correction (CCSD is already exact for a 2-electron system, so triples ≈ 0).

## 09 — Density Functional Theory

1. **LDA/PBE vs HF on N₂.** DFT total energies are *not* variationally bounded by HF
   (different Hamiltonians), so they can lie below HF — DFT includes (approximate)
   correlation that HF lacks.
2. **HOMO vs experiment.** All of HF/LDA/PBE underestimate the IP via the HOMO in this
   minimal basis; HF (Koopmans) is usually closest for water here, but none is quantitative.
3. **XC integral check.** Summing $\rho\,\varepsilon_{xc}$ over the grid reproduces
   `energy_xc` to grid accuracy (~$10^{-5}$ for LDA).
4. **Grid convergence.** `∫ρ ≈ N_e` to $<0.01$ already around `n_rad≈50, n_ang≈194`;
   the default 75×302 is comfortably converged for the density norm.
5. **Advanced — H₂ bond.** LDA over-binds (shorter bond) while HF under-binds; the LDA
   length is typically closer to the experimental 0.741 Å.

## 10 — Excited States

1. **H₂O⁺ UHF.** Doublet; $\langle S^2\rangle\gtrsim0.75$. Spin contamination is usually
   modest for the cation, comparable to or smaller than OH.
2. **CIS overestimation.** CIS lacks excited-state correlation; its first singlet of water
   sits ~1 eV (often more) above the EOM-CCSD value.
3. **Dark states.** States with oscillator strength `f≈0` are symmetry-forbidden
   (the transition dipole integral vanishes by symmetry).
4. **Basis dependence (EOM-CCSD).** Excitation energies drop noticeably going STO-3G→6-31G\*,
   especially for Rydberg-like states that need diffuse functions.
5. **Advanced — OH SOMO.** Compare the UHF SOMO energy of OH to the RHF HOMO of the closed-shell
   cation; $-\varepsilon_\text{SOMO}$ is the Koopmans ionisation estimate of the radical.

## 11 — Analytical Gradients & GPU

1. **Newton's third law.** $\sum_A \nabla_A E = 0$ (translational invariance); the per-atom
   gradient components sum to ~0 to gradient precision.
2. **Homonuclear symmetry.** At equilibrium the force on each atom is zero; off-equilibrium
   the two forces are equal and opposite along the bond axis.
3. **Zero crossing.** The bond-axis gradient component of H₂ changes sign at the equilibrium
   bond length — that sign change *is* the minimum.
4. **GPU crossover.** For tiny systems (N≲50) data-transfer overhead dominates, so CPU wins;
   GPU pays off only once the Fock build / einsum cost outgrows the transfer (much larger N).
5. **Advanced — overlap derivative.** $\partial\phi_\mu/\partial R_A=-\partial\phi_\mu/\partial r$;
   differentiating the McMurchie-Davidson overlap gives $\partial S_{\mu\nu}/\partial R$, which
   matches the finite-difference result to ~$10^{-7}$.

## 12 — Periodic Systems & Phonons

1. **Bandwidth vs $a$.** The bonding–antibonding splitting shrinks as $a$ grows (orbital
   overlap, hence hopping $t$, decays roughly exponentially with separation).
2. **Van Hove singularities.** In 1D the DOS diverges as $1/\sqrt{E}$ at the band edges
   ($\Gamma$ and $X$) — those are the peaks; their positions are the band extrema.
3. **LiH occupied bands.** `n_occ = n_electrons//2`. With STO-3G, Li+H give a small number of
   basis functions per cell; the count of occupied bands equals $N_e/2$ — verify with
   `len(STO3G.basis_functions(lih_chain))`.
4. **Acoustic sum rule.** The acoustic phonon must satisfy $\omega(\Gamma)=0$ (uniform
   translation costs no energy); it comes out at ~0 to numerical precision. If not enforced,
   finite-difference noise leaves a small spurious nonzero (or imaginary) $\omega$.
5. **Advanced — phonon vs $a$.** Stretching the chain lowers the force constant
   $k\sim d^2E/da^2$, so the zone-boundary phonon frequency *decreases* with larger $a$.
