#!/usr/bin/env python3
"""
scripts/validate_population.py — Phase 7: Mulliken population analysis
and dipole moment for H2O and HeH+ at RHF/STO-3G level.

Validation methodology
-----------------------
Checks are split into two tiers:

  EXACT   — internal invariants that must hold to machine precision:
              • tr(PS) = N_elec  (electron count conservation)
              • Σq_A  = mol.charge  (charge conservation)
              • μ_x = μ_y = 0 for H2O in yz-plane  (C2v symmetry)
              • μ_x = μ_y = 0 for HeH+ on z-axis   (axial symmetry)
              These do NOT depend on any external reference.

  PYSCF   — comparison against an independently computed PySCF/STO-3G
              reference at the same geometry.  If PySCF is not installed,
              these checks are SKIPPED and marked "qualitative only".
              Reference values are never adjusted to match MOLEKUL output.

External reference (Szabo & Ostlund Table 3.6) is used for gross orbital
populations.  Tolerances allow for the slight geometry difference between
the S&O geometry (R=1.8698 bohr, θ=100.02°) and the one used here.

Usage
-----
  python scripts/validate_population.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from molekul.atoms       import Atom
from molekul.molecule    import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf         import rhf_scf
from molekul.population  import analyze, DEBYE_PER_AU
from molekul.logging_utils import ExperimentLogger

REPO = Path(__file__).resolve().parents[1]
SEP  = "─" * 64


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _make_h2o() -> Molecule:
    """H2O at near-equilibrium STO-3G geometry (O at origin, yz-plane)."""
    R, th = 1.870, np.radians(50.0)   # θ/2 = 50° → θ(HOH) = 100°
    Hx = R * np.sin(th)
    Hz = -R * np.cos(th)
    return Molecule(
        atoms=[
            Atom("O", [0.0,  0.0,  0.0]),
            Atom("H", [0.0,  Hx,   Hz]),
            Atom("H", [0.0, -Hx,   Hz]),
        ],
        charge=0, multiplicity=1, name="H2O",
    )


def _make_heh() -> Molecule:
    """HeH+ at R=1.4632 bohr (He at origin, H along +z)."""
    return Molecule(
        atoms=[
            Atom("He", [0.0, 0.0, 0.0    ]),
            Atom("H",  [0.0, 0.0, 1.4632 ]),
        ],
        charge=1, multiplicity=1, name="HeH+",
    )


# ---------------------------------------------------------------------------
# PySCF reference values — computed independently, NOT from MOLEKUL output
# ---------------------------------------------------------------------------

def _get_pyscf_refs(mol_h2o: Molecule, mol_heh: Molecule) -> dict | None:
    """
    Compute PySCF/STO-3G reference values at the exact same geometries.

    Returns a dict with keys:
        h2o_dipole_d, h2o_q_O, h2o_q_H
        heh_dipole_d, heh_q_He, heh_q_H
    Returns None if PySCF is not installed.
    """
    try:
        from pyscf import gto, scf as pyscf_scf
    except ImportError:
        return None

    def _to_pyscf_str(mol: Molecule) -> str:
        return "; ".join(
            f"{a.symbol} {a.coords[0]:.15f} {a.coords[1]:.15f} {a.coords[2]:.15f}"
            for a in mol.atoms
        )

    # H2O
    pm_h2o = gto.M(
        atom=_to_pyscf_str(mol_h2o), basis="sto-3g",
        unit="Bohr", charge=0, spin=0, verbose=0,
    )
    mf_h2o = pyscf_scf.RHF(pm_h2o).run()
    mull_h2o = mf_h2o.mulliken_pop(verbose=0)
    dip_h2o  = mf_h2o.dip_moment(unit="Debye", verbose=0)

    # HeH+
    pm_heh = gto.M(
        atom=_to_pyscf_str(mol_heh), basis="sto-3g",
        unit="Bohr", charge=1, spin=0, verbose=0,
    )
    mf_heh = pyscf_scf.RHF(pm_heh).run()
    mull_heh = mf_heh.mulliken_pop(verbose=0)
    dip_heh  = mf_heh.dip_moment(unit="Debye", verbose=0)

    return {
        "h2o_dipole_d": float(np.linalg.norm(dip_h2o)),
        "h2o_q_O":      float(mull_h2o[1][0]),
        "h2o_q_H":      float(mull_h2o[1][1]),
        "heh_dipole_d": float(np.linalg.norm(dip_heh)),
        "heh_q_He":     float(mull_heh[1][0]),
        "heh_q_H":      float(mull_heh[1][1]),
        "h2o_scf_energy": float(mf_h2o.e_tot),
        "heh_scf_energy": float(mf_heh.e_tot),
    }


# ---------------------------------------------------------------------------
# Szabo & Ostlund Table 3.6 gross orbital populations (H2O, nearby geometry)
# ---------------------------------------------------------------------------

_SO_LABELS = ["O 1s", "O 2s", "O 2px", "O 2py", "O 2pz", "H1 1s", "H2 1s"]
_SO_VALUES = [1.9975, 1.8401,  2.0000,   1.0733,   1.4108,   0.8392,   0.8392]
_SO_TOL    = 0.02   # geometry difference between S&O (R=1.8698, θ=100.02°) and ours


# ---------------------------------------------------------------------------
# H2O validation
# ---------------------------------------------------------------------------

def run_h2o(mol: Molecule, log: ExperimentLogger,
            pyscf_refs: dict | None) -> bool:
    tag = "h2o"
    print(f"\n{SEP}")
    print("  H2O   (10 electrons, neutral, STO-3G)")
    print(SEP)

    t0 = time.time()
    result = rhf_scf(mol, STO3G, verbose=False)
    print(f"  RHF: {result.converged}, {result.n_iter} iter, "
          f"E = {result.energy_total:.10f} Ha  ({time.time()-t0:.2f}s)")
    log.metric(f"{tag}_scf_energy_ha", result.energy_total)

    pop  = analyze(mol, STO3G, result)
    mull = pop.mulliken
    dip  = pop.dipole

    # ── Orbital populations table ──────────────────────────────────────────
    print(f"\n  Gross orbital populations (vs Szabo & Ostlund Table 3.6):")
    print(f"  {'Label':<14} {'S&O ref':>10} {'MOLEKUL':>10} {'err':>10}")
    for lbl, ref, (orb_lbl, n_mu) in zip(
            _SO_LABELS, _SO_VALUES,
            zip(mull.orbital_labels, mull.gross_orbital_pop)):
        err = n_mu - ref
        print(f"  {lbl:<14} {ref:10.4f} {n_mu:10.4f} {err:+10.4f}")

    # ── Atomic populations ─────────────────────────────────────────────────
    print(f"\n  Atomic populations and charges:")
    for sym, Z, N_A, q_A in zip(mull.atom_symbols,
                                  [a.Z for a in mol.atoms],
                                  mull.gross_atomic_pop,
                                  mull.mulliken_charges):
        print(f"    {sym}:  Z={Z}  N={N_A:.6f}  q={q_A:+.6f}")

    # ── Dipole ─────────────────────────────────────────────────────────────
    print(f"\n  Dipole moment:")
    print(f"    Nuclear  : [{dip.nuclear_au[0]:+.4f}  {dip.nuclear_au[1]:+.4f}  "
          f"{dip.nuclear_au[2]:+.4f}] ea₀")
    print(f"    Electron : [{dip.electronic_au[0]:+.4f}  {dip.electronic_au[1]:+.4f}  "
          f"{dip.electronic_au[2]:+.4f}] ea₀")
    print(f"    Total    : [{dip.total_debye[0]:+.4f}  {dip.total_debye[1]:+.4f}  "
          f"{dip.total_debye[2]:+.4f}] D")
    print(f"    |μ|      : {dip.magnitude_debye:.6f} D")
    if pyscf_refs:
        print(f"    PySCF ref: {pyscf_refs['h2o_dipole_d']:.6f} D")

    log.metric(f"{tag}_dipole_magnitude_debye", dip.magnitude_debye)
    log.metric(f"{tag}_dipole_z_debye",          float(dip.total_debye[2]))
    log.metric(f"{tag}_q_O",    float(mull.mulliken_charges[0]))
    log.metric(f"{tag}_q_H",    float(mull.mulliken_charges[1]))
    log.metric(f"{tag}_total_electrons", mull.total_electrons)

    all_ok = True

    # ── EXACT checks: internal invariants ─────────────────────────────────
    ok = abs(mull.total_electrons - mol.n_electrons) < 1e-8
    log.check("H2O [EXACT]: tr(PS) = 10 electrons",
               ok, f"got {mull.total_electrons:.10f}")
    all_ok &= ok

    ok = abs(float(mull.mulliken_charges.sum())) < 1e-8
    log.check("H2O [EXACT]: Σq_A = 0 (neutral)",
               ok, f"Σq = {float(mull.mulliken_charges.sum()):.2e}")
    all_ok &= ok

    ok = abs(dip.total_debye[0]) < 1e-6
    log.check("H2O [EXACT]: μ_x = 0 (molecule in yz-plane)",
               ok, f"μ_x = {dip.total_debye[0]:.2e} D")
    all_ok &= ok

    ok = abs(dip.total_debye[1]) < 1e-6
    log.check("H2O [EXACT]: μ_y = 0 (C2v symmetry)",
               ok, f"μ_y = {dip.total_debye[1]:.2e} D")
    all_ok &= ok

    # ── Szabo & Ostlund orbital populations ───────────────────────────────
    for ref_lbl, ref_val, n_mu in zip(_SO_LABELS, _SO_VALUES,
                                       mull.gross_orbital_pop):
        ok = abs(n_mu - ref_val) < _SO_TOL
        log.check(f"H2O [S&O]: N({ref_lbl}) within {_SO_TOL} of Szabo&Ostlund",
                   ok, f"got {n_mu:.4f}, S&O {ref_val:.4f}")
        all_ok &= ok

    # ── PYSCF checks: only when PySCF reference is available ──────────────
    if pyscf_refs:
        tol_dip  = 1e-4   # D  — identical code path, expect near-exact match
        tol_chg  = 1e-4   # e

        ok = abs(dip.magnitude_debye - pyscf_refs["h2o_dipole_d"]) < tol_dip
        log.check(f"H2O [PySCF]: |μ| within {tol_dip:.0e} D",
                   ok, f"MOLEKUL={dip.magnitude_debye:.6f}, PySCF={pyscf_refs['h2o_dipole_d']:.6f}")
        all_ok &= ok

        ok = abs(float(mull.mulliken_charges[0]) - pyscf_refs["h2o_q_O"]) < tol_chg
        log.check(f"H2O [PySCF]: q(O) within {tol_chg:.0e} e",
                   ok, f"MOLEKUL={float(mull.mulliken_charges[0]):+.6f}, PySCF={pyscf_refs['h2o_q_O']:+.6f}")
        all_ok &= ok

        ok = abs(float(mull.mulliken_charges[1]) - pyscf_refs["h2o_q_H"]) < tol_chg
        log.check(f"H2O [PySCF]: q(H) within {tol_chg:.0e} e",
                   ok, f"MOLEKUL={float(mull.mulliken_charges[1]):+.6f}, PySCF={pyscf_refs['h2o_q_H']:+.6f}")
        all_ok &= ok

        ok = abs(result.energy_total - pyscf_refs["h2o_scf_energy"]) < 1e-6
        log.check("H2O [PySCF]: SCF energy agrees within 1e-6 Ha",
                   ok, f"MOLEKUL={result.energy_total:.10f}, PySCF={pyscf_refs['h2o_scf_energy']:.10f}")
        all_ok &= ok
    else:
        print("\n  [PySCF not available — dipole/charge comparisons are qualitative only]")
        log.check("H2O [PySCF]: SKIPPED — PySCF not installed", True, "qualitative only")

    return all_ok


# ---------------------------------------------------------------------------
# HeH+ validation
# ---------------------------------------------------------------------------

def run_heh(mol: Molecule, log: ExperimentLogger,
            pyscf_refs: dict | None) -> bool:
    tag = "heh"
    print(f"\n{SEP}")
    print("  HeH+  (2 electrons, charge=+1, STO-3G)")
    print("  Note: dipole is ORIGIN-DEPENDENT for charged species.")
    print("        Here He is at the coordinate origin.")
    print(SEP)

    t0 = time.time()
    result = rhf_scf(mol, STO3G, verbose=False)
    print(f"  RHF: {result.converged}, {result.n_iter} iter, "
          f"E = {result.energy_total:.10f} Ha  ({time.time()-t0:.2f}s)")
    log.metric(f"{tag}_scf_energy_ha", result.energy_total)

    pop  = analyze(mol, STO3G, result)
    mull = pop.mulliken
    dip  = pop.dipole

    print(f"\n  Gross orbital populations:")
    for lbl, n_mu in zip(mull.orbital_labels, mull.gross_orbital_pop):
        print(f"    {lbl:<16}  {n_mu:.6f}")

    print(f"\n  Atomic populations and charges:")
    for sym, Z, N_A, q_A in zip(mull.atom_symbols,
                                  [a.Z for a in mol.atoms],
                                  mull.gross_atomic_pop,
                                  mull.mulliken_charges):
        print(f"    {sym}:  Z={Z}  N={N_A:.6f}  q={q_A:+.6f}")
    print(f"  Total electrons: {mull.total_electrons:.6f} (exact: 2)")

    print(f"\n  Dipole moment (origin: He at (0,0,0)):")
    print(f"    Nuclear  : [{dip.nuclear_au[0]:+.4f}  {dip.nuclear_au[1]:+.4f}  "
          f"{dip.nuclear_au[2]:+.4f}] ea₀")
    print(f"    Electron : [{dip.electronic_au[0]:+.4f}  {dip.electronic_au[1]:+.4f}  "
          f"{dip.electronic_au[2]:+.4f}] ea₀")
    print(f"    Total    : [{dip.total_debye[0]:+.4f}  {dip.total_debye[1]:+.4f}  "
          f"{dip.total_debye[2]:+.4f}] D")
    print(f"    |μ|      : {dip.magnitude_debye:.6f} D")
    if pyscf_refs:
        print(f"    PySCF ref: {pyscf_refs['heh_dipole_d']:.6f} D")

    log.metric(f"{tag}_dipole_magnitude_debye", dip.magnitude_debye)
    log.metric(f"{tag}_dipole_z_debye",          float(dip.total_debye[2]))
    log.metric(f"{tag}_q_He",   float(mull.mulliken_charges[0]))
    log.metric(f"{tag}_q_H",    float(mull.mulliken_charges[1]))
    log.metric(f"{tag}_total_electrons", mull.total_electrons)

    all_ok = True

    # ── EXACT checks ──────────────────────────────────────────────────────
    ok = abs(mull.total_electrons - mol.n_electrons) < 1e-8
    log.check("HeH+ [EXACT]: tr(PS) = 2 electrons",
               ok, f"got {mull.total_electrons:.10f}")
    all_ok &= ok

    ok = abs(float(mull.mulliken_charges.sum()) - mol.charge) < 1e-8
    log.check("HeH+ [EXACT]: Σq_A = +1 (charge conservation)",
               ok, f"Σq = {float(mull.mulliken_charges.sum()):.8f}")
    all_ok &= ok

    ok = abs(dip.total_debye[0]) < 1e-8 and abs(dip.total_debye[1]) < 1e-8
    log.check("HeH+ [EXACT]: μ_x = μ_y = 0 (linear molecule on z-axis)",
               ok, f"μ_x={dip.total_debye[0]:.2e}, μ_y={dip.total_debye[1]:.2e}")
    all_ok &= ok

    # Physical sanity: He must hold more electron density than H in HeH+
    ok = mull.gross_atomic_pop[0] > mull.gross_atomic_pop[1]
    log.check("HeH+ [PHYSICAL]: N(He) > N(H)  (He more electronegative)",
               ok, f"N(He)={mull.gross_atomic_pop[0]:.4f}, N(H)={mull.gross_atomic_pop[1]:.4f}")
    all_ok &= ok

    # ── PySCF checks ──────────────────────────────────────────────────────
    if pyscf_refs:
        tol_dip = 1e-4
        tol_chg = 1e-4

        ok = abs(dip.magnitude_debye - pyscf_refs["heh_dipole_d"]) < tol_dip
        log.check(f"HeH+ [PySCF]: |μ| within {tol_dip:.0e} D (He at origin)",
                   ok, f"MOLEKUL={dip.magnitude_debye:.6f}, PySCF={pyscf_refs['heh_dipole_d']:.6f}")
        all_ok &= ok

        ok = abs(float(mull.mulliken_charges[0]) - pyscf_refs["heh_q_He"]) < tol_chg
        log.check(f"HeH+ [PySCF]: q(He) within {tol_chg:.0e} e",
                   ok, f"MOLEKUL={float(mull.mulliken_charges[0]):+.6f}, PySCF={pyscf_refs['heh_q_He']:+.6f}")
        all_ok &= ok

        ok = abs(float(mull.mulliken_charges[1]) - pyscf_refs["heh_q_H"]) < tol_chg
        log.check(f"HeH+ [PySCF]: q(H) within {tol_chg:.0e} e",
                   ok, f"MOLEKUL={float(mull.mulliken_charges[1]):+.6f}, PySCF={pyscf_refs['heh_q_H']:+.6f}")
        all_ok &= ok

        ok = abs(result.energy_total - pyscf_refs["heh_scf_energy"]) < 1e-6
        log.check("HeH+ [PySCF]: SCF energy agrees within 1e-6 Ha",
                   ok, f"MOLEKUL={result.energy_total:.10f}, PySCF={pyscf_refs['heh_scf_energy']:.10f}")
        all_ok &= ok
    else:
        print("\n  [PySCF not available — dipole/charge comparisons are qualitative only]")
        log.check("HeH+ [PySCF]: SKIPPED — PySCF not installed", True, "qualitative only")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    log = ExperimentLogger("phase7", "population")

    print("=" * 64)
    print("  Phase 7: Mulliken Population Analysis & Dipole Moment")
    print("  RHF/STO-3G  —  H2O and HeH+")
    print("=" * 64)

    mol_h2o = _make_h2o()
    mol_heh = _make_heh()

    # Get PySCF references independently (not from MOLEKUL output)
    print("\n  Computing PySCF/STO-3G reference values ...")
    t_ref = time.time()
    pyscf_refs = _get_pyscf_refs(mol_h2o, mol_heh)
    if pyscf_refs:
        print(f"  PySCF references ready ({time.time()-t_ref:.2f}s)")
        print(f"    H2O: |μ|={pyscf_refs['h2o_dipole_d']:.6f} D, "
              f"q(O)={pyscf_refs['h2o_q_O']:+.6f}, q(H)={pyscf_refs['h2o_q_H']:+.6f}")
        print(f"    HeH+: |μ|={pyscf_refs['heh_dipole_d']:.6f} D, "
              f"q(He)={pyscf_refs['heh_q_He']:+.6f}, q(H)={pyscf_refs['heh_q_H']:+.6f}")
        log.metric("pyscf_available", True)
    else:
        print("  PySCF not available — quantitative comparisons will be skipped.")
        log.metric("pyscf_available", False)

    h2o_ok = run_h2o(mol_h2o, log, pyscf_refs)
    heh_ok = run_heh(mol_heh, log, pyscf_refs)

    dt = time.time() - t_start
    log.metric("elapsed_s", dt)

    print(f"\n{SEP}")
    print("  Known limitations of Mulliken analysis")
    print(SEP)
    print("""\
  1. Basis-set dependence: STO-3G charges are qualitative only. Charges
     change by 0.1-0.3 e between minimal and triple-zeta basis sets.
  2. Not a physical observable: Mulliken charges have no quantum-mechanical
     operator; they depend on the arbitrary choice of basis set centre.
  3. Diffuse basis failure: aug-cc-pVDZ and larger augmented bases can give
     Mulliken populations outside [0, Z_A] — the partitioning breaks down.
  4. Origin dependence (charged molecules): for ions (e.g. HeH+), the dipole
     moment depends on the choice of coordinate origin.
  5. Preferred alternatives: Natural Population Analysis (NPA/NBO),
     QTAIM (Bader atoms-in-molecules), Hirshfeld/CM5 charges.
  6. Dipole moment IS a physical observable for neutral molecules and is
     correctly computed here via analytic dipole integrals.""")

    print(f"\n  Total time: {dt:.2f}s")

    txt_path, json_path = log.save()
    print(f"  Log written: {txt_path.name},  {json_path.name}")

    if h2o_ok and heh_ok:
        print("\n  All checks passed.")
    else:
        print("\n  SOME CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
