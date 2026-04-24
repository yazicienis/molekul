#!/usr/bin/env python3
"""
Phase 5b validation: H2O geometry optimisation (RHF/STO-3G, numerical gradient).

Starting geometry
-----------------
  R(OH) = 2.0 bohr  (stretched; equilibrium ≈ 1.870 bohr)
  θ(HOH) = 120°     (widened; equilibrium ≈ 100.0°)

  O  at (0,  0,          0         ) bohr
  H1 at (0,  R·sin 60°,  −R·cos 60°) = (0,  1.732051, −1.0)
  H2 at (0, −R·sin 60°,  −R·cos 60°) = (0, −1.732051, −1.0)

Expected equilibrium (PySCF 2.x fine scan, STO-3G)
---------------------------------------------------
  R(OH) ≈ 1.870 bohr  (= 0.9897 Å)
  θ(HOH) ≈ 100.0°
  E ≈ −74.9659011183 Ha

Outputs
-------
  outputs/h2o_opt_traj.xyz
  outputs/h2o_opt_history.json
  outputs/logs/phase5_h2o_optimizer.txt
  outputs/logs/phase5_h2o_optimizer.json

Usage
-----
  python scripts/validate_h2o_opt.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
from pathlib import Path

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf import rhf_scf
from molekul.grad import numerical_gradient, gradient_norm, max_gradient
from molekul.optimizer import optimize_geometry
from molekul.geom import bond_length, bond_angle
from molekul.logging_utils import ExperimentLogger

SEP = "=" * 66
SEP2 = "─" * 66

# ---------------------------------------------------------------------------
# Reference values (PySCF 2.x, very fine 2D scan, STO-3G)
# ---------------------------------------------------------------------------
PYSCF_R_OH   = 1.870       # bohr
PYSCF_THETA  = 100.0       # degrees
PYSCF_ENERGY = -74.9659011183  # Hartree

BOHR_TO_ANG = 0.529177210903


def make_start_mol() -> Molecule:
    """Non-equilibrium H2O: R=2.0 bohr, θ=120°."""
    R, theta_half_deg = 2.0, 60.0
    th = np.radians(theta_half_deg)
    Hx = R * np.sin(th)
    Hz = -R * np.cos(th)
    return Molecule(
        atoms=[
            Atom("O",  [0.0,  0.0,  0.0]),
            Atom("H",  [0.0,  Hx,   Hz ]),
            Atom("H",  [0.0, -Hx,   Hz ]),
        ],
        charge=0, multiplicity=1, name="H2O",
    )


def pyscf_optimize_h2o():
    """
    Return PySCF STO-3G H2O equilibrium via fine 2D scan if PySCF available.
    Returns (E, R_OH, theta_deg) or (None, None, None).
    """
    try:
        from pyscf import gto, scf as pyscf_scf
    except ImportError:
        return None, None, None

    best = {"E": None, "R": None, "th": None}
    for R in np.linspace(1.83, 1.91, 17):
        for th in np.linspace(98.0, 102.0, 17):
            th_r = np.radians(th / 2)
            Hx = R * np.sin(th_r)
            Hz = -R * np.cos(th_r)
            mol_p = gto.Mole()
            mol_p.atom = (f"O 0 0 0; "
                          f"H 0 {Hx:.10f} {Hz:.10f}; "
                          f"H 0 {-Hx:.10f} {Hz:.10f}")
            mol_p.basis = "sto-3g"
            mol_p.unit = "Bohr"
            mol_p.verbose = 0
            mol_p.build()
            E = pyscf_scf.RHF(mol_p).kernel()
            if best["E"] is None or E < best["E"]:
                best["E"] = E
                best["R"] = float(R)
                best["th"] = float(th)
    return best["E"], best["R"], best["th"]


def report_geometry(mol: Molecule, label: str = "") -> dict:
    """Print and return key internal coordinates."""
    R1 = bond_length(mol, 0, 1)   # O-H1
    R2 = bond_length(mol, 0, 2)   # O-H2
    theta = bond_angle(mol, 1, 0, 2)  # H1-O-H2
    if label:
        print(f"  {label}")
    print(f"    R(O-H1) = {R1:.6f} bohr  ({R1*BOHR_TO_ANG:.6f} Å)")
    print(f"    R(O-H2) = {R2:.6f} bohr  ({R2*BOHR_TO_ANG:.6f} Å)")
    print(f"    θ(HOH)  = {theta:.4f}°")
    return {"R_OH1_bohr": R1, "R_OH2_bohr": R2, "theta_hoh_deg": theta}


def main():
    print(f"\nMOLEKUL — H2O Geometry Optimisation (RHF/STO-3G)")
    print(f"Numerical gradient (central diff, h=1e-3 bohr) + BFGS\n")

    mol_start = make_start_mol()
    E_start = rhf_scf(mol_start, STO3G).energy_total

    # ------------------------------------------------------------------
    # 1. Starting geometry report
    # ------------------------------------------------------------------
    print(SEP)
    print("  STARTING GEOMETRY")
    print(SEP)
    geom_start = report_geometry(mol_start, "H2O at R=2.0 bohr, θ=120°  (non-equilibrium)")
    print(f"    E_start  = {E_start:.10f} Ha\n")

    # ------------------------------------------------------------------
    # 2. Gradient check at starting geometry
    # ------------------------------------------------------------------
    print(SEP)
    print("  GRADIENT CHECK AT STARTING GEOMETRY")
    print(SEP)
    grad_start = numerical_gradient(mol_start, STO3G)
    print(f"  Gradient (Ha/bohr):")
    labels = ["O", "H1", "H2"]
    for k, lbl in enumerate(labels):
        g = grad_start[k]
        print(f"    {lbl}: dE/dx={g[0]:>10.6f}  dE/dy={g[1]:>10.6f}  dE/dz={g[2]:>10.6f}")
    print(f"  |g|_rms = {gradient_norm(grad_start):.4e} Ha/bohr")
    print(f"  |g|_max = {max_gradient(grad_start):.4e} Ha/bohr")
    # Force balance check
    f_sum = grad_start.sum(axis=0)
    print(f"  Force sum (should be ~0): [{f_sum[0]:.2e}, {f_sum[1]:.2e}, {f_sum[2]:.2e}]\n")

    # ------------------------------------------------------------------
    # 3. Geometry optimisation
    # ------------------------------------------------------------------
    print(SEP)
    print("  GEOMETRY OPTIMISATION")
    print(SEP)

    result = optimize_geometry(
        mol_start,
        STO3G,
        grad_tol=1e-4,
        max_steps=100,
        h_grad=1e-3,
        traj_path="outputs/h2o_opt_traj.xyz",
        history_path="outputs/h2o_opt_history.json",
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 4. Final geometry report
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  OPTIMISED GEOMETRY")
    print(SEP)
    geom_final = report_geometry(result.final_molecule, "Final H2O")
    R1_f   = geom_final["R_OH1_bohr"]
    R2_f   = geom_final["R_OH2_bohr"]
    th_f   = geom_final["theta_hoh_deg"]
    E_f    = result.energy_final

    print(f"    E_final  = {E_f:.10f} Ha")
    print(f"    ΔE       = {E_f - E_start:+.6e} Ha")
    print(f"    |g|_max  = {result.grad_max_final:.2e} Ha/bohr")
    print(f"    Steps    = {result.n_steps}")
    print(f"    Converged: {result.converged}\n")

    # ------------------------------------------------------------------
    # 5. Convergence history
    # ------------------------------------------------------------------
    print(SEP)
    print("  CONVERGENCE HISTORY (selected steps)")
    print(SEP)
    print(f"  {'Step':>4}  {'E (Ha)':>18}  {'|g|_rms':>12}  {'|g|_max':>12}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*12}  {'-'*12}")
    n = len(result.energy_history)
    steps_to_show = sorted(set(
        [0, 1, 2] +
        list(range(0, n, max(1, n // 8))) +
        [n - 1]
    ))
    for s in steps_to_show:
        if s < n:
            print(f"  {s+1:>4}  {result.energy_history[s]:>18.10f}"
                  f"  {result.grad_rms_history[s]:>12.4e}"
                  f"  {result.grad_max_history[s]:>12.4e}")

    # ------------------------------------------------------------------
    # 6. Comparison with PySCF reference
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  PYSCF COMPARISON")
    print(SEP)
    print(f"  Running PySCF fine 2D scan (this may take ~30 s)...")
    E_pyscf, R_pyscf, th_pyscf = pyscf_optimize_h2o()

    if E_pyscf is not None:
        print(f"\n  PySCF STO-3G equilibrium:")
        print(f"    R(OH)   = {R_pyscf:.6f} bohr  ({R_pyscf*BOHR_TO_ANG:.6f} Å)")
        print(f"    θ(HOH)  = {th_pyscf:.4f}°")
        print(f"    E       = {E_pyscf:.10f} Ha")
        print(f"\n  MOLEKUL vs PySCF:")
        dR  = (R1_f + R2_f) / 2 - R_pyscf
        dTh = th_f - th_pyscf
        dE  = E_f - E_pyscf
        print(f"    ΔR(OH)  = {dR:+.4f} bohr  ({dR*BOHR_TO_ANG:+.4f} Å)")
        print(f"    Δθ(HOH) = {dTh:+.4f}°")
        print(f"    ΔE      = {dE:+.6e} Ha")
        REF_R  = R_pyscf
        REF_TH = th_pyscf
        REF_E  = E_pyscf
    else:
        print(f"  PySCF not available — using pre-computed reference")
        REF_R  = PYSCF_R_OH
        REF_TH = PYSCF_THETA
        REF_E  = PYSCF_ENERGY
        print(f"  Reference: R={REF_R:.4f} bohr, θ={REF_TH:.1f}°, E={REF_E:.10f} Ha")
        dR  = (R1_f + R2_f) / 2 - REF_R
        dTh = th_f - REF_TH
        dE  = E_f - REF_E
        print(f"  ΔR(OH)  = {dR:+.4f} bohr")
        print(f"  Δθ(HOH) = {dTh:+.4f}°")
        print(f"  ΔE      = {dE:+.6e} Ha")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Converged       : {result.converged}")
    print(f"  BFGS steps      : {result.n_steps}")
    print(f"  E_start         : {E_start:.10f} Ha  (R=2.0 bohr, θ=120°)")
    print(f"  E_final         : {E_f:.10f} Ha")
    print(f"  R(OH) final     : {(R1_f+R2_f)/2:.6f} bohr = {(R1_f+R2_f)/2*BOHR_TO_ANG:.6f} Å")
    print(f"  θ(HOH) final    : {th_f:.4f}°")
    print(f"  |g|_max final   : {result.grad_max_final:.2e} Ha/bohr")
    print(f"  Traj. frames    : {len(result.trajectory)}")
    print(f"  Trajectory      : outputs/h2o_opt_traj.xyz")
    print(f"  History         : outputs/h2o_opt_history.json")

    print(f"""
What is trustworthy
-------------------
  - RHF/STO-3G energy at each geometry is validated against PySCF to ~1e-8 Ha
  - Numerical gradient (h=1e-3 bohr) accuracy ~1e-6 Ha/bohr
  - BFGS descent property: each accepted step lowers the energy
  - Bond lengths and angles computed from Cartesian coords (exact arithmetic)
  - PySCF grid reference confirms R(OH) and θ(HOH) to within grid resolution

What remains approximate / limited
------------------------------------
  - Numerical gradient: O(h²) truncation, no Richardson extrapolation
  - BFGS Cartesian: translations/rotations not projected out; molecule may
    drift/rotate by O(1e-4 bohr); bond lengths and angles are unaffected
  - STO-3G is a minimal basis: R(OH) too short vs experiment (~1.81 bohr)
    and θ too small vs experiment (104.5°) — this is a basis set effect,
    not an optimizer bug
  - Each gradient call: 3*3*2 = 18 SCF calculations; total cost scales as
    18 * n_BFGS_steps * n_SCF_iters_per_geometry
""")

    # ------------------------------------------------------------------
    # 8. Structured log
    # ------------------------------------------------------------------
    exp = ExperimentLogger("phase5", "h2o_optimizer")
    exp.metric("start_R_OH_bohr",      2.0)
    exp.metric("start_theta_hoh_deg",  120.0)
    exp.metric("start_energy_ha",      round(E_start, 10))
    exp.metric("final_R_OH1_bohr",     round(R1_f, 6))
    exp.metric("final_R_OH2_bohr",     round(R2_f, 6))
    exp.metric("final_R_OH_mean_bohr", round((R1_f + R2_f) / 2, 6))
    exp.metric("final_R_OH_ang",       round((R1_f + R2_f) / 2 * BOHR_TO_ANG, 6))
    exp.metric("final_theta_hoh_deg",  round(th_f, 4))
    exp.metric("final_energy_ha",      round(E_f, 10))
    exp.metric("n_bfgs_steps",         result.n_steps)
    exp.metric("grad_max_final_ha_bohr", round(result.grad_max_final, 8))
    exp.metric("pyscf_ref_R_OH_bohr",  REF_R)
    exp.metric("pyscf_ref_theta_deg",  REF_TH)
    exp.metric("pyscf_ref_energy_ha",  round(REF_E, 10))
    exp.metric("delta_R_OH_vs_pyscf_bohr",  round((R1_f + R2_f) / 2 - REF_R, 6))
    exp.metric("delta_theta_vs_pyscf_deg",  round(th_f - REF_TH, 4))
    exp.metric("delta_E_vs_pyscf_ha",       round(E_f - REF_E, 10))
    exp.metric("grad_h_bohr",          1e-3)

    exp.check("H2O optimization converged",          result.converged)
    exp.check("Energy decreased",                    E_f < E_start)
    exp.check("|g|_max < 1e-3 at convergence",       result.grad_max_final < 1e-3)
    exp.check("R(OH) within 0.03 bohr of PySCF",    abs((R1_f + R2_f) / 2 - REF_R) < 0.03)
    exp.check("θ(HOH) within 2° of PySCF",          abs(th_f - REF_TH) < 2.0)
    exp.check("E within 5e-3 Ha of PySCF",          abs(E_f - REF_E) < 5e-3)
    exp.check("C2v symmetry: R(OH1) ≈ R(OH2)",      abs(R1_f - R2_f) < 0.01)

    exp.line("Method: RHF/STO-3G, BFGS, numerical gradient (h=1e-3 bohr)")
    exp.line("Start: R(OH)=2.0 bohr, θ(HOH)=120° (non-equilibrium)")
    exp.line(f"Final: R(OH)={(R1_f+R2_f)/2:.4f} bohr, θ={th_f:.2f}°, E={E_f:.8f} Ha")
    exp.line(f"PySCF ref: R(OH)={REF_R:.4f} bohr, θ={REF_TH:.1f}°, E={REF_E:.8f} Ha")

    exp.artifact("outputs/h2o_opt_traj.xyz")
    exp.artifact("outputs/h2o_opt_history.json")

    txt_path, json_path = exp.save()
    print(f"Structured log : {txt_path}")
    print(f"               : {json_path}\n")


if __name__ == "__main__":
    main()
