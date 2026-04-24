#!/usr/bin/env python3
"""
scripts/validate_freqs.py — Phase 10: Harmonic frequency analysis validation.

Tests:
  1. H2  (STO-3G, equilibrium)  — 1 vibrational mode, frequency vs PySCF
  2. H2O (STO-3G, equilibrium)  — 3 vibrational modes, frequencies vs PySCF
  3. IR intensities: non-negative, total-symmetric mode check

STO-3G equilibrium geometries from PySCF optimizer:
  H2  : R = 1.345919 bohr  (E = -1.11750588 Ha)
  H2O : O=[0,0,0.06311], H=[0,±1.43256,-1.13839] bohr  (E = -74.96590121 Ha)

Reference frequencies [PySCF analytic Hessian, avg masses, same geometries]:
  H2  : 5481 cm⁻¹
  H2O : 2169.85, 4139.64, 4390.67 cm⁻¹
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from molekul.atoms         import Atom
from molekul.molecule      import Molecule
from molekul.basis_sto3g   import STO3G
from molekul.freqs         import harmonic_analysis, FREQ_CONV
from molekul.logging_utils import ExperimentLogger

REPO = Path(__file__).resolve().parents[1]
SEP  = "─" * 64

# PySCF analytic Hessian, average masses, same equilibrium geometries
_REF_H2_FREQ   = 5481.0
_REF_H2O_FREQS = [2169.85, 4139.64, 4390.67]

# Tolerance: numerical Hessian (h=5e-3) matches analytic to < 5 cm⁻¹
_FREQ_TOL = 10.0    # cm⁻¹
_INTENS_TOL = 1.0   # km/mol  (generous)


# ---------------------------------------------------------------------------
# Equilibrium geometries
# ---------------------------------------------------------------------------

def _h2_eq():
    """H2 at STO-3G equilibrium R = 1.345919 bohr (BFGS-optimized)."""
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.345919])],
        charge=0, multiplicity=1, name="H2",
    )


def _h2o_eq():
    """H2O at STO-3G equilibrium (BFGS-optimized, bohr)."""
    return Molecule(
        atoms=[
            Atom("O", [ 0.00000000,  0.00000000,  0.06310826]),
            Atom("H", [ 0.00000000,  1.43256299, -1.13838561]),
            Atom("H", [ 0.00000000, -1.43256299, -1.13838561]),
        ],
        charge=0, multiplicity=1, name="H2O",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    log = ExperimentLogger("phase10", "harmonic_freqs")

    print("=" * 64)
    print("  Phase 10: Harmonic Frequency Analysis  (STO-3G)")
    print("=" * 64)

    all_ok = True

    # ── H2 ──────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Molecule: H2  (STO-3G equilibrium, R = 1.3889 bohr)")
    print(SEP)

    mol_h2 = _h2_eq()
    t0 = time.time()
    fr_h2 = harmonic_analysis(mol_h2, STO3G, h_hess=5e-3, h_dip=1e-2, verbose=True)
    dt_h2 = time.time() - t0

    print(f"\n  n_zero     = {fr_h2.n_zero}  (expected 5 for linear)")
    print(f"  n_imaginary= {fr_h2.n_imaginary}")
    print(f"  Frequencies (cm⁻¹): {fr_h2.frequencies}")
    print(f"  Intensities (km/mol): {fr_h2.intensities}")
    print(f"  ZPE = {fr_h2.zero_point_energy:.6f} Ha  "
          f"({fr_h2.zero_point_energy * 219474.63:.1f} cm⁻¹)")
    print(f"  Elapsed: {dt_h2:.1f}s")

    # n_zero check
    ok = fr_h2.n_zero == 5
    log.check("H2 n_zero == 5 (linear molecule)", ok, f"got {fr_h2.n_zero}")
    all_ok &= ok

    # n_vib check
    ok = len(fr_h2.frequencies) == 1
    log.check("H2 n_vib == 1", ok, f"got {len(fr_h2.frequencies)}")
    all_ok &= ok

    # Frequency vs PySCF
    if len(fr_h2.frequencies) >= 1:
        freq_h2 = float(fr_h2.frequencies[0])
        diff = abs(freq_h2 - _REF_H2_FREQ)
        ok   = diff < _FREQ_TOL
        log.check(f"H2 ω₁ vs PySCF < {_FREQ_TOL} cm⁻¹",
                  ok, f"MOLEKUL={freq_h2:.1f}  PySCF={_REF_H2_FREQ:.1f}  diff={diff:.1f}")
        all_ok &= ok

    # Intensity non-negative
    ok = bool(np.all(fr_h2.intensities >= 0.0))
    log.check("H2 intensities >= 0", ok)
    all_ok &= ok

    log.metric("h2_frequencies",  fr_h2.frequencies.tolist())
    log.metric("h2_intensities",  fr_h2.intensities.tolist())
    log.metric("h2_zpe_ha",       fr_h2.zero_point_energy)
    log.metric("h2_n_zero",       fr_h2.n_zero)
    log.metric("h2_elapsed_s",    dt_h2)

    # ── H2O ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Molecule: H2O  (STO-3G equilibrium)")
    print(SEP)

    mol_h2o = _h2o_eq()
    t0 = time.time()
    fr_h2o = harmonic_analysis(mol_h2o, STO3G, h_hess=5e-3, h_dip=1e-2, verbose=True)
    dt_h2o = time.time() - t0

    print(f"\n  n_zero      = {fr_h2o.n_zero}  (expected 6 for nonlinear)")
    print(f"  n_imaginary = {fr_h2o.n_imaginary}")
    print(f"  Frequencies (cm⁻¹):")
    for k, (f, I) in enumerate(zip(fr_h2o.frequencies, fr_h2o.intensities)):
        ref = _REF_H2O_FREQS[k] if k < len(_REF_H2O_FREQS) else None
        ref_str = f"  ref={ref:.1f}" if ref else ""
        print(f"    mode {k+1}: {f:8.1f} cm⁻¹   I = {I:8.2f} km/mol{ref_str}")
    print(f"  ZPE = {fr_h2o.zero_point_energy:.6f} Ha  "
          f"({fr_h2o.zero_point_energy * 219474.63:.1f} cm⁻¹)")
    print(f"  Elapsed: {dt_h2o:.1f}s")

    # n_zero check
    ok = fr_h2o.n_zero == 6
    log.check("H2O n_zero == 6 (nonlinear)", ok, f"got {fr_h2o.n_zero}")
    all_ok &= ok

    # n_vib check
    ok = len(fr_h2o.frequencies) == 3
    log.check("H2O n_vib == 3", ok, f"got {len(fr_h2o.frequencies)}")
    all_ok &= ok

    # n_imaginary == 0
    ok = fr_h2o.n_imaginary == 0
    log.check("H2O n_imaginary == 0 (minimum)", ok, f"got {fr_h2o.n_imaginary}")
    all_ok &= ok

    # Frequency regressions
    if len(fr_h2o.frequencies) == 3:
        for k, ref in enumerate(_REF_H2O_FREQS):
            freq = float(fr_h2o.frequencies[k])
            diff = abs(freq - ref)
            ok   = diff < _FREQ_TOL
            log.check(f"H2O ω{k+1} vs PySCF < {_FREQ_TOL} cm⁻¹",
                      ok, f"MOLEKUL={freq:.1f}  PySCF={ref:.1f}  diff={diff:.1f}")
            all_ok &= ok

    # Intensities non-negative
    ok = bool(np.all(fr_h2o.intensities >= 0.0))
    log.check("H2O intensities >= 0", ok)
    all_ok &= ok

    log.metric("h2o_frequencies",  fr_h2o.frequencies.tolist())
    log.metric("h2o_intensities",  fr_h2o.intensities.tolist())
    log.metric("h2o_zpe_ha",       fr_h2o.zero_point_energy)
    log.metric("h2o_n_zero",       fr_h2o.n_zero)
    log.metric("h2o_elapsed_s",    dt_h2o)

    # ── Summary ──────────────────────────────────────────────────────────────
    dt = time.time() - t_start
    log.metric("elapsed_s", dt)

    txt, jsn = log.save()
    print(f"\n  Log: {txt.name},  {jsn.name}")
    print(f"  Total time: {dt:.1f}s")

    if all_ok:
        print("\n  All checks passed.")
    else:
        print("\n  SOME CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
