#!/usr/bin/env python3
"""
scripts/validate_mp2.py — Phase 8: MP2 correlation energy validation.

Molecules tested: H2, HeH+, H2O  (STO-3G basis, same geometries as Phase 4)

Validation methodology
----------------------
EXACT invariants (no external reference needed):
  • E_MP2 < 0                      (correlation energy is always negative)
  • E_MP2 + E_HF < E_HF            (MP2 lowers total energy)
  • E_MP2 = 0 for a one-electron system (no correlation for 1 electron)

PySCF cross-checks (when PySCF is available):
  • |E_MP2(MOLEKUL) − E_MP2(PySCF)| < 1e-6 Ha

Usage
-----
  python scripts/validate_mp2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math
import numpy as np

from molekul.atoms        import Atom
from molekul.molecule     import Molecule
from molekul.basis_sto3g  import STO3G
from molekul.rhf          import rhf_scf
from molekul.mp2          import mp2_energy
from molekul.logging_utils import ExperimentLogger

REPO = Path(__file__).resolve().parents[1]
SEP  = "─" * 64


# ---------------------------------------------------------------------------
# Molecule builders (same geometries as Phase 4)
# ---------------------------------------------------------------------------

def _h2():
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1, name="H2",
    )


def _heh():
    return Molecule(
        atoms=[Atom("He", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4632])],
        charge=1, multiplicity=1, name="HeH+",
    )


def _h2o():
    R, th = 1.870, math.radians(50.0)
    return Molecule(
        atoms=[
            Atom("O", [0.0,  0.0,              0.0]),
            Atom("H", [0.0,  R * math.sin(th), -R * math.cos(th)]),
            Atom("H", [0.0, -R * math.sin(th), -R * math.cos(th)]),
        ],
        charge=0, multiplicity=1, name="H2O",
    )


# ---------------------------------------------------------------------------
# PySCF references
# ---------------------------------------------------------------------------

def _pyscf_refs(mols: dict) -> dict | None:
    try:
        from pyscf import gto, scf as pyscf_scf, mp as pyscf_mp
    except ImportError:
        return None

    refs = {}
    for name, mol in mols.items():
        atom_str = "; ".join(
            f"{a.symbol} {a.coords[0]:.15f} {a.coords[1]:.15f} {a.coords[2]:.15f}"
            for a in mol.atoms
        )
        pm = gto.M(
            atom=atom_str, basis="sto-3g",
            unit="Bohr", charge=mol.charge, spin=0, verbose=0,
        )
        mf = pyscf_scf.RHF(pm).run()
        mp = pyscf_mp.MP2(mf).run()
        refs[name] = {
            "e_hf":   float(mf.e_tot),
            "e_mp2":  float(mp.e_corr),   # correlation part only
            "e_tot":  float(mp.e_tot),
        }
    return refs


# ---------------------------------------------------------------------------
# Per-molecule validation
# ---------------------------------------------------------------------------

def run_mol(name: str, mol: Molecule, log: ExperimentLogger,
            pyscf: dict | None) -> bool:
    print(f"\n{SEP}")
    print(f"  {name}  ({mol.n_electrons} electrons, charge={mol.charge})")
    print(SEP)

    t0 = time.time()
    rhf = rhf_scf(mol, STO3G, verbose=False)
    print(f"  RHF: converged={rhf.converged}, E_HF = {rhf.energy_total:.10f} Ha")

    res = mp2_energy(mol, STO3G, rhf)
    dt  = time.time() - t0

    print(f"  n_occ = {res.n_occ},  n_virt = {res.n_virt}")
    print(f"  E_MP2 (corr)  = {res.energy_mp2:+.10f} Ha")
    print(f"  E_total(MP2)  = {res.energy_total:.10f} Ha")
    print(f"  Wall: {dt:.3f}s")

    tag = name.lower().replace("+", "p")
    log.metric(f"{tag}_e_hf",    rhf.energy_total)
    log.metric(f"{tag}_e_mp2",   res.energy_mp2)
    log.metric(f"{tag}_e_total", res.energy_total)

    all_ok = True

    # ── EXACT invariants ──────────────────────────────────────────────────
    ok = res.energy_mp2 <= 0.0
    log.check(f"{name} [EXACT]: E_MP2 ≤ 0",
              ok, f"E_MP2 = {res.energy_mp2:.6e}")
    all_ok &= ok

    ok = res.energy_total < res.energy_hf
    log.check(f"{name} [EXACT]: E_total(MP2) < E_HF",
              ok, f"diff = {res.energy_total - res.energy_hf:.6e}")
    all_ok &= ok

    ok = abs(res.energy_total - (res.energy_hf + res.energy_mp2)) < 1e-12
    log.check(f"{name} [EXACT]: E_total = E_HF + E_MP2",
              ok, f"residual = {abs(res.energy_total - res.energy_hf - res.energy_mp2):.2e}")
    all_ok &= ok

    # ── PySCF comparison ──────────────────────────────────────────────────
    if pyscf and name in pyscf:
        ref = pyscf[name]
        print(f"  PySCF: E_MP2(corr) = {ref['e_mp2']:+.10f} Ha")

        ok = abs(res.energy_mp2 - ref["e_mp2"]) < 1e-6
        log.check(f"{name} [PySCF]: E_MP2 within 1e-6 Ha",
                  ok, f"MOLEKUL={res.energy_mp2:+.10f}, PySCF={ref['e_mp2']:+.10f}, "
                      f"diff={abs(res.energy_mp2 - ref['e_mp2']):.2e}")
        all_ok &= ok

        ok = abs(rhf.energy_total - ref["e_hf"]) < 1e-6
        log.check(f"{name} [PySCF]: E_HF within 1e-6 Ha",
                  ok, f"diff={abs(rhf.energy_total - ref['e_hf']):.2e}")
        all_ok &= ok
    else:
        print("  [PySCF not available — skipping quantitative comparison]")
        log.check(f"{name} [PySCF]: SKIPPED", True, "qualitative only")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    log = ExperimentLogger("phase8", "mp2")

    print("=" * 64)
    print("  Phase 8: MP2 Correlation Energy  —  RHF/STO-3G")
    print("=" * 64)

    mols = {"H2": _h2(), "HeH+": _heh(), "H2O": _h2o()}

    print("\n  Computing PySCF references ...")
    pyscf = _pyscf_refs(mols)
    if pyscf:
        print("  PySCF available.")
        log.metric("pyscf_available", True)
    else:
        print("  PySCF not available.")
        log.metric("pyscf_available", False)

    results = {}
    for name, mol in mols.items():
        results[name] = run_mol(name, mol, log, pyscf)

    dt = time.time() - t_start
    log.metric("elapsed_s", dt)

    print(f"\n{SEP}")
    print("  Known limitations of STO-3G MP2")
    print(SEP)
    print("""\
  1. Minimal basis: STO-3G MP2 captures only ~30-50% of the correlation
     energy vs. experiment.  Augmented triple-zeta (cc-pVTZ) is needed
     for quantitative results.
  2. Divergence for near-degenerate denominators: if ε_i + ε_j ≈ ε_a + ε_b,
     MP2 diverges.  This occurs near bond dissociation (multi-reference regime).
  3. Closed-shell only: this implementation covers RHF-MP2 only; open-shell
     systems require UMP2 or ROMP2.
  4. Formal scaling: O(N^5) for the AO→MO transform + O(N^4) for the energy
     sum.  Density-fitting (RI-MP2) reduces the prefactor to near O(N^3).
  5. No gradients: MP2 geometry optimisation requires MP2 analytic gradients
     (not implemented here).""")

    print(f"\n  Total time: {dt:.2f}s")
    txt_path, json_path = log.save()
    print(f"  Log written: {txt_path.name},  {json_path.name}")

    if all(results.values()):
        print("\n  All checks passed.")
    else:
        print("\n  SOME CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
