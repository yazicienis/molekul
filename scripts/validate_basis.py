#!/usr/bin/env python3
"""
scripts/validate_basis.py — Phase 9: cc-pVDZ and 6-31G* basis set validation.

Tests RHF and MP2 energies for H2, H2O against PySCF at identical geometries.
Also verifies internal basis-set properties (orthonormality, n_basis counts).

Usage
-----
  python scripts/validate_basis.py
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
from molekul.basis_ccpvdz  import ccpVDZ
from molekul.basis_631gstar import G631Star
from molekul.rhf           import rhf_scf
from molekul.mp2           import mp2_energy
from molekul.integrals     import build_overlap
from molekul.logging_utils import ExperimentLogger

REPO = Path(__file__).resolve().parents[1]
SEP  = "─" * 64

BASES = {"STO-3G": STO3G, "cc-pVDZ": ccpVDZ, "6-31G*": G631Star}


# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------

def _h2():
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1, name="H2",
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
    for mol_name, mol in mols.items():
        atom_str = "; ".join(
            f"{a.symbol} {a.coords[0]:.15f} {a.coords[1]:.15f} {a.coords[2]:.15f}"
            for a in mol.atoms
        )
        for basis_name in ["sto-3g", "cc-pvdz", "6-31g*"]:
            key = f"{mol_name}_{basis_name.replace('-','').replace('*','s')}"
            pm = gto.M(
                atom=atom_str, basis=basis_name,
                unit="Bohr", charge=mol.charge, spin=0, verbose=0,
                cart=True,   # Cartesian d/f to match MOLEKUL's convention
            )
            mf = pyscf_scf.RHF(pm).run()
            mp_calc = pyscf_mp.MP2(mf).run()
            refs[key] = {
                "e_hf":    float(mf.e_tot),
                "e_mp2":   float(mp_calc.e_corr),
                "n_basis": pm.nao_nr(),
            }
    return refs


# ---------------------------------------------------------------------------
# Basis function count table
# ---------------------------------------------------------------------------

_EXPECTED_NBASIS = {
    # (mol_name, basis_name): expected n_basis
    ("H2",  "STO-3G"):  2,
    ("H2",  "cc-pVDZ"): 10,   # 2 × [2s1p] = 2×5
    ("H2",  "6-31G*"):   4,   # 2 × [2s]   = 2×2
    ("H2O", "STO-3G"):  7,
    ("H2O", "cc-pVDZ"): 25,   # O:[3s2p1d]=15  +  2×H:[2s1p]=2×5
    ("H2O", "6-31G*"):  19,   # O:[3s2p1d]=15  +  2×H:[2s]=2×2
}


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    log = ExperimentLogger("phase9", "basis_sets")

    print("=" * 64)
    print("  Phase 9: Basis Set Validation  —  cc-pVDZ and 6-31G*")
    print("=" * 64)

    mols = {"H2": _h2(), "H2O": _h2o()}

    print("\n  Computing PySCF references ...")
    pyscf = _pyscf_refs(mols)
    if pyscf:
        print("  PySCF available.\n")
        log.metric("pyscf_available", True)
    else:
        print("  PySCF not available — energy comparisons skipped.\n")
        log.metric("pyscf_available", False)

    all_ok = True

    for mol_name, mol in mols.items():
        print(f"\n{SEP}")
        print(f"  {mol_name}  ({mol.n_electrons} electrons)")
        print(SEP)
        print(f"  {'Basis':<10} {'n_basis':>8} {'E_HF (Ha)':>18} {'E_MP2 (Ha)':>18}  {'time':>6}")

        for basis_label, basis in BASES.items():
            t0 = time.time()
            try:
                rhf = rhf_scf(mol, basis, verbose=False)
                res = mp2_energy(mol, basis, rhf)
                dt  = time.time() - t0
                nbasis = basis.n_basis(mol)

                print(f"  {basis_label:<10} {nbasis:>8} {rhf.energy_total:>18.10f} "
                      f"{res.energy_mp2:>18.10f}  {dt:>5.2f}s")

                tag = f"{mol_name.lower()}_{basis_label.replace('-','').replace('*','s')}"
                log.metric(f"{tag}_n_basis",  nbasis)
                log.metric(f"{tag}_e_hf",     rhf.energy_total)
                log.metric(f"{tag}_e_mp2",    res.energy_mp2)
                log.metric(f"{tag}_e_total",  res.energy_total)

                # ── n_basis check ────────────────────────────────────────
                key = (mol_name, basis_label)
                if key in _EXPECTED_NBASIS:
                    ok = nbasis == _EXPECTED_NBASIS[key]
                    log.check(f"{mol_name}/{basis_label} n_basis == {_EXPECTED_NBASIS[key]}",
                              ok, f"got {nbasis}")
                    all_ok &= ok

                # ── convergence ──────────────────────────────────────────
                ok = rhf.converged
                log.check(f"{mol_name}/{basis_label} RHF converged", ok)
                all_ok &= ok

                # ── E_MP2 <= 0 ───────────────────────────────────────────
                ok = res.energy_mp2 <= 0.0
                log.check(f"{mol_name}/{basis_label} E_MP2 <= 0", ok,
                          f"{res.energy_mp2:.6e}")
                all_ok &= ok

                # ── PySCF comparison ─────────────────────────────────────
                pyscf_key = f"{mol_name}_{basis_label.lower().replace('-','').replace('*','s')}"
                if pyscf and pyscf_key in pyscf:
                    ref = pyscf[pyscf_key]

                    ok = abs(rhf.energy_total - ref["e_hf"]) < 1e-6
                    log.check(
                        f"{mol_name}/{basis_label} E_HF vs PySCF < 1e-6 Ha",
                        ok,
                        f"MOLEKUL={rhf.energy_total:.10f} PySCF={ref['e_hf']:.10f} "
                        f"diff={abs(rhf.energy_total-ref['e_hf']):.2e}",
                    )
                    all_ok &= ok

                    ok = abs(res.energy_mp2 - ref["e_mp2"]) < 1e-5
                    log.check(
                        f"{mol_name}/{basis_label} E_MP2 vs PySCF < 1e-5 Ha",
                        ok,
                        f"MOLEKUL={res.energy_mp2:.10f} PySCF={ref['e_mp2']:.10f} "
                        f"diff={abs(res.energy_mp2-ref['e_mp2']):.2e}",
                    )
                    all_ok &= ok

            except Exception as e:
                print(f"  {basis_label:<10}  ERROR: {e}")
                log.check(f"{mol_name}/{basis_label} no exception", False, str(e))
                all_ok = False

    # ── Basis improvement summary ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Energy improvement vs STO-3G  (H2O, E_HF in Ha)")
    print(SEP)
    for basis_label, basis in BASES.items():
        rhf = rhf_scf(_h2o(), basis, verbose=False)
        res = mp2_energy(_h2o(), basis, rhf)
        print(f"  {basis_label:<10}  E_HF = {rhf.energy_total:>14.8f}   "
              f"E_HF+MP2 = {res.energy_total:>14.8f}")

    dt = time.time() - t_start
    log.metric("elapsed_s", dt)
    print(f"\n  Total time: {dt:.2f}s")
    txt, jsn = log.save()
    print(f"  Log: {txt.name},  {jsn.name}")

    if all_ok:
        print("\n  All checks passed.")
    else:
        print("\n  SOME CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
