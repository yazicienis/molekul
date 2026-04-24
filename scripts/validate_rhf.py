#!/usr/bin/env python3
"""
Phase 4 validation: RHF SCF energies vs. reference values.

Molecules tested
----------------
  H2     R = 1.4 bohr,  STO-3G reference: -1.117496 Hartree  (Szabo & Ostlund, Table 3.12)
  HeH+   R = 1.4632 bohr, STO-3G
  H2O    C2v geometry,  STO-3G reference: -74.9659012 Hartree

Optionally compares with PySCF if installed (import pyscf).

Usage
-----
  python scripts/validate_rhf.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf import rhf_scf

SEPARATOR = "=" * 62


def run_rhf(mol, label, ref_energy=None):
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)
    result = rhf_scf(mol, STO3G, verbose=True)

    status = "CONVERGED" if result.converged else "*** NOT CONVERGED ***"
    print(f"\nStatus       : {status}")
    print(f"SCF iters    : {result.n_iter}")
    print(f"E_nuc        : {result.energy_nuclear:>20.10f} Hartree")
    print(f"E_electronic : {result.energy_electronic:>20.10f} Hartree")
    print(f"E_total      : {result.energy_total:>20.10f} Hartree")
    print(f"MO energies  : {result.mo_energies}")

    if ref_energy is not None:
        diff = result.energy_total - ref_energy
        tol = 1e-4
        ok = abs(diff) < tol
        print(f"\nReference    : {ref_energy:>20.10f} Hartree  ({label.split()[0]})")
        print(f"Difference   : {diff:>+.2e} Hartree  [tol={tol:.0e}]  {'PASS' if ok else 'FAIL'}")

    return result


def compare_pyscf(mol_atoms, basis_name, label, charge=0):
    """Optional PySCF comparison — skipped if pyscf not installed."""
    try:
        from pyscf import gto, scf as pyscf_scf
    except ImportError:
        print(f"\n  [PySCF not available — skipping {label} comparison]")
        return

    print(f"\n--- PySCF cross-check: {label} ---")
    atom_str = "; ".join(
        f"{sym} {x:.6f} {y:.6f} {z:.6f}"
        for sym, (x, y, z) in mol_atoms
    )
    mol_pyscf = gto.Mole()
    mol_pyscf.atom = atom_str
    mol_pyscf.basis = basis_name
    mol_pyscf.unit = "Bohr"
    mol_pyscf.charge = charge
    mol_pyscf.verbose = 0
    mol_pyscf.build()
    mf = pyscf_scf.RHF(mol_pyscf)
    mf.kernel()
    print(f"  PySCF E_total = {mf.e_tot:.10f} Hartree")


def main():
    print(f"\nMOLEKUL — Phase 4 RHF Validation")
    print(f"STO-3G basis, all geometries in Bohr\n")

    # ------------------------------------------------------------------
    # H2  (R = 1.4 bohr)
    # Reference: Szabo & Ostlund Table 3.12 — E = -1.1174963 Hartree
    # ------------------------------------------------------------------
    h2 = Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1, name="H2",
    )
    # PySCF reference at this geometry: -1.1167143251 Hartree
    r1 = run_rhf(h2, "H2  (R=1.4 bohr, STO-3G)", ref_energy=-1.1167143251)
    compare_pyscf(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        "sto-3g", "H2", charge=0
    )

    # ------------------------------------------------------------------
    # HeH+  (R = 1.4632 bohr)
    # ------------------------------------------------------------------
    heh = Molecule(
        atoms=[Atom("He", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4632])],
        charge=1, multiplicity=1, name="HeH+",
    )
    r2 = run_rhf(heh, "HeH+  (R=1.4632 bohr, STO-3G)")
    compare_pyscf(
        [("He", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4632))],
        "sto-3g", "HeH+", charge=1
    )

    # ------------------------------------------------------------------
    # H2O  (C2v, standard STO-3G geometry in Angstrom)
    # Reference: E = -74.9659012 Hartree
    # ------------------------------------------------------------------
    water = Molecule(
        atoms=[
            Atom.from_angstrom("O",  0.000,  0.000,  0.000),
            Atom.from_angstrom("H",  0.000,  0.757, -0.586),
            Atom.from_angstrom("H",  0.000, -0.757, -0.586),
        ],
        charge=0, multiplicity=1, name="H2O",
    )
    # PySCF reference at this geometry: -74.9629466405 Hartree
    r3 = run_rhf(water, "H2O  (C2v STO-3G)", ref_energy=-74.9629466405)
    compare_pyscf(
        [
            ("O", tuple(water.atoms[0].coords)),
            ("H", tuple(water.atoms[1].coords)),
            ("H", tuple(water.atoms[2].coords)),
        ],
        "sto-3g", "H2O"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("  SUMMARY")
    print(SEPARATOR)
    molecules = ["H2", "HeH+", "H2O"]
    results   = [r1,   r2,    r3]
    refs      = [-1.1167143251, None, -74.9629466405]
    for name, res, ref in zip(molecules, results, refs):
        conv = "OK" if res.converged else "FAIL"
        E = res.energy_total
        if ref is not None:
            diff = E - ref
            match = "OK" if abs(diff) < 1e-4 else "FAIL"
            print(f"  {name:<6}  E={E:>18.8f}  conv={conv}  vs ref {diff:>+.2e}  {match}")
        else:
            print(f"  {name:<6}  E={E:>18.8f}  conv={conv}")
    print()

    # --- Dual-format log to outputs/logs/ -----------------------------------
    from molekul.logging_utils import ExperimentLogger
    exp = ExperimentLogger("phase4", "rhf")
    for mol_name, res, ref in zip(molecules, results, refs):
        k = mol_name.lower().replace("+", "plus").replace("2", "2")
        exp.metric(f"{k}_energy_ha",   round(res.energy_total, 10))
        exp.metric(f"{k}_enuc_ha",     round(res.energy_nuclear, 10))
        exp.metric(f"{k}_converged",   res.converged)
        exp.metric(f"{k}_n_iter",      res.n_iter)
        if ref is not None:
            diff = res.energy_total - ref
            exp.metric(f"{k}_pyscf_ref_ha", ref)
            exp.metric(f"{k}_pyscf_diff_ha", round(diff, 12))
            exp.check(f"{mol_name} energy within 1e-4 of PySCF", abs(diff) < 1e-4)
        exp.check(f"{mol_name} SCF converged", res.converged)
    exp.line("Method: RHF/STO-3G, DIIS, symmetric orthogonalisation")
    exp.line("Molecules: H2 (R=1.4 bohr), HeH+ (R=1.4632 bohr), H2O (C2v)")
    exp.line("References: PySCF 2.x at identical geometries")
    exp.artifact("outputs/phase4_rhf.txt")
    txt_path, json_path = exp.save()
    print(f"Structured log : {txt_path}")
    print(f"               : {json_path}")


if __name__ == "__main__":
    main()
