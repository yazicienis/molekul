"""
scripts/benchmark_14mol.py
==========================
14-molecule RHF/STO-3G validation table: MOLEKUL vs PySCF.

Molecules cover H, He, Li, B, C, N, O, F — first-row elements typical
of quantum chemistry validation suites.

Usage
-----
    python scripts/benchmark_14mol.py [--tol 1e-6] [--verbose]

Exit codes
----------
0 — all energies agree within tolerance
1 — PySCF not available
2 — one or more molecules failed
"""

import argparse
import json
import sys
import time

sys.path.insert(0, "src")

try:
    from pyscf import gto, scf
except ImportError:
    print("PySCF is not installed.  pip install pyscf")
    sys.exit(1)

import numpy as np

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf

PASS_STR = "\033[32mPASS\033[0m"
FAIL_STR = "\033[31mFAIL\033[0m"


# ---------------------------------------------------------------------------
# Molecule definitions (all coordinates in Bohr for unambiguous comparison)
# Geometries: standard reference / STO-3G published structures
# ---------------------------------------------------------------------------

def _m(name, atoms_bohr, charge=0, mult=1):
    """Build a MOLEKUL Molecule from (symbol, x, y, z) tuples in Bohr."""
    atoms = [Atom(sym, list(coords)) for sym, *coords in atoms_bohr]
    return Molecule(atoms, charge=charge, multiplicity=mult, name=name)


MOLECULES = [
    # (name, molekul_mol, pyscf_atom_str [Bohr], charge, spin)
    ("H2",
     _m("H2", [("H", 0, 0, 0), ("H", 0, 0, 1.4)]),
     "H 0 0 0; H 0 0 1.4", 0, 0),

    ("HeH+",
     _m("HeH+", [("He", 0, 0, 0), ("H", 0, 0, 1.4632)], charge=1),
     "He 0 0 0; H 0 0 1.4632", 1, 0),

    ("LiH",
     _m("LiH", [("Li", 0, 0, 0), ("H", 0, 0, 3.0154)]),
     "Li 0 0 0; H 0 0 3.0154", 0, 0),

    ("HF",
     _m("HF", [("H", 0, 0, 0), ("F", 0, 0, 1.7329)]),
     "H 0 0 0; F 0 0 1.7329", 0, 0),

    ("N2",
     _m("N2", [("N", 0, 0, -1.0372), ("N", 0, 0, 1.0372)]),
     "N 0 0 -1.0372; N 0 0 1.0372", 0, 0),

    ("CO",
     _m("CO", [("C", 0, 0, 0), ("O", 0, 0, 2.1316)]),
     "C 0 0 0; O 0 0 2.1316", 0, 0),

    ("H2O",
     _m("H2O", [
         ("O",  0.0000,  0.0000,  0.0000),
         ("H",  0.0000,  1.4297, -1.1075),
         ("H",  0.0000, -1.4297, -1.1075),
     ]),
     "O 0 0 0; H 0 1.4297 -1.1075; H 0 -1.4297 -1.1075", 0, 0),

    ("NH3",
     _m("NH3", [
         ("N",  0.0000,  0.0000,  0.2188),
         ("H",  0.0000,  1.7732, -0.5105),
         ("H",  1.5353, -0.8866, -0.5105),
         ("H", -1.5353, -0.8866, -0.5105),
     ]),
     "N 0 0 0.2188; H 0 1.7732 -0.5105; H 1.5353 -0.8866 -0.5105; H -1.5353 -0.8866 -0.5105",
     0, 0),

    ("CH4",
     _m("CH4", [
         ("C",  0.0000,  0.0000,  0.0000),
         ("H",  1.1880,  1.1880,  1.1880),
         ("H", -1.1880, -1.1880,  1.1880),
         ("H", -1.1880,  1.1880, -1.1880),
         ("H",  1.1880, -1.1880, -1.1880),
     ]),
     "C 0 0 0; H 1.188 1.188 1.188; H -1.188 -1.188 1.188; H -1.188 1.188 -1.188; H 1.188 -1.188 -1.188",
     0, 0),

    ("BH3",
     _m("BH3", [
         ("B",  0.0000,  0.0000,  0.0000),
         ("H",  2.2677,  0.0000,  0.0000),
         ("H", -1.1339,  1.9643,  0.0000),
         ("H", -1.1339, -1.9643,  0.0000),
     ]),
     "B 0 0 0; H 2.2677 0 0; H -1.1339 1.9643 0; H -1.1339 -1.9643 0",
     0, 0),

    ("HCN",
     _m("HCN", [("H", 0, 0, 0), ("C", 0, 0, 2.0130), ("N", 0, 0, 4.2279)]),
     "H 0 0 0; C 0 0 2.013; N 0 0 4.2279", 0, 0),

    ("C2H2",
     _m("C2H2", [
         ("H",  0, 0, -3.9943),
         ("C",  0, 0, -1.9943),
         ("C",  0, 0,  1.9943),
         ("H",  0, 0,  3.9943),
     ]),
     "H 0 0 -3.9943; C 0 0 -1.9943; C 0 0 1.9943; H 0 0 3.9943", 0, 0),

    ("C2H4",
     _m("C2H4", [
         ("C",  0.0000,  0.0000,  1.2624),
         ("C",  0.0000,  0.0000, -1.2624),
         ("H",  0.0000,  1.7433,  2.3381),
         ("H",  0.0000, -1.7433,  2.3381),
         ("H",  0.0000,  1.7433, -2.3381),
         ("H",  0.0000, -1.7433, -2.3381),
     ]),
     "C 0 0 1.2624; C 0 0 -1.2624; H 0 1.7433 2.3381; H 0 -1.7433 2.3381; H 0 1.7433 -2.3381; H 0 -1.7433 -2.3381",
     0, 0),

    ("CH2O",
     _m("CH2O", [
         ("C",  0.0000,  0.0000,  0.0000),
         ("O",  0.0000,  0.0000,  2.2716),
         ("H",  0.0000,  1.7617, -0.6680),
         ("H",  0.0000, -1.7617, -0.6680),
     ]),
     "C 0 0 0; O 0 0 2.2716; H 0 1.7617 -0.6680; H 0 -1.7617 -0.6680",
     0, 0),
]


def run_pyscf(atom_str, charge, spin):
    pm = gto.M(atom=atom_str, basis="sto-3g", unit="Bohr",
               charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(pm)
    mf.max_cycle = 200
    mf.conv_tol = 1e-12
    e = mf.kernel()
    return e, mf.converged


def run_molekul(mol):
    t0 = time.perf_counter()
    res = rhf_scf(mol, STO3G, e_conv=1e-10, d_conv=1e-8, max_iter=200)
    dt = time.perf_counter() - t0
    return res.energy_total, res.converged, res.n_iter, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Absolute energy tolerance in Hartree (default 1e-6)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    tol = args.tol

    print(f"\nMOLEKUL vs PySCF — RHF/STO-3G total energies")
    print(f"Tolerance: {tol:.0e} Hartree  |  {len(MOLECULES)} molecules\n")

    header = f"{'Molecule':<10}  {'MOLEKUL (Eh)':>18}  {'PySCF (Eh)':>18}  {'|ΔE| (Eh)':>13}  {'Iters':>5}  {'Status'}"
    print(header)
    print("-" * len(header))

    rows = []
    failures = []

    for name, mol, pyscf_str, charge, spin in MOLECULES:
        try:
            e_mol, conv_mol, n_iter, t_mol = run_molekul(mol)
            e_pyscf, conv_pyscf = run_pyscf(pyscf_str, charge, spin)
            delta = abs(e_mol - e_pyscf)
            ok = delta <= tol and conv_mol and conv_pyscf
            status = PASS_STR if ok else FAIL_STR
            print(f"{name:<10}  {e_mol:>18.10f}  {e_pyscf:>18.10f}  {delta:>13.3e}  {n_iter:>5}  {status}")
            rows.append({
                "molecule": name,
                "E_molekul": float(e_mol),
                "E_pyscf": float(e_pyscf),
                "delta_E": float(delta),
                "n_iter": int(n_iter),
                "converged_mol": bool(conv_mol),
                "converged_pyscf": bool(conv_pyscf),
                "pass": bool(ok),
            })
            if not ok:
                failures.append(name)
        except Exception as exc:
            print(f"{name:<10}  {'ERROR':>18}  {'':>18}  {'':>13}  {'':>5}  \033[31mERROR: {exc}\033[0m")
            failures.append(name)
            rows.append({"molecule": name, "error": str(exc), "pass": False})

    print()
    n_pass = sum(1 for r in rows if r.get("pass"))
    print(f"Results: {n_pass}/{len(MOLECULES)} passed (tol = {tol:.0e} Hartree)")

    if failures:
        print(f"\n\033[31mFailed:\033[0m {', '.join(failures)}")

    # Save JSON
    out = {
        "tolerance_hartree": tol,
        "n_molecules": len(MOLECULES),
        "n_pass": n_pass,
        "rows": rows,
    }
    outfile = "outputs/logs/benchmark_14mol.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {outfile}")

    if args.verbose:
        deltas = [r["delta_E"] for r in rows if "delta_E" in r]
        if deltas:
            print(f"\nMax |ΔE|: {max(deltas):.3e} Hartree")
            print(f"Mean |ΔE|: {np.mean(deltas):.3e} Hartree")

    sys.exit(2 if failures else 0)


if __name__ == "__main__":
    main()
