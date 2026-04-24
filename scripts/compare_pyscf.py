"""
scripts/compare_pyscf.py
========================
Element-by-element comparison of MOLEKUL one-electron integrals against PySCF.

Requires PySCF:  pip install pyscf

Exits 0 if max absolute deviation ≤ threshold on all matrices.
Exits 1 if PySCF is not available (prints install instructions).
Exits 2 if any matrix deviates beyond threshold.

Usage
-----
    python scripts/compare_pyscf.py [--tol 1e-5]
"""

import sys
import argparse

sys.path.insert(0, "src")

try:
    from pyscf import gto
except ImportError:
    print("PySCF is not installed.")
    print("Install with:  pip install pyscf")
    sys.exit(1)

import numpy as np

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.integrals import build_overlap, build_kinetic, build_nuclear


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[32m  PASS\033[0m"
FAIL = "\033[31m  FAIL\033[0m"

_failures = []

def compare_matrix(name: str, M_mol: np.ndarray, M_pyscf: np.ndarray,
                   tol: float) -> dict:
    delta = M_mol - M_pyscf
    max_abs = np.max(np.abs(delta))
    max_rel = np.max(np.abs(delta) / (np.abs(M_pyscf) + 1e-30))
    i_max, j_max = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    ok = max_abs <= tol
    status = PASS if ok else FAIL
    print(f"{status}  {name}: max|Δ| = {max_abs:.3e}  max|Δ_rel| = {max_rel:.3e}"
          f"  worst=[{i_max},{j_max}] mol={M_mol[i_max,j_max]:.8f}"
          f" pyscf={M_pyscf[i_max,j_max]:.8f}")
    if not ok:
        _failures.append(name)
    return {"max_abs": max_abs, "max_rel": max_rel, "ok": ok}


def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# Molecule definitions  (identical geometry used in both codes)
# ─────────────────────────────────────────────────────────────────────────────

def _mol_to_pyscf_str(mol) -> str:
    """
    Serialise a MOLEKUL Molecule to PySCF atom string in Bohr.
    Passing Bohr coordinates directly avoids Å→Bohr conversion-factor
    differences between MOLEKUL and PySCF (PySCF uses 0.52917721092,
    MOLEKUL uses 0.52917724899 — a 7e-9 relative difference that
    compounds for multi-atom molecules).
    """
    parts = []
    for atom in mol.atoms:
        x, y, z = atom.coords
        parts.append(f"{atom.symbol} {x:.15f} {y:.15f} {z:.15f}")
    return "; ".join(parts)


_H2 = Molecule(
    atoms=[Atom("H", [0, 0, 0]), Atom("H", [0, 0, 1.4])],
    charge=0, multiplicity=1, name="H2"
)
_HEH = Molecule(
    atoms=[Atom("He", [0, 0, 0]), Atom("H", [0, 0, 1.4632])],
    charge=1, multiplicity=1, name="HeH+"
)
_H2O = Molecule(
    atoms=[
        Atom.from_angstrom("O",  0.0,  0.000, 0.000),
        Atom.from_angstrom("H",  0.0,  0.757, -0.586),
        Atom.from_angstrom("H",  0.0, -0.757, -0.586),
    ],
    charge=0, multiplicity=1, name="H2O"
)

MOLECULES = {
    "H2_R14": {
        "molekul": _H2,
        "pyscf_str": _mol_to_pyscf_str(_H2),
        "pyscf_unit": "Bohr",
        "pyscf_charge": 0,
        "pyscf_spin": 0,
        "label": "H2 at R=1.4 bohr",
    },
    "HeH_plus": {
        "molekul": _HEH,
        "pyscf_str": _mol_to_pyscf_str(_HEH),
        "pyscf_unit": "Bohr",
        "pyscf_charge": 1,
        "pyscf_spin": 0,
        "label": "HeH+ at R=1.4632 bohr",
    },
    "H2O": {
        "molekul": _H2O,
        "pyscf_str": _mol_to_pyscf_str(_H2O),
        "pyscf_unit": "Bohr",
        "pyscf_charge": 0,
        "pyscf_spin": 0,
        "label": "H2O at C2v geometry",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare MOLEKUL integrals to PySCF")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Absolute tolerance for pass/fail (default 1e-5)")
    args = parser.parse_args()
    tol = args.tol

    print(f"MOLEKUL vs PySCF  —  one-electron integrals (STO-3G)")
    print(f"Tolerance: {tol:.1e}")

    summary_rows = []

    for key, cfg in MOLECULES.items():
        section(cfg["label"])

        mol_mol = cfg["molekul"]

        # Build PySCF molecule
        pyscf_mol = gto.M(
            atom=cfg["pyscf_str"],
            basis="sto-3g",
            unit=cfg["pyscf_unit"],
            charge=cfg["pyscf_charge"],
            spin=cfg["pyscf_spin"],
            verbose=0,
        )

        # MOLEKUL integrals
        S_mol = build_overlap(STO3G, mol_mol)
        T_mol = build_kinetic(STO3G, mol_mol)
        V_mol = build_nuclear(STO3G, mol_mol)

        # PySCF integrals
        S_pyscf = pyscf_mol.intor("int1e_ovlp")
        T_pyscf = pyscf_mol.intor("int1e_kin")
        V_pyscf = pyscf_mol.intor("int1e_nuc")

        n_mol = S_mol.shape[0]
        n_pyscf = S_pyscf.shape[0]

        print(f"\n  n_basis: MOLEKUL={n_mol}  PySCF={n_pyscf}")
        if n_mol != n_pyscf:
            print(f"\033[31m  ERROR: basis sizes differ — cannot compare\033[0m")
            _failures.append(f"{key}: basis size mismatch")
            continue

        rS = compare_matrix(f"{key} S", S_mol, S_pyscf, tol)
        rT = compare_matrix(f"{key} T", T_mol, T_pyscf, tol)
        rV = compare_matrix(f"{key} V", V_mol, V_pyscf, tol)

        summary_rows.append({
            "label": cfg["label"],
            "S": rS, "T": rT, "V": rV,
        })

    # ── Per-molecule detailed output ─────────────────────────────────────────
    section("Nuclear Repulsion Energies")
    for key, cfg in MOLECULES.items():
        mol = cfg["molekul"]
        pyscf_mol = gto.M(
            atom=cfg["pyscf_str"], basis="sto-3g",
            unit=cfg["pyscf_unit"],
            charge=cfg["pyscf_charge"], spin=cfg["pyscf_spin"], verbose=0,
        )
        E_mol = mol.nuclear_repulsion_energy()
        E_pyscf = pyscf_mol.energy_nuc()
        delta = abs(E_mol - E_pyscf)
        ok = delta < 1e-8
        status = PASS if ok else FAIL
        print(f"{status}  {cfg['label']}: "
              f"MOLEKUL={E_mol:.10f}  PySCF={E_pyscf:.10f}  |Δ|={delta:.2e}")
        if not ok:
            _failures.append(f"{key}: E_nuc mismatch")

    # ── Summary ──────────────────────────────────────────────────────────────
    section("SUMMARY")
    print(f"\n  {'Molecule':<30}  {'max|ΔS|':>12}  {'max|ΔT|':>12}  {'max|ΔV|':>12}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")
    for row in summary_rows:
        print(f"  {row['label']:<30}  {row['S']['max_abs']:>12.3e}"
              f"  {row['T']['max_abs']:>12.3e}  {row['V']['max_abs']:>12.3e}")

    if _failures:
        print(f"\n\033[31mFAILED ({len(_failures)}):\033[0m")
        for f in _failures:
            print(f"  ✗  {f}")
        sys.exit(2)
    else:
        print(f"\n\033[32mAll matrices agree with PySCF within tol={tol:.1e}.\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
