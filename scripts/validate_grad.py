#!/usr/bin/env python3
"""Phase 18 validation: RHF gradient vs numerical finite differences."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.constants import ANGSTROM_TO_BOHR
from molekul.grad import numerical_gradient, rhf_gradient
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf


def h2_molecule():
    return Molecule(
        atoms=[
            Atom("H", [0.0, 0.0, 0.0]),
            Atom("H", [0.0, 0.0, 0.74 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
        name="H2",
    )


def h2o_molecule():
    return Molecule(
        atoms=[
            Atom("O", [0.0, 0.0, 0.0]),
            Atom("H", [0.0, 0.757 * ANGSTROM_TO_BOHR, -0.469 * ANGSTROM_TO_BOHR]),
            Atom("H", [0.0, -0.757 * ANGSTROM_TO_BOHR, -0.469 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
        name="H2O",
    )


def co_molecule():
    return Molecule(
        atoms=[
            Atom("C", [0.0, 0.0, 0.0]),
            Atom("O", [0.0, 0.0, 1.128 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
        name="CO",
    )


def run_case(mol: Molecule) -> dict:
    t0 = time.perf_counter()
    rhf = rhf_scf(mol, STO3G, verbose=False)
    analytic = rhf_gradient(mol, STO3G, rhf)
    numerical = numerical_gradient(mol, STO3G)
    elapsed = time.perf_counter() - t0
    diff = np.abs(analytic - numerical)
    trans = analytic.sum(axis=0)
    return {
        "molecule": mol.name,
        "energy_total_Ha": float(rhf.energy_total),
        "analytic_gradient_Ha_per_bohr": analytic.tolist(),
        "numerical_gradient_Ha_per_bohr": numerical.tolist(),
        "max_abs_diff_Ha_per_bohr": float(np.max(diff)),
        "translational_residual_Ha_per_bohr": trans.tolist(),
        "max_translational_residual_Ha_per_bohr": float(np.max(np.abs(trans))),
        "within_gradient_tolerance": bool(np.max(diff) < 1e-5),
        "within_translation_tolerance": bool(np.max(np.abs(trans)) < 1e-10),
        "elapsed_s": elapsed,
    }


def main() -> None:
    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [h2_molecule(), h2o_molecule(), co_molecule()]
    results = [run_case(mol) for mol in cases]
    status = all(r["within_gradient_tolerance"] and r["within_translation_tolerance"] for r in results)
    payload = {
        "phase": 18,
        "description": "RHF nuclear gradient validation",
        "basis": "STO-3G",
        "gradient_tolerance_Ha_per_bohr": 1e-5,
        "translation_tolerance_Ha_per_bohr": 1e-10,
        "status": "PASS" if status else "FAIL",
        "results": results,
    }

    json_path = out_dir / "phase18_grad.json"
    txt_path = out_dir / "phase18_grad.txt"
    json_path.write_text(json.dumps(payload, indent=2))

    lines = ["Phase 18 RHF gradient validation", "Basis: STO-3G", f"Status: {payload['status']}", ""]
    for result in results:
        lines.extend([
            result["molecule"],
            f"  E_total = {result['energy_total_Ha']:.12f} Ha",
            f"  max |analytic - numerical| = {result['max_abs_diff_Ha_per_bohr']:.6e} Ha/Bohr",
            f"  max translational residual = {result['max_translational_residual_Ha_per_bohr']:.6e} Ha/Bohr",
            f"  elapsed = {result['elapsed_s']:.2f} s",
            "",
        ])
    txt_path.write_text("\n".join(lines))
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    if not status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
