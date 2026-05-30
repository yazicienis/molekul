#!/usr/bin/env python3
"""Validate Phase 20b cutoff periodic HF on a 1D H chain."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.constants import BOHR_TO_ANGSTROM
from molekul.periodic import Crystal, monkhorst_pack, periodic_hf

FALLBACK = {
    "gamma": 0.08161353812017258,
    "kmesh4": 0.8487041965918238,
}


def h_chain() -> Crystal:
    return Crystal(
        lattice=np.array([[1.8, 0.0, 0.0]]),
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
        name="H chain",
    )


def pyscf_reference(mesh: tuple[int, ...]):
    info = {
        "available": False,
        "version": None,
        "energy": None,
        "error": None,
        "parameters": {
            "atom": "H 0 0 0",
            "a_angstrom": [
                [1.8 * BOHR_TO_ANGSTROM, 0.0, 0.0],
                [0.0, 20.0 * BOHR_TO_ANGSTROM, 0.0],
                [0.0, 0.0, 20.0 * BOHR_TO_ANGSTROM],
            ],
            "basis": "sto-3g",
            "dimension": 1,
            "low_dim_ft_type": "inf_vacuum",
            "mesh": list(mesh),
        },
    }
    try:
        import pyscf
        from pyscf.pbc import gto, scf
        info["version"] = pyscf.__version__
        cell = gto.Cell()
        cell.atom = "H 0 0 0"
        cell.a = info["parameters"]["a_angstrom"]
        cell.basis = "sto-3g"
        cell.dimension = 1
        cell.low_dim_ft_type = "inf_vacuum"
        cell.verbose = 0
        cell.build()
        if mesh == (1,):
            mf = scf.RHF(cell)
        else:
            mf = scf.KRHF(cell, kpts=cell.make_kpts([mesh[0], 1, 1]))
        energy = float(mf.kernel())
        if np.isfinite(energy):
            info["available"] = True
            info["energy"] = energy
    except Exception as exc:
        info["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return info


def run_case(label: str, mesh: tuple[int, ...]):
    crystal = h_chain()
    kpoints = monkhorst_pack(crystal.lattice, mesh)
    result = periodic_hf(crystal, STO3G, kpoints)
    ref = pyscf_reference(mesh)
    fallback = FALLBACK[label]
    reference = ref["energy"] if ref["available"] else fallback
    return {
        "label": label,
        "mesh": list(mesh),
        "energy_molekul": result.energy_per_cell,
        "energy_reference": reference,
        "reference_source": "pyscf" if ref["available"] else "fallback",
        "abs_diff": abs(result.energy_per_cell - reference),
        "converged": result.converged,
        "n_iter": result.n_iter,
        "band_energies": result.band_energies.tolist(),
        "kpoints_bohr_inv": result.kpoints.tolist(),
        "pyscf": ref,
    }


def main() -> int:
    cases = [run_case("gamma", (1,)), run_case("kmesh4", (4,))]
    status = "PASS" if all(c["abs_diff"] < 1e-3 and c["converged"] for c in cases) else "FAIL"
    data = {
        "status": status,
        "convention": "Phase 20b uses Phase 20a per-cell V_ne and finite real-space cutoff; Ewald deferred to Phase 20c.",
        "fallback_references": FALLBACK,
        "cases": cases,
    }
    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase20b_periodic_hf_1d.json"
    txt_path = out_dir / "phase20b_periodic_hf_1d.txt"
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    lines = ["Phase 20b periodic HF 1D validation", f"Status: {status}", data["convention"]]
    for case in cases:
        lines.extend([
            "",
            case["label"],
            f"  mesh: {case['mesh']}",
            f"  MOLEKUL energy/cell: {case['energy_molekul']:.12f} Ha",
            f"  reference ({case['reference_source']}): {case['energy_reference']:.12f} Ha",
            f"  abs diff: {case['abs_diff']:.6e} Ha",
            f"  converged: {case['converged']} in {case['n_iter']} iterations",
            f"  band energies: {case['band_energies']}",
        ])
        if not case["pyscf"]["available"]:
            lines.append(f"  PySCF unavailable/failed: {case['pyscf']['error']}")
        else:
            lines.append(f"  PySCF version: {case['pyscf']['version']}")
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
