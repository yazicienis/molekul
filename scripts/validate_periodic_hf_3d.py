#!/usr/bin/env python3
"""Validate Phase 20c 3D periodic HF on rock-salt LiH."""

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
from molekul.periodic import Crystal, ewald_energy, monkhorst_pack, periodic_hf

FALLBACK = {
    "gamma": -17.613976509694957,
    "kmesh222": -18.521159535600447,
}


def lih_crystal() -> Crystal:
    a = 7.608
    return Crystal(
        lattice=np.eye(3) * a,
        atoms=[Atom("Li", [0.0, 0.0, 0.0]), Atom("H", [a / 2.0, a / 2.0, a / 2.0])],
        charge=0,
        multiplicity=1,
        name="LiH rock-salt",
    )


def pyscf_reference(mesh: tuple[int, int, int]):
    crystal = lih_crystal()
    info = {
        "available": False,
        "version": None,
        "energy": None,
        "error": None,
        "parameters": {
            "basis": "sto-3g",
            "lattice_bohr": crystal.lattice.tolist(),
            "atom_coords_bohr": [atom.coords.tolist() for atom in crystal.atoms],
            "mesh": list(mesh),
            "cell_precision": 1e-4,
            "cell_mesh": [9, 9, 9],
        },
    }
    try:
        import pyscf
        from pyscf.pbc import gto, scf

        info["version"] = pyscf.__version__
        cell = gto.Cell()
        cell.atom = "\n".join(
            f"{atom.symbol} {atom.coords[0] * BOHR_TO_ANGSTROM:.12f} "
            f"{atom.coords[1] * BOHR_TO_ANGSTROM:.12f} {atom.coords[2] * BOHR_TO_ANGSTROM:.12f}"
            for atom in crystal.atoms
        )
        cell.a = (crystal.lattice * BOHR_TO_ANGSTROM).tolist()
        cell.unit = "Angstrom"
        cell.basis = "sto-3g"
        cell.verbose = 0
        cell.precision = 1e-4
        cell.mesh = [9, 9, 9]
        cell.build()
        kpts = cell.make_kpts(mesh)
        mf = scf.RHF(cell) if len(kpts) == 1 else scf.KRHF(cell, kpts=kpts)
        mf.conv_tol = 1e-8
        mf.max_cycle = 100
        energy = float(mf.kernel())
        if np.isfinite(energy):
            info["available"] = True
            info["energy"] = energy
            info["converged"] = bool(mf.converged)
    except Exception as exc:
        info["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return info


def run_case(label: str, mesh: tuple[int, int, int]):
    crystal = lih_crystal()
    result = periodic_hf(crystal, STO3G, monkhorst_pack(crystal.lattice, mesh))
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
        "density_max_imag": float(np.max(np.abs(np.imag(result.density_matrix)))),
        "pyscf": ref,
    }


def main() -> int:
    crystal = lih_crystal()
    cases = [run_case("gamma", (1, 1, 1)), run_case("kmesh222", (2, 2, 2))]
    e_nn = ewald_energy(crystal)
    status = "PASS" if e_nn > 0.0 and all(c["abs_diff"] < 1e-2 and c["converged"] for c in cases) else "FAIL"
    data = {
        "status": status,
        "system": "LiH rock-salt, a=7.608 Bohr, STO-3G",
        "ewald_nn": e_nn,
        "fallback_references": FALLBACK,
        "cases": cases,
    }
    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase20c_periodic_hf_3d.json"
    txt_path = out_dir / "phase20c_periodic_hf_3d.txt"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    lines = [
        "Phase 20c periodic HF 3D validation",
        f"Status: {status}",
        data["system"],
        f"E_nn Ewald: {e_nn:.12f} Ha",
    ]
    for case in cases:
        lines.extend([
            "",
            case["label"],
            f"  mesh: {case['mesh']}",
            f"  MOLEKUL energy/cell: {case['energy_molekul']:.12f} Ha",
            f"  reference ({case['reference_source']}): {case['energy_reference']:.12f} Ha",
            f"  abs diff: {case['abs_diff']:.6e} Ha",
            f"  converged: {case['converged']} in {case['n_iter']} iterations",
            f"  density max imag: {case['density_max_imag']:.6e}",
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
