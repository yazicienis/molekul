#!/usr/bin/env python3
"""Validate the optional CuPy backend used by Phase 19."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molekul.atoms import Atom
from molekul.backend import use_gpu
from molekul.basis_sto3g import STO3G
from molekul.ccsd import transform_mo_full
from molekul.eri import build_eri
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf


def h2o() -> Molecule:
    return Molecule(
        atoms=[
            Atom.from_angstrom("O", 0.000, 0.000, 0.000),
            Atom.from_angstrom("H", 0.000, 0.757, -0.586),
            Atom.from_angstrom("H", 0.000, -0.757, -0.586),
        ],
        charge=0,
        multiplicity=1,
        name="H2O",
    )


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> int:
    mol = h2o()
    rhf_cpu = rhf_scf(mol, STO3G)
    eri = build_eri(STO3G, mol)

    t0 = perf_counter()
    mo_cpu = transform_mo_full(eri, rhf_cpu.mo_coefficients)
    cpu_time = perf_counter() - t0

    data = {
        "backend_detected": "cupy" if cupy_available() else "numpy",
        "rhf_cpu_energy": rhf_cpu.energy_total,
        "rhf_gpu_energy": None,
        "energy_diff": None,
        "transform_cpu_seconds": cpu_time,
        "transform_gpu_seconds": None,
        "transform_max_diff": None,
        "status": "CPU_ONLY",
    }

    if cupy_available():
        with use_gpu():
            t0 = perf_counter()
            mo_gpu = transform_mo_full(eri, rhf_cpu.mo_coefficients)
            gpu_time = perf_counter() - t0
            rhf_gpu = rhf_scf(mol, STO3G)
        data.update({
            "rhf_gpu_energy": rhf_gpu.energy_total,
            "energy_diff": abs(rhf_cpu.energy_total - rhf_gpu.energy_total),
            "transform_gpu_seconds": gpu_time,
            "transform_max_diff": float(np.max(np.abs(mo_cpu - mo_gpu))),
            "status": "PASS" if abs(rhf_cpu.energy_total - rhf_gpu.energy_total) < 1e-10 else "FAIL",
        })

    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase19_gpu_backend.json"
    txt_path = out_dir / "phase19_gpu_backend.txt"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    lines = [
        "Phase 19 GPU backend validation",
        f"Status: {data['status']}",
        f"Backend detected: {data['backend_detected']}",
        f"H2O RHF CPU energy: {data['rhf_cpu_energy']:.12f}",
        f"transform_mo_full CPU seconds: {data['transform_cpu_seconds']:.6f}",
    ]
    if data["status"] == "CPU_ONLY":
        lines.append("GPU not available; CPU only.")
    else:
        lines.extend([
            f"H2O RHF GPU energy: {data['rhf_gpu_energy']:.12f}",
            f"Energy diff: {data['energy_diff']:.6e}",
            f"transform_mo_full GPU seconds: {data['transform_gpu_seconds']:.6f}",
            f"transform max diff: {data['transform_max_diff']:.6e}",
        ])
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0 if data["status"] in {"PASS", "CPU_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
