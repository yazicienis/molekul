#!/usr/bin/env python3
"""Validate Phase 23 full periodic-HF phonons."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.periodic import Crystal, phonon_band_structure, phonon_band_structure_full


def h_chain() -> Crystal:
    a = 1.8
    return Crystal(
        lattice=np.array([[a, 0.0, 0.0]]),
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
        name="H chain",
    )


def main() -> int:
    crystal = h_chain()
    a = 1.8
    special_points = {"G": np.zeros(3), "X": np.array([np.pi / a, 0.0, 0.0])}
    full = phonon_band_structure_full(crystal, STO3G, special_points, "G-X", n_points=4)
    nuclear = phonon_band_structure(crystal, STO3G, special_points, "G-X", n_points=4)

    gamma_max = float(np.max(np.abs(full.frequencies[0])))
    x_full = full.frequencies[-1]
    x_nuclear = nuclear.frequencies[-1]
    x_diff = float(np.max(np.abs(x_full - x_nuclear)))
    status = "PASS" if gamma_max < 1e-3 and x_diff > 1e-3 and np.max(x_full) > 1e-3 else "FAIL"

    data = {
        "status": status,
        "system": "1D H chain, a=1.8 Bohr, STO-3G",
        "method": "periodic_hf total-energy finite differences; nuclear + electronic response",
        "finite_difference_h": 0.01,
        "path": full.tick_labels,
        "full": {
            "shape": list(full.frequencies.shape),
            "gamma_frequencies": full.frequencies[0].tolist(),
            "x_frequencies": x_full.tolist(),
            "gamma_max_frequency": gamma_max,
        },
        "nuclear_only_reference": {
            "x_frequencies": x_nuclear.tolist(),
            "scope": "Phase 22 nuclear-repulsion-only force constants",
        },
        "comparison": {
            "x_max_abs_diff": x_diff,
            "electronic_contribution_detected": bool(x_diff > 1e-3),
        },
    }

    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase23_phonons_full.json"
    txt_path = out_dir / "phase23_phonons_full.txt"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    lines = [
        "Phase 23 full phonons validation",
        f"Status: {status}",
        data["system"],
        data["method"],
        "",
        f"Path: {full.tick_labels}",
        f"h: {data['finite_difference_h']:.6f} Bohr",
        f"Full shape: {full.frequencies.shape}",
        f"Full Gamma frequencies: {full.frequencies[0].tolist()}",
        f"Full X frequencies: {x_full.tolist()}",
        f"Nuclear-only X frequencies: {x_nuclear.tolist()}",
        f"X max abs diff: {x_diff:.12f}",
        f"Gamma max frequency: {gamma_max:.12e}",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
