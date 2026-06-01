#!/usr/bin/env python3
"""Validate Phase 21 native tight-binding band structures."""

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
from molekul.periodic import Crystal, band_structure


def h_chain() -> Crystal:
    return Crystal(
        lattice=np.array([[1.8, 0.0, 0.0]]),
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
        name="H chain",
    )


def lih_crystal() -> Crystal:
    a = 7.608
    return Crystal(
        lattice=np.eye(3) * a,
        atoms=[Atom("Li", [0.0, 0.0, 0.0]), Atom("H", [a / 2.0, a / 2.0, a / 2.0])],
        charge=0,
        multiplicity=1,
        name="LiH rock-salt",
    )


def main() -> int:
    h_a = 1.8
    h_result = band_structure(
        h_chain(),
        STO3G,
        {"G": np.zeros(3), "X": np.array([np.pi / h_a, 0.0, 0.0])},
        "G-X",
        n_points=50,
    )
    a = 7.608
    lih_result = band_structure(
        lih_crystal(),
        STO3G,
        {
            "G": np.zeros(3),
            "X": np.array([np.pi / a, 0.0, 0.0]),
            "M": np.array([np.pi / a, np.pi / a, 0.0]),
        },
        "G-X-M-G",
        n_points=50,
    )

    h_band = h_result.band_energies[:, 0]
    h_band_width = float(np.ptp(h_band))
    occupied = lih_result.band_energies[:, : lih_result.n_occ]
    virtual = lih_result.band_energies[:, lih_result.n_occ :]
    lih_gap = float(np.min(virtual) - np.max(occupied))
    status = "PASS" if h_band_width > 1e-3 and lih_result.band_energies.shape == (150, 6) else "FAIL"

    data = {
        "status": status,
        "method": "native one-electron tight-binding H_core(k) generalized diagonalization",
        "h_chain": {
            "path": h_result.tick_labels,
            "n_kpts": len(h_result.kpoints),
            "band_width": h_band_width,
            "gamma_energy": float(h_band[0]),
            "x_energy": float(h_band[-1]),
        },
        "lih": {
            "path": lih_result.tick_labels,
            "tick_positions": lih_result.tick_positions,
            "shape": list(lih_result.band_energies.shape),
            "n_occ": lih_result.n_occ,
            "one_electron_gap": lih_gap,
        },
    }

    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase21_band_structure.json"
    txt_path = out_dir / "phase21_band_structure.txt"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    lines = [
        "Phase 21 native band-structure validation",
        f"Status: {status}",
        data["method"],
        "",
        "H chain",
        f"  path: {h_result.tick_labels}",
        f"  n_kpts: {len(h_result.kpoints)}",
        f"  band width: {h_band_width:.12f} Ha",
        f"  Gamma energy: {h_band[0]:.12f} Ha",
        f"  X energy: {h_band[-1]:.12f} Ha",
        "",
        "LiH rock-salt",
        f"  path: {lih_result.tick_labels}",
        f"  shape: {lih_result.band_energies.shape}",
        f"  n_occ: {lih_result.n_occ}",
        f"  one-electron gap: {lih_gap:.12f} Ha",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
