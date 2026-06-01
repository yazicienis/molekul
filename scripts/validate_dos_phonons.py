#!/usr/bin/env python3
"""Validate Phase 22 DOS and nuclear-repulsion-only phonons."""

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
from molekul.periodic import Crystal, band_structure, dos, phonon_band_structure


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
    bands = band_structure(crystal, STO3G, special_points, "G-X", n_points=50)
    dos_result = dos(bands)
    phonons = phonon_band_structure(crystal, STO3G, special_points, "G-X")

    dos_integral = float(np.trapezoid(dos_result.dos, dos_result.energies))
    dos_peak_energy = float(dos_result.energies[int(np.argmax(dos_result.dos))])
    gamma_max = float(np.max(np.abs(phonons.frequencies[0])))
    x_max = float(np.max(phonons.frequencies[-1]))
    status = "PASS" if dos_integral > 0.0 and gamma_max < 1e-3 else "FAIL"

    data = {
        "status": status,
        "dos": {
            "n_grid": int(len(dos_result.energies)),
            "sigma": 0.02,
            "e_fermi": float(dos_result.e_fermi),
            "integral": dos_integral,
            "peak_energy": dos_peak_energy,
            "energy_min": float(dos_result.energies[0]),
            "energy_max": float(dos_result.energies[-1]),
        },
        "phonons": {
            "scope": "1D nuclear-repulsion-only force constants; electronic force constants omitted",
            "h": 0.01,
            "path": phonons.tick_labels,
            "shape": list(phonons.frequencies.shape),
            "gamma_max_frequency": gamma_max,
            "x_max_frequency": x_max,
        },
    }

    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase22_dos_phonons.json"
    txt_path = out_dir / "phase22_dos_phonons.txt"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    lines = [
        "Phase 22 DOS + phonons validation",
        f"Status: {status}",
        "",
        "DOS",
        f"  n_grid: {len(dos_result.energies)}",
        "  sigma: 0.020000 Ha",
        f"  Fermi level: {dos_result.e_fermi:.12f} Ha",
        f"  integral: {dos_integral:.12f}",
        f"  peak energy: {dos_peak_energy:.12f} Ha",
        "",
        "Phonons",
        "  scope: 1D nuclear-repulsion-only force constants; electronic force constants omitted",
        "  h: 0.010000 Bohr",
        f"  path: {phonons.tick_labels}",
        f"  shape: {phonons.frequencies.shape}",
        f"  Gamma max frequency: {gamma_max:.12e}",
        f"  X max frequency: {x_max:.12f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
