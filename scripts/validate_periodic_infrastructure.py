#!/usr/bin/env python3
"""Validate Phase 20a periodic infrastructure."""

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
from molekul.integrals import build_core_hamiltonian, build_overlap
from molekul.molecule import Molecule
from molekul.periodic import Crystal, bloch_hcore, bloch_overlap, monkhorst_pack


def h2_molecule() -> Molecule:
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
        name="H2",
    )


def main() -> int:
    mol = h2_molecule()
    crystal = Crystal(
        lattice=np.diag([20.0, 20.0, 20.0]),
        atoms=mol.atoms,
        charge=0,
        multiplicity=1,
        name="H2 box",
    )
    gamma = np.zeros((1, 3))

    S_periodic = bloch_overlap(crystal, STO3G, gamma)[0]
    S_molecular = build_overlap(STO3G, mol)
    H_periodic = bloch_hcore(crystal, STO3G, gamma)[0]
    H_molecular = build_core_hamiltonian(STO3G, mol)

    s_diff = float(np.max(np.abs(S_periodic - S_molecular)))
    h_diff = float(np.max(np.abs(H_periodic - H_molecular)))
    k_1d = monkhorst_pack(np.array([[2.0, 0.0, 0.0]]), (4,))
    k_3d = monkhorst_pack(np.eye(3) * 5.0, (2, 2, 2))

    data = {
        "molecule": "H2",
        "lattice_bohr": crystal.lattice.tolist(),
        "r_max_factor": 4.0,
        "overlap_gamma_max_diff": s_diff,
        "hcore_gamma_max_diff": h_diff,
        "overlap_pass": s_diff < 1e-10,
        "hcore_pass": h_diff < 1e-8,
        "monkhorst_pack_1d_mesh_4": k_1d.tolist(),
        "monkhorst_pack_3d_mesh_2_2_2": k_3d.tolist(),
    }
    data["status"] = "PASS" if data["overlap_pass"] and data["hcore_pass"] else "FAIL"

    out_dir = ROOT / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase20a_periodic_infrastructure.json"
    txt_path = out_dir / "phase20a_periodic_infrastructure.txt"
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    lines = [
        "Phase 20a periodic infrastructure validation",
        f"Status: {data['status']}",
        f"H2 box lattice: {crystal.lattice.tolist()} Bohr",
        f"S(Gamma) max diff vs molecular S: {s_diff:.6e}",
        f"H_core(Gamma) max diff vs molecular H_core: {h_diff:.6e}",
        f"1D MP mesh=(4,) k-points: {k_1d.tolist()}",
        f"3D MP mesh=(2,2,2) k-points: {k_3d.tolist()}",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0 if data["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
