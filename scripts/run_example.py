#!/usr/bin/env python3
"""
Phase 1 example runner.

Reads XYZ files, prints molecular information, and saves a JSON log.
"""

import sys
import json
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molekul.io_xyz import read_xyz, write_xyz
from molekul.logging_utils import save_json_log, Timer, get_outputs_dir
from molekul.constants import HARTREE_TO_EV


def process_molecule(xyz_path: Path, charge: int = 0, multiplicity: int = 1) -> dict:
    timer = Timer()
    mol = read_xyz(xyz_path, charge=charge, multiplicity=multiplicity)
    e_nuc = mol.nuclear_repulsion_energy()
    elapsed = timer.elapsed()

    print(f"\n{'='*60}")
    print(mol)
    print(f"  Nuclear repulsion energy: {e_nuc:.8f} Hartree  ({e_nuc * 27.211396132:.6f} eV)")
    print(f"  Parsed in {elapsed*1000:.2f} ms")

    return {
        "name": mol.name,
        "source": str(xyz_path),
        "charge": mol.charge,
        "multiplicity": mol.multiplicity,
        "n_atoms": mol.n_atoms,
        "n_electrons": mol.n_electrons,
        "n_alpha": mol.n_alpha,
        "n_beta": mol.n_beta,
        "nuclear_repulsion_hartree": e_nuc,
        "nuclear_repulsion_ev": e_nuc * HARTREE_TO_EV,
        "atoms": [
            {"symbol": a.symbol, "Z": a.Z,
             "x_bohr": float(a.coords[0]),
             "y_bohr": float(a.coords[1]),
             "z_bohr": float(a.coords[2])}
            for a in mol.atoms
        ],
        "parse_time_ms": elapsed * 1000,
    }


def main():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    molecules = [
        (examples_dir / "h2.xyz",       0, 1),
        (examples_dir / "h2o.xyz",      0, 1),
        (examples_dir / "heh_plus.xyz", 1, 1),
    ]

    results = []
    for xyz_path, charge, mult in molecules:
        if not xyz_path.exists():
            print(f"WARNING: {xyz_path} not found, skipping.")
            continue
        result = process_molecule(xyz_path, charge=charge, multiplicity=mult)
        results.append(result)

    # Test roundtrip: write then re-read H2
    from molekul.io_xyz import read_xyz
    h2_mol = read_xyz(examples_dir / "h2.xyz")
    out_dir = get_outputs_dir()
    roundtrip_path = out_dir / "h2_roundtrip.xyz"
    write_xyz(h2_mol, roundtrip_path, comment="H2 roundtrip test")
    h2_reread = read_xyz(roundtrip_path)
    import numpy as np
    max_diff = np.max(np.abs(h2_mol.coords_bohr - h2_reread.coords_bohr))
    print(f"\nRoundtrip test (H2): max coord diff = {max_diff:.2e} Bohr  ({'PASS' if max_diff < 1e-8 else 'FAIL'})")

    log_path = save_json_log({"molecules": results}, "phase1_molecules")
    print(f"\nLog saved to: {log_path}")
    print("\nPhase 1 complete. Next: Phase 2 (basis sets + one-electron integrals).")

    # --- Dual-format log to outputs/logs/ -----------------------------------
    from molekul.logging_utils import ExperimentLogger
    exp = ExperimentLogger("phase1", "molecules")
    exp.metric("n_molecules", len(results))
    for r in results:
        key = r["name"].split()[0].lower().replace("+", "plus")
        exp.metric(f"{key}_n_electrons", r["n_electrons"])
        exp.metric(f"{key}_enuc_ha", r["nuclear_repulsion_hartree"])
    exp.metric("roundtrip_max_diff_bohr", float(max_diff))
    exp.check("H2 roundtrip", float(max_diff) < 1e-8)
    exp.artifact(str(log_path))
    exp.artifact(str(out_dir / "h2_roundtrip.xyz"))
    exp.line("Molecules processed: " + ", ".join(r["name"].split()[0] for r in results))
    exp.line(f"Roundtrip max coord diff: {max_diff:.2e} bohr")
    txt_path, json_path = exp.save()
    print(f"Structured log : {txt_path}\n                 {json_path}")


if __name__ == "__main__":
    main()
