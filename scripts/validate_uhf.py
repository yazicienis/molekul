"""Phase 16 UHF validation log generator."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molekul.atoms import Atom  # noqa: E402
from molekul.basis_sto3g import STO3G  # noqa: E402
from molekul.constants import ANGSTROM_TO_BOHR  # noqa: E402
from molekul.molecule import Molecule  # noqa: E402
from molekul.rhf import rhf_scf  # noqa: E402
from molekul.uhf import uhf_scf  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    atoms_angstrom: tuple[tuple[str, tuple[float, float, float]], ...]
    charge: int
    multiplicity: int
    reference_energy: float
    tolerance: float


CASES = (
    Case(
        name="H2O",
        atoms_angstrom=(
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.586)),
            ("H", (0.0, -0.757, 0.586)),
        ),
        charge=0,
        multiplicity=1,
        reference_energy=-74.96294665654,
        tolerance=1e-8,
    ),
    Case(
        name="OH",
        atoms_angstrom=(
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 0.96966)),
        ),
        charge=0,
        multiplicity=2,
        reference_energy=-74.3626332768421,
        tolerance=1e-4,
    ),
    Case(
        name="H",
        atoms_angstrom=(("H", (0.0, 0.0, 0.0)),),
        charge=0,
        multiplicity=2,
        reference_energy=-0.46658184955727533,
        tolerance=1e-5,
    ),
)


def build_molekul_molecule(case: Case) -> Molecule:
    return Molecule(
        atoms=[
            Atom(symbol, [coord * ANGSTROM_TO_BOHR for coord in coords])
            for symbol, coords in case.atoms_angstrom
        ],
        charge=case.charge,
        multiplicity=case.multiplicity,
    )


def pyscf_reference(case: Case) -> tuple[float, float, str]:
    try:
        from pyscf import gto, scf
    except ImportError:
        return case.reference_energy, float("nan"), "hardcoded PySCF reference"

    atom_spec = "; ".join(
        f"{symbol} {x:.12g} {y:.12g} {z:.12g}"
        for symbol, (x, y, z) in case.atoms_angstrom
    )
    mol = gto.M(
        atom=atom_spec,
        basis="sto-3g",
        unit="Angstrom",
        charge=case.charge,
        spin=case.multiplicity - 1,
        verbose=0,
    )
    mf = scf.UHF(mol).run(verbose=0)
    return float(mf.e_tot), float(mf.spin_square()[0]), "PySCF runtime UHF reference"


def run_case(case: Case) -> dict[str, object]:
    mol = build_molekul_molecule(case)
    result = uhf_scf(mol, STO3G, verbose=False)
    reference, ref_s2, source = pyscf_reference(case)
    diff = abs(result.energy_total - reference)
    payload = {
        "molecule": case.name,
        "basis": "STO-3G",
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "energy_total_Ha": float(result.energy_total),
        "S2": float(result.S2),
        "ref_pyscf_Ha": float(reference),
        "ref_pyscf_S2": float(ref_s2),
        "reference_source": source,
        "diff": float(diff),
        "tolerance": float(case.tolerance),
        "within_tolerance": bool(diff < case.tolerance),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
    }
    if case.name == "H2O":
        rhf = rhf_scf(mol, STO3G, verbose=False)
        payload["rhf_energy_total_Ha"] = float(rhf.energy_total)
        payload["uhf_minus_rhf_Ha"] = float(result.energy_total - rhf.energy_total)
        payload["matches_rhf"] = bool(abs(result.energy_total - rhf.energy_total) < 1e-8)
    return payload


def write_text_log(path: Path, payload: dict[str, object]) -> None:
    lines = ["Phase 16 UHF validation", f"Date: {payload['date']}", ""]
    for result in payload["results"]:
        lines.extend([
            f"{result['molecule']} / {result['basis']} charge={result['charge']} mult={result['multiplicity']}",
            f"  UHF total   = {result['energy_total_Ha']:.12f} Ha",
            f"  PySCF ref   = {result['ref_pyscf_Ha']:.12f} Ha",
            f"  diff        = {result['diff']:.6e} Ha",
            f"  tolerance   = {result['tolerance']:.1e} Ha",
            f"  <S^2>       = {result['S2']:.12f}",
            f"  PySCF <S^2> = {result['ref_pyscf_S2']:.12f}",
            f"  converged   = {result['converged']} ({result['n_iter']} iterations)",
            f"  ref source  = {result['reference_source']}",
            f"  pass        = {result['within_tolerance']}",
            "",
        ])
    lines.append(f"Notes: {payload['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = [run_case(case) for case in CASES]
    payload = {
        "phase": 16,
        "description": "Unrestricted Hartree-Fock",
        "date": date.today().isoformat(),
        "results": results,
        "notes": "Runtime PySCF UHF references are used when PySCF is installed; hardcoded references are fallbacks.",
    }

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "phase16_uhf.json"
    txt_path = log_dir / "phase16_uhf.txt"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text_log(txt_path, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    for result in results:
        print(
            f"{result['molecule']}: E={result['energy_total_Ha']:.12f} Ha, "
            f"diff={result['diff']:.6e} Ha, S2={result['S2']:.6f}, "
            f"pass={result['within_tolerance']}"
        )

    return 0 if all(result["within_tolerance"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
