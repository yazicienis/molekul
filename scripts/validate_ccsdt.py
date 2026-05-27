"""Phase 14 CCSD(T) validation log generator."""

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
from molekul.ccsd import ccsdt_energy  # noqa: E402
from molekul.constants import ANGSTROM_TO_BOHR  # noqa: E402
from molekul.molecule import Molecule  # noqa: E402
from molekul.rhf import rhf_scf  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    atoms_angstrom: tuple[tuple[str, tuple[float, float, float]], ...]
    reference_ccsdt_corr: float
    tolerance: float


CASES = (
    Case(
        name="H2",
        atoms_angstrom=(
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 0.74)),
        ),
        reference_ccsdt_corr=-0.020524691214,
        tolerance=1e-6,
    ),
    Case(
        name="H2O",
        atoms_angstrom=(
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, -0.469)),
            ("H", (0.0, -0.757, -0.469)),
        ),
        reference_ccsdt_corr=-0.040032219581,
        tolerance=1e-5,
    ),
)


def build_molekul_molecule(case: Case) -> Molecule:
    return Molecule(
        atoms=[
            Atom(symbol, [coord * ANGSTROM_TO_BOHR for coord in coords])
            for symbol, coords in case.atoms_angstrom
        ],
        charge=0,
        multiplicity=1,
    )


def pyscf_reference(case: Case) -> tuple[float, str]:
    try:
        from pyscf import cc, gto, scf
    except ImportError:
        return case.reference_ccsdt_corr, "hardcoded PySCF reference"

    atom_spec = "; ".join(
        f"{symbol} {x:.12g} {y:.12g} {z:.12g}"
        for symbol, (x, y, z) in case.atoms_angstrom
    )
    mol = gto.M(atom=atom_spec, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    mycc = cc.CCSD(mf).run(verbose=0)
    return float(mycc.e_corr + mycc.ccsd_t()), "PySCF runtime reference"


def run_case(case: Case) -> dict[str, object]:
    mol = build_molekul_molecule(case)
    rhf = rhf_scf(mol, STO3G, verbose=False)
    result = ccsdt_energy(mol, STO3G, rhf, verbose=False)
    e_corr = result.energy_ccsd + result.energy_ccsdt
    reference, source = pyscf_reference(case)
    diff = abs(e_corr - reference)
    return {
        "molecule": case.name,
        "basis": "STO-3G",
        "E_ccsd_corr_Ha": result.energy_ccsd,
        "E_triples_Ha": result.energy_ccsdt,
        "E_ccsdt_corr_Ha": e_corr,
        "E_total_Ha": result.energy_total,
        "ref_pyscf_Ha": reference,
        "reference_source": source,
        "diff": diff,
        "tolerance": case.tolerance,
        "within_tolerance": diff < case.tolerance,
        "converged": result.converged,
        "n_iter": result.n_iter,
    }


def write_text_log(path: Path, payload: dict[str, object]) -> None:
    lines = ["Phase 14 CCSD(T) validation", f"Date: {payload['date']}", ""]
    for result in payload["results"]:
        lines.extend([
            f"{result['molecule']} / {result['basis']}",
            f"  E_CCSD corr   = {result['E_ccsd_corr_Ha']:.12f} Ha",
            f"  E_(T)         = {result['E_triples_Ha']:.12f} Ha",
            f"  E_CCSD(T) corr= {result['E_ccsdt_corr_Ha']:.12f} Ha",
            f"  PySCF ref     = {result['ref_pyscf_Ha']:.12f} Ha",
            f"  diff          = {result['diff']:.6e} Ha",
            f"  tolerance     = {result['tolerance']:.1e} Ha",
            f"  converged     = {result['converged']} ({result['n_iter']} iterations)",
            f"  ref source    = {result['reference_source']}",
            f"  pass          = {result['within_tolerance']}",
            "",
        ])
    lines.append(f"Notes: {payload['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = [run_case(case) for case in CASES]
    payload = {
        "phase": 14,
        "description": "CCSD(T) perturbative triples",
        "date": date.today().isoformat(),
        "results": results,
        "notes": "Runtime PySCF references are used when PySCF is installed; hardcoded references are fallbacks.",
    }

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "phase14_ccsdt.json"
    txt_path = log_dir / "phase14_ccsdt.txt"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text_log(txt_path, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    for result in results:
        print(
            f"{result['molecule']}: E_corr={result['E_ccsdt_corr_Ha']:.12f} Ha, "
            f"diff={result['diff']:.6e} Ha, pass={result['within_tolerance']}"
        )

    return 0 if all(result["within_tolerance"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
