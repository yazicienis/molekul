"""Phase 11 CCSD validation log generator.

Runs MOLEKUL CCSD for H2 and H2O in STO-3G and compares correlation
energies with PySCF when available. If PySCF is unavailable, hardcoded PySCF
references are used.
"""

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
from molekul.ccsd import ccsd_energy  # noqa: E402
from molekul.constants import ANGSTROM_TO_BOHR  # noqa: E402
from molekul.molecule import Molecule  # noqa: E402
from molekul.rhf import rhf_scf  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    atoms_angstrom: tuple[tuple[str, tuple[float, float, float]], ...]
    charge: int
    multiplicity: int
    reference_ccsd_corr: float
    tolerance: float


CASES = (
    Case(
        name="H2",
        atoms_angstrom=(
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 0.74)),
        ),
        charge=0,
        multiplicity=1,
        reference_ccsd_corr=-0.020524691,
        tolerance=1e-6,
    ),
    Case(
        name="H2O",
        atoms_angstrom=(
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, -0.469)),
            ("H", (0.0, -0.757, -0.469)),
        ),
        charge=0,
        multiplicity=1,
        reference_ccsd_corr=-0.039957830712,
        tolerance=1e-5,
    ),
)


def build_molekul_molecule(case: Case) -> Molecule:
    atoms = [
        Atom(
            symbol,
            [coord * ANGSTROM_TO_BOHR for coord in coords],
        )
        for symbol, coords in case.atoms_angstrom
    ]
    return Molecule(atoms=atoms, charge=case.charge, multiplicity=case.multiplicity)


def pyscf_reference(case: Case) -> tuple[float, str]:
    try:
        from pyscf import cc, gto, scf
    except ImportError:
        return case.reference_ccsd_corr, "hardcoded PySCF reference"

    atom_spec = "; ".join(
        f"{symbol} {x:.12g} {y:.12g} {z:.12g}"
        for symbol, (x, y, z) in case.atoms_angstrom
    )
    mol = gto.M(
        atom=atom_spec,
        basis="sto-3g",
        charge=case.charge,
        spin=case.multiplicity - 1,
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    mycc = cc.CCSD(mf).run(verbose=0)
    return float(mycc.e_corr), "PySCF runtime reference"


def run_case(case: Case) -> dict[str, object]:
    mol = build_molekul_molecule(case)
    rhf = rhf_scf(mol, STO3G, verbose=False)
    result = ccsd_energy(mol, STO3G, rhf, verbose=False)
    reference, reference_source = pyscf_reference(case)
    diff = abs(result.energy_ccsd - reference)
    return {
        "molecule": case.name,
        "basis": "STO-3G",
        "E_corr_Ha": result.energy_ccsd,
        "E_total_Ha": result.energy_total,
        "ref_pyscf_Ha": reference,
        "reference_source": reference_source,
        "diff": diff,
        "tolerance": case.tolerance,
        "within_tolerance": diff < case.tolerance,
        "converged": result.converged,
        "n_iter": result.n_iter,
    }


def write_text_log(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "Phase 11 CCSD validation",
        f"Date: {payload['date']}",
        "",
    ]
    for result in payload["results"]:
        lines.extend(
            [
                f"{result['molecule']} / {result['basis']}",
                f"  E_corr       = {result['E_corr_Ha']:.12f} Ha",
                f"  PySCF ref    = {result['ref_pyscf_Ha']:.12f} Ha",
                f"  diff         = {result['diff']:.6e} Ha",
                f"  tolerance    = {result['tolerance']:.1e} Ha",
                f"  converged    = {result['converged']} ({result['n_iter']} iterations)",
                f"  ref source   = {result['reference_source']}",
                f"  pass         = {result['within_tolerance']}",
                "",
            ]
        )
    lines.extend(["Sign fixes:"])
    lines.extend(f"- {fix}" for fix in payload["sign_fixes"])
    lines.extend(["", f"Notes: {payload['notes']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = [run_case(case) for case in CASES]
    h2_result = next(result for result in results if result["molecule"] == "H2")
    notes = (
        "H2 is the acceptance reference from tests/test_ccsd.py. H2O is logged "
        "against the requested PySCF reference geometry for reviewer visibility."
    )
    payload = {
        "phase": 11,
        "description": "CCSD spin-orbital implementation",
        "date": date.today().isoformat(),
        "results": results,
        "sign_fixes": [
            "Wvvvv P-hat(ab) signs corrected in _make_intermediates_so",
            "Wovvo T1T1 coefficient corrected in _make_intermediates_so",
            "R1 ovov contraction sign corrected in _t1_residual_so",
            "R1 ovvv contraction sign corrected in _t1_residual_so",
            "T2 modified-F and final T1 antisymmetrisers corrected in _t2_residual_so",
        ],
        "notes": notes,
    }

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "phase11_ccsd.json"
    txt_path = log_dir / "phase11_ccsd.txt"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text_log(txt_path, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    for result in results:
        print(
            f"{result['molecule']}: E_corr={result['E_corr_Ha']:.12f} Ha, "
            f"diff={result['diff']:.6e} Ha, pass={result['within_tolerance']}"
        )

    return 0 if h2_result["diff"] < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
