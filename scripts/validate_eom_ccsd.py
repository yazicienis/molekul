"""Phase 15 EOM-CCSD validation log generator."""

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
from molekul.eom_ccsd import eom_ccsd_ee  # noqa: E402
from molekul.molecule import Molecule  # noqa: E402
from molekul.rhf import rhf_scf  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    atoms_angstrom: tuple[tuple[str, tuple[float, float, float]], ...]
    reference_state1: float
    tolerance: float


CASES = (
    Case(
        name="H2",
        atoms_angstrom=(
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 0.74)),
        ),
        reference_state1=0.96893157,
        tolerance=0.001,
    ),
    Case(
        name="H2O",
        atoms_angstrom=(
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.586)),
            ("H", (0.0, -0.757, 0.586)),
        ),
        reference_state1=0.45711995,
        tolerance=0.001,
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


def pyscf_reference(case: Case, n_states: int) -> tuple[list[float], str]:
    try:
        from pyscf import cc, gto, scf
    except ImportError:
        return [case.reference_state1], "hardcoded PySCF reference"

    atom_spec = "; ".join(
        f"{symbol} {x:.12g} {y:.12g} {z:.12g}"
        for symbol, (x, y, z) in case.atoms_angstrom
    )
    mol = gto.M(atom=atom_spec, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    mycc = cc.CCSD(mf).run(verbose=0)
    roots, _ = mycc.eomee_ccsd_singlet(nroots=n_states)
    return [float(root) for root in roots], "PySCF runtime EOM-CCSD singlet reference"


def run_case(case: Case) -> dict[str, object]:
    mol = build_molekul_molecule(case)
    rhf = rhf_scf(mol, STO3G, verbose=False)
    result = eom_ccsd_ee(mol, STO3G, rhf, n_states=5, verbose=False)
    reference, source = pyscf_reference(case, min(5, result.n_states))
    diff = abs(float(result.excitation_energies[0]) - reference[0])
    return {
        "molecule": case.name,
        "basis": "STO-3G",
        "excitation_energies_Ha": result.excitation_energies.tolist(),
        "excitation_energies_eV": result.excitation_eV.tolist(),
        "ref_pyscf_Ha": reference,
        "reference_source": source,
        "state1_diff": diff,
        "tolerance": case.tolerance,
        "within_tolerance": diff < case.tolerance,
        "n_states": result.n_states,
    }


def write_text_log(path: Path, payload: dict[str, object]) -> None:
    lines = ["Phase 15 EOM-CCSD validation", f"Date: {payload['date']}", ""]
    for result in payload["results"]:
        lines.extend([
            f"{result['molecule']} / {result['basis']}",
            f"  EOM-CCSD state 1 = {result['excitation_energies_Ha'][0]:.12f} Ha",
            f"  PySCF state 1    = {result['ref_pyscf_Ha'][0]:.12f} Ha",
            f"  diff             = {result['state1_diff']:.6e} Ha",
            f"  tolerance        = {result['tolerance']:.1e} Ha",
            f"  ref source       = {result['reference_source']}",
            f"  pass             = {result['within_tolerance']}",
            "",
        ])
    lines.append(f"Notes: {payload['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = [run_case(case) for case in CASES]
    payload = {
        "phase": 15,
        "description": "EOM-CCSD-EE singlet excitation energies",
        "date": date.today().isoformat(),
        "results": results,
        "notes": "Runtime PySCF EOM-CCSD singlet references are used when PySCF is installed; hardcoded references are fallbacks.",
    }

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "phase15_eom_ccsd.json"
    txt_path = log_dir / "phase15_eom_ccsd.txt"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_text_log(txt_path, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    for result in results:
        print(
            f"{result['molecule']}: state1={result['excitation_energies_Ha'][0]:.12f} Ha, "
            f"diff={result['state1_diff']:.6e} Ha, pass={result['within_tolerance']}"
        )

    return 0 if all(result["within_tolerance"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
