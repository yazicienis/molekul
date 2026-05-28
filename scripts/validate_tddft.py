"""Phase 17 TD-DFT/TDA validation log generator."""

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
from molekul.tddft import tddft_tda  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    atoms_angstrom: tuple[tuple[str, tuple[float, float, float]], ...]
    functional: str
    reference_state1: float
    tolerance: float


CASES = (
    Case(
        name="H2",
        atoms_angstrom=(
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 0.74)),
        ),
        functional="lda",
        reference_state1=0.97176243,
        tolerance=0.01,
    ),
    Case(
        name="H2O",
        atoms_angstrom=(
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.586)),
            ("H", (0.0, -0.757, 0.586)),
        ),
        functional="lda",
        reference_state1=0.42245348,
        tolerance=0.01,
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


def pyscf_reference(case: Case, n_states: int) -> tuple[list[float], list[float], str]:
    try:
        from pyscf import dft, gto, tddft
    except ImportError:
        return [case.reference_state1], [], "hardcoded PySCF reference"

    atom_spec = "; ".join(
        f"{symbol} {x:.12g} {y:.12g} {z:.12g}"
        for symbol, (x, y, z) in case.atoms_angstrom
    )
    mol = gto.M(atom=atom_spec, basis="sto-3g", unit="Angstrom", verbose=0)
    ks = dft.RKS(mol)
    ks.xc = "lda,vwn" if case.functional == "lda" else case.functional
    ks.grids.level = 3
    ks.run(verbose=0)
    td = tddft.TDA(ks).run(nstates=n_states, verbose=0)
    return (
        [float(root) for root in td.e],
        [float(osc) for osc in td.oscillator_strength()],
        "PySCF runtime TDA reference",
    )


def run_case(case: Case) -> dict[str, object]:
    mol = build_molekul_molecule(case)
    result = tddft_tda(mol, STO3G, functional=case.functional, n_states=5, verbose=False)
    reference, ref_osc, source = pyscf_reference(case, min(5, result.n_states))
    diff = abs(float(result.excitation_energies[0]) - reference[0])
    return {
        "molecule": case.name,
        "basis": "STO-3G",
        "functional": case.functional,
        "excitation_energies_Ha": result.excitation_energies.tolist(),
        "excitation_energies_eV": result.excitation_eV.tolist(),
        "oscillator_strengths": result.oscillator_strengths.tolist(),
        "ref_pyscf_Ha": reference,
        "ref_pyscf_oscillator_strengths": ref_osc,
        "reference_source": source,
        "state1_diff": float(diff),
        "tolerance": float(case.tolerance),
        "within_tolerance": bool(diff < case.tolerance),
        "n_states": int(result.n_states),
    }


def write_text_log(path: Path, payload: dict[str, object]) -> None:
    lines = ["Phase 17 TD-DFT/TDA validation", f"Date: {payload['date']}", ""]
    for result in payload["results"]:
        lines.extend([
            f"{result['molecule']} / {result['basis']} / {result['functional'].upper()}",
            f"  TDA state 1 = {result['excitation_energies_Ha'][0]:.12f} Ha",
            f"  PySCF state 1 = {result['ref_pyscf_Ha'][0]:.12f} Ha",
            f"  diff = {result['state1_diff']:.6e} Ha",
            f"  tolerance = {result['tolerance']:.1e} Ha",
            f"  ref source = {result['reference_source']}",
            f"  pass = {result['within_tolerance']}",
            "",
        ])
    lines.append(f"Notes: {payload['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = [run_case(case) for case in CASES]
    payload = {
        "phase": 17,
        "description": "TD-DFT Tamm-Dancoff approximation",
        "date": date.today().isoformat(),
        "results": results,
        "notes": "Runtime PySCF TDA references are used when PySCF is installed; hardcoded references are fallbacks.",
    }

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "phase17_tddft.json"
    txt_path = log_dir / "phase17_tddft.txt"
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
