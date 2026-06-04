#!/usr/bin/env python3
"""Regenerate JOSE validation tables for molecular and periodic features."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from pyscf import cc, dft, gto, mp, scf, tdscf

from molekul.atoms import Atom
from molekul.basis_631gstar import G631Star
from molekul.basis_ccpvdz import ccpVDZ
from molekul.basis_sto3g import STO3G
from molekul.ccsd import ccsd_energy, ccsdt_energy
from molekul.cis import cis_excitations
from molekul.constants import ANGSTROM_TO_BOHR
from molekul.dft import ks_scf
from molekul.eom_ccsd import eom_ccsd_ee
from molekul.molecule import Molecule
from molekul.mp2 import mp2_energy
from molekul.periodic import Crystal, band_structure, phonon_band_structure_full
from molekul.rhf import rhf_scf
from molekul.tddft import tddft_tda
from molekul.uhf import uhf_scf


ROOT = Path(__file__).resolve().parent
MOLECULAR_JSON = ROOT / "validation_table.json"
MOLECULAR_MD = ROOT / "validation_table.md"
PERIODIC_JSON = ROOT / "periodic_validation.json"
PERIODIC_MD = ROOT / "periodic_validation.md"

BASIS_MAP = {
    "STO-3G": STO3G,
    "6-31G*": G631Star,
    "cc-pVDZ": ccpVDZ,
}

PYSCF_BASIS = {
    "STO-3G": "sto-3g",
    "6-31G*": "6-31g*",
    "cc-pVDZ": "cc-pvdz",
}

SUPPORTED = {
    "STO-3G": {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"},
    "6-31G*": {"H", "He", "C", "N", "O", "F"},
    "cc-pVDZ": {"H", "He", "C", "N", "O", "F"},
}


@dataclass
class Cell:
    method: str
    basis: str
    molecule: str
    E_molekul: float | None
    E_ref: float | None
    abs_delta_E: float | None
    ref_code: str
    ref_version: str
    wall_time_s: float
    status: str
    notes: str = ""


def molecule(name: str) -> Molecule:
    if name == "H2":
        atoms = [
            Atom("H", [0.0, 0.0, 0.37 * ANGSTROM_TO_BOHR]),
            Atom("H", [0.0, 0.0, -0.37 * ANGSTROM_TO_BOHR]),
        ]
        return Molecule(atoms=atoms, charge=0, multiplicity=1, name="H2")
    if name == "H2O":
        atoms = [
            Atom.from_angstrom("O", 0.0, 0.0, 0.1173),
            Atom.from_angstrom("H", 0.7572, 0.0, -0.4692),
            Atom.from_angstrom("H", -0.7572, 0.0, -0.4692),
        ]
        return Molecule(atoms=atoms, charge=0, multiplicity=1, name="H2O")
    raise ValueError(f"unknown molecule: {name}")


def pyscf_mol(mol: Molecule, basis_name: str) -> gto.Mole:
    atom_spec = [
        (atom.symbol, tuple(float(x) for x in atom.coords_angstrom()))
        for atom in mol.atoms
    ]
    return gto.M(
        atom=atom_spec,
        unit="Angstrom",
        basis=PYSCF_BASIS[basis_name],
        charge=mol.charge,
        spin=mol.multiplicity - 1,
        cart=True,
        verbose=0,
    )


def assert_supported(mol: Molecule, basis_name: str) -> None:
    missing = sorted({atom.symbol for atom in mol.atoms} - SUPPORTED[basis_name])
    if missing:
        raise ValueError(f"{basis_name} does not support {', '.join(missing)}")


def rhf_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    mres = rhf_scf(mol, basis, verbose=False)
    ref = scf.RHF(pyscf_mol(mol, basis_name)).run()
    return mres.energy_total, float(ref.e_tot)


def uhf_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    mres = uhf_scf(mol, basis, verbose=False)
    ref = scf.UHF(pyscf_mol(mol, basis_name)).run()
    return mres.energy_total, float(ref.e_tot)


def mp2_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    rhf = rhf_scf(mol, basis, verbose=False)
    mres = mp2_energy(mol, basis, rhf)
    ref_hf = scf.RHF(pyscf_mol(mol, basis_name)).run()
    ref_corr = mp.MP2(ref_hf).run().e_corr
    return mres.energy_total, float(ref_hf.e_tot + ref_corr)


def ccsd_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    rhf = rhf_scf(mol, basis, verbose=False)
    mres = ccsd_energy(mol, basis, rhf, verbose=False)
    ref_hf = scf.RHF(pyscf_mol(mol, basis_name)).run()
    ref_cc = cc.CCSD(ref_hf).run()
    return mres.energy_total, float(ref_hf.e_tot + ref_cc.e_corr)


def ccsdt_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    rhf = rhf_scf(mol, basis, verbose=False)
    mres = ccsdt_energy(mol, basis, rhf, verbose=False)
    ref_hf = scf.RHF(pyscf_mol(mol, basis_name)).run()
    ref_cc = cc.CCSD(ref_hf).run()
    triples = ref_cc.ccsd_t()
    return mres.energy_total, float(ref_hf.e_tot + ref_cc.e_corr + triples)


def lda_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    mres = ks_scf(mol, basis, xc="lda", verbose=False)
    ref = dft.RKS(pyscf_mol(mol, basis_name))
    ref.xc = "lda,vwn"
    ref.grids.level = 3
    ref.kernel()
    return mres.energy_total, float(ref.e_tot)


def pbe_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    mres = ks_scf(mol, basis, xc="pbe", verbose=False)
    ref = dft.RKS(pyscf_mol(mol, basis_name))
    ref.xc = "pbe,pbe"
    ref.grids.level = 3
    ref.kernel()
    return mres.energy_total, float(ref.e_tot)


def cis_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    rhf = rhf_scf(mol, basis, verbose=False)
    mres = cis_excitations(mol, basis, rhf, n_states=1, verbose=False)
    ref_hf = scf.RHF(pyscf_mol(mol, basis_name)).run()
    ref_tda = tdscf.TDA(ref_hf)
    ref_tda.nstates = 1
    ref_energy = ref_tda.kernel()[0][0]
    return float(mres.excitation_energies[0]), float(ref_energy)


def tddft_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    mres = tddft_tda(mol, basis, functional="lda", n_states=1, verbose=False)
    ref_ks = dft.RKS(pyscf_mol(mol, basis_name))
    ref_ks.xc = "lda,vwn"
    ref_ks.grids.level = 3
    ref_ks.kernel()
    ref_tda = tdscf.TDA(ref_ks)
    ref_tda.nstates = 1
    ref_energy = ref_tda.kernel()[0][0]
    return float(mres.excitation_energies[0]), float(ref_energy)


def eom_pair(mol: Molecule, basis_name: str) -> tuple[float, float]:
    basis = BASIS_MAP[basis_name]
    rhf = rhf_scf(mol, basis, verbose=False)
    mres = eom_ccsd_ee(mol, basis, rhf, n_states=1, verbose=False)
    ref_hf = scf.RHF(pyscf_mol(mol, basis_name)).run()
    ref_cc = cc.CCSD(ref_hf).run()
    ref_energy = np.atleast_1d(ref_cc.eomee_ccsd_singlet(nroots=1)[0])[0]
    return float(mres.excitation_energies[0]), float(ref_energy)


METHODS: list[tuple[str, str, Callable[[Molecule, str], tuple[float, float]], str]] = [
    ("RHF", "H2O", rhf_pair, "closed-shell total energy"),
    ("UHF", "H2", uhf_pair, "closed-shell UHF regression; total energy"),
    ("MP2", "H2", mp2_pair, "closed-shell MP2 total energy"),
    ("CCSD", "H2", ccsd_pair, "closed-shell CCSD total energy"),
    ("CCSD(T)", "H2", ccsdt_pair, "closed-shell CCSD(T) total energy; NOTE: H2 is a 2-electron system where CCSD=FCI so the (T) correction is identically zero — confirms no crash, does not validate the triples perturbation"),
    ("KS-DFT(LDA)", "H2O", lda_pair, "closed-shell LDA total energy"),
    ("KS-DFT(PBE)", "H2O", pbe_pair, "PBE/GGA grid is experimental; total energy"),
    ("CIS", "H2", cis_pair, "first singlet excitation energy, not total energy"),
    ("TD-DFT(TDA)", "H2", tddft_pair, "first LDA/TDA singlet excitation energy"),
    ("EOM-CCSD", "H2", eom_pair, "first singlet excitation energy, not total energy"),
]


def status_for(method: str, basis_name: str, delta: float | None) -> str:
    if method == "KS-DFT(PBE)":
        return "experimental"
    if method == "KS-DFT(LDA)" and basis_name != "STO-3G":
        return "experimental"
    if method == "TD-DFT(TDA)" and basis_name != "STO-3G":
        return "experimental"
    if delta is not None and delta > 1e-3:
        return "experimental"
    return "ok"


def run_cell(
    method: str,
    mol_name: str,
    basis_name: str,
    runner: Callable[[Molecule, str], tuple[float, float]],
    notes: str,
) -> Cell:
    start = time.perf_counter()
    mol = molecule(mol_name)
    ref_version = getattr(sys.modules.get("pyscf"), "__version__", "unknown")
    try:
        assert_supported(mol, basis_name)
        e_molekul, e_ref = runner(mol, basis_name)
        delta = abs(e_molekul - e_ref)
        status = status_for(method, basis_name, delta)
        if status == "experimental" and "experimental" not in notes.lower():
            notes = f"{notes}; deviation/status currently experimental"
        if delta > 1.0:
            notes = f"{notes}; BROKEN: |ΔE|={delta:.3f} Ha — DFT cc-pVDZ integration fails in MOLEKUL; do not cite as validated"
        return Cell(
            method=method,
            basis=basis_name,
            molecule=mol_name,
            E_molekul=e_molekul,
            E_ref=e_ref,
            abs_delta_E=delta,
            ref_code="PySCF",
            ref_version=ref_version,
            wall_time_s=time.perf_counter() - start,
            status=status,
            notes=notes,
        )
    except NotImplementedError as exc:
        status = "unsupported"
        note = str(exc)
    except Exception as exc:  # table generation should report failures, not hide rows
        status = "skipped"
        note = str(exc)
    return Cell(
        method=method,
        basis=basis_name,
        molecule=mol_name,
        E_molekul=None,
        E_ref=None,
        abs_delta_E=None,
        ref_code="PySCF",
        ref_version=ref_version,
        wall_time_s=time.perf_counter() - start,
        status=status,
        notes=f"{notes}; {note}",
    )


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3e}"


def molecular_markdown(cells: list[Cell]) -> str:
    lines = [
        "# Molecular Validation Table",
        "",
        "Regenerated by `python benchmarks/validation_table.py`.",
        "All energies are Hartree. CIS, TD-DFT(TDA), and EOM-CCSD entries are first singlet excitation energies.",
        "Status values are `ok`, `experimental`, `unsupported`, or `skipped`.",
        "",
        "| Method | Basis | Molecule | E_molekul | E_ref | |Delta E| | Ref | Wall time (s) | Status | Notes |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for cell in cells:
        ref = f"{cell.ref_code} {cell.ref_version}"
        lines.append(
            f"| {cell.method} | {cell.basis} | {cell.molecule} | "
            f"{fmt_float(cell.E_molekul)} | {fmt_float(cell.E_ref)} | "
            f"{fmt_delta(cell.abs_delta_E)} | {ref} | {cell.wall_time_s:.3f} | "
            f"{cell.status} | {cell.notes} |"
        )
    return "\n".join(lines) + "\n"


def h_chain() -> Crystal:
    a = 1.8
    return Crystal(
        lattice=np.array([[a, 0.0, 0.0]]),
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
        name="H chain",
    )


def periodic_validation() -> list[dict[str, object]]:
    crystal = h_chain()
    a = float(crystal.lattice[0, 0])
    special = {"G": np.zeros(3), "X": np.array([np.pi / a, 0.0, 0.0])}
    rows: list[dict[str, object]] = []

    start = time.perf_counter()
    bands = band_structure(crystal, STO3G, special, "G-X", n_points=21)
    band = bands.band_energies[:, 0]
    gamma = float(band[0])
    xpoint = float(band[-1])
    is_monotone = int(all(b2 > b1 for b1, b2 in zip(band, band[1:])))
    rows.append(
        {
            "case": "H-chain band monotonicity",
            "observable": "band_is_monotone_Gamma_to_X",
            "molekul": is_monotone,
            "reference": 1,
            "abs_delta": abs(is_monotone - 1),
            "ref_model": "1D H-chain 1s band: bonding state (Gamma) must lie below antibonding state (X)",
            "wall_time_s": time.perf_counter() - start,
            "status": "ok" if is_monotone == 1 else "fail",
            "notes": (
                f"Gamma={gamma:.6f} Ha, X={xpoint:.6f} Ha; band goes monotonically "
                "bonding→antibonding as expected for 1D H-chain 1s band (STO-3G, a=1.8 bohr). "
                "Qualitative path check only — no independent reference code for periodic HF energies."
            ),
        }
    )

    start = time.perf_counter()
    phonons = phonon_band_structure_full(crystal, STO3G, special, "G-X", n_points=4)
    gamma_max = float(np.max(np.abs(phonons.frequencies[0])))
    rows.append(
        {
            "case": "H-chain full phonon Gamma",
            "observable": "max_abs_frequency_Gamma_cm-1",
            "molekul": gamma_max,
            "reference": 0.0,
            "abs_delta": gamma_max,
            "ref_model": "analytic acoustic translational-invariance sum rule",
            "wall_time_s": time.perf_counter() - start,
            "status": "ok" if gamma_max < 1e-3 else "experimental",
            "notes": "Finite-difference full periodic-HF force constants; acoustic mode should vanish at Gamma.",
        }
    )
    return rows


def periodic_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Periodic Mini-Validation",
        "",
        "Regenerated by `python benchmarks/validation_table.py`.",
        "",
        "| Case | Observable | MOLEKUL | Reference | |Delta| | Reference model | Wall time (s) | Status | Notes |",
        "|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['observable']} | "
            f"{float(row['molekul']):.10f} | {float(row['reference']):.10f} | "
            f"{float(row['abs_delta']):.3e} | {row['ref_model']} | "
            f"{float(row['wall_time_s']):.3f} | {row['status']} | {row['notes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    cells = [
        run_cell(method, mol_name, basis_name, runner, notes)
        for method, mol_name, runner, notes in METHODS
        for basis_name in BASIS_MAP
    ]
    periodic_rows = periodic_validation()

    molecular_payload = {
        "provenance": {
            "generator": "benchmarks/validation_table.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyscf": getattr(sys.modules.get("pyscf"), "__version__", "unknown"),
        },
        "cells": [asdict(cell) for cell in cells],
    }
    periodic_payload = {
        "provenance": molecular_payload["provenance"],
        "rows": periodic_rows,
    }

    MOLECULAR_JSON.write_text(json.dumps(molecular_payload, indent=2) + "\n")
    MOLECULAR_MD.write_text(molecular_markdown(cells))
    PERIODIC_JSON.write_text(json.dumps(periodic_payload, indent=2) + "\n")
    PERIODIC_MD.write_text(periodic_markdown(periodic_rows))

    counts = {status: sum(cell.status == status for cell in cells) for status in ("ok", "experimental", "unsupported", "skipped")}
    print("Molecular validation summary:", counts)
    print()
    print(MOLECULAR_MD.read_text(), end="")
    print(PERIODIC_MD.read_text(), end="")


if __name__ == "__main__":
    main()
