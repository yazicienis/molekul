"""Tests for CCSD(T) perturbative triples (Phase 14)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.ccsd import CCSDTResult, ccsdt_energy
from molekul.constants import ANGSTROM_TO_BOHR
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf


def _mol(atoms_angstrom):
    return Molecule(
        atoms=[
            Atom(symbol, [coord * ANGSTROM_TO_BOHR for coord in coords])
            for symbol, coords in atoms_angstrom
        ],
        charge=0,
        multiplicity=1,
    )


@pytest.fixture(scope="module")
def h2_ccsdt():
    mol = _mol([
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.74)),
    ])
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return ccsdt_energy(mol, STO3G, rhf, verbose=False)


@pytest.fixture(scope="module")
def h2o_ccsdt():
    mol = _mol([
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.757, -0.469)),
        ("H", (0.0, -0.757, -0.469)),
    ])
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return ccsdt_energy(mol, STO3G, rhf, verbose=False)


@pytest.fixture(scope="module")
def hf_ccsdt():
    mol = _mol([
        ("H", (0.0, 0.0, 0.0)),
        ("F", (0.0, 0.0, 0.917)),
    ])
    rhf = rhf_scf(mol, STO3G, verbose=False)
    return ccsdt_energy(mol, STO3G, rhf, verbose=False)


def test_h2_triples_zero(h2_ccsdt):
    assert abs(h2_ccsdt.energy_ccsdt) < 1e-10


def test_h2o_ccsdt_ecorr(h2o_ccsdt):
    # PySCF RHF/CCSD(T), same geometry and STO-3G basis: E_corr = -0.040032219581 Ha
    e_corr = h2o_ccsdt.energy_ccsd + h2o_ccsdt.energy_ccsdt
    assert abs(e_corr - (-0.040032219581)) < 1e-5


def test_hf_ccsdt_ecorr(hf_ccsdt):
    # PySCF RHF/CCSD(T), same geometry and STO-3G basis: E_corr = -0.025844164536 Ha
    e_corr = hf_ccsdt.energy_ccsd + hf_ccsdt.energy_ccsdt
    assert abs(e_corr - (-0.025844164536)) < 1e-5


def test_ccsdt_result_fields(h2o_ccsdt):
    for attr in (
        "energy_ccsdt", "energy_total", "energy_hf", "energy_ccsd",
        "energy_mp2", "t1", "t2", "converged", "n_iter", "n_occ",
        "n_virt", "n_basis",
    ):
        assert hasattr(h2o_ccsdt, attr)
    assert isinstance(h2o_ccsdt, CCSDTResult)


def test_ccsdt_lower_than_ccsd(h2o_ccsdt, hf_ccsdt):
    assert h2o_ccsdt.energy_total <= h2o_ccsdt.energy_hf + h2o_ccsdt.energy_ccsd + 1e-12
    assert hf_ccsdt.energy_total <= hf_ccsdt.energy_hf + hf_ccsdt.energy_ccsd + 1e-12
