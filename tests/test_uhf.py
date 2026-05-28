"""Tests for unrestricted Hartree-Fock (Phase 16)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.constants import ANGSTROM_TO_BOHR
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf
from molekul.uhf import UHFResult, uhf_scf


OH_UHF_REF = -74.3626332768421
H_UHF_REF = -0.46658184955727533


def water_molecule():
    return Molecule(
        atoms=[
            Atom("O", [0, 0, 0]),
            Atom("H", [0, 0.757 * ANGSTROM_TO_BOHR, 0.586 * ANGSTROM_TO_BOHR]),
            Atom("H", [0, -0.757 * ANGSTROM_TO_BOHR, 0.586 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
    )


def oh_molecule():
    return Molecule(
        atoms=[
            Atom("O", [0, 0, 0]),
            Atom("H", [0, 0, 0.96966 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=2,
    )


def test_h2o_uhf_matches_rhf():
    mol = water_molecule()
    rhf = rhf_scf(mol, STO3G, verbose=False)
    uhf = uhf_scf(mol, STO3G, verbose=False)
    assert uhf.converged
    assert abs(uhf.energy_total - rhf.energy_total) < 1e-8


def test_h2o_s2_zero():
    uhf = uhf_scf(water_molecule(), STO3G, verbose=False)
    assert abs(uhf.S2) < 1e-10


def test_oh_converged():
    result = uhf_scf(oh_molecule(), STO3G, verbose=False)
    assert result.converged
    assert result.n_occ_a == 5
    assert result.n_occ_b == 4


def test_oh_energy():
    result = uhf_scf(oh_molecule(), STO3G, verbose=False)
    assert abs(result.energy_total - OH_UHF_REF) < 1e-4


def test_oh_s2_reasonable():
    result = uhf_scf(oh_molecule(), STO3G, verbose=False)
    assert 0.75 < result.S2 < 1.0


def test_h_atom_energy():
    mol = Molecule(atoms=[Atom("H", [0, 0, 0])], charge=0, multiplicity=2)
    result = uhf_scf(mol, STO3G, verbose=False)
    assert result.converged
    assert abs(result.energy_total - H_UHF_REF) < 1e-5
    assert abs(result.S2 - 0.75) < 1e-12


def test_uhf_result_fields():
    result = uhf_scf(oh_molecule(), STO3G, verbose=False)
    assert isinstance(result, UHFResult)
    for attr in (
        "energy_total",
        "energy_hf",
        "Ca",
        "Cb",
        "epsa",
        "epsb",
        "Pa",
        "Pb",
        "S2",
        "converged",
        "n_iter",
        "n_occ_a",
        "n_occ_b",
        "n_basis",
    ):
        assert hasattr(result, attr)
    assert result.Ca.shape == (result.n_basis, result.n_basis)
    assert result.Cb.shape == (result.n_basis, result.n_basis)
    assert result.Pa.shape == (result.n_basis, result.n_basis)
    assert result.Pb.shape == (result.n_basis, result.n_basis)
    assert np.isclose(result.energy_hf, result.energy_total)
