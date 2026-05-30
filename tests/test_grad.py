"""Tests for Phase 18 RHF nuclear gradients."""

import numpy as np

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.constants import ANGSTROM_TO_BOHR
from molekul.eri import build_eri
from molekul.grad import (
    _displaced,
    eri_derivative,
    hcore_derivative,
    numerical_gradient,
    overlap_derivative,
    rhf_gradient,
)
from molekul.integrals import build_core_hamiltonian, build_overlap
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf


DERIV_H = 1e-4


def h2_molecule():
    return Molecule(
        atoms=[
            Atom("H", [0.0, 0.0, 0.0]),
            Atom("H", [0.0, 0.0, 0.74 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
        name="H2",
    )


def h2o_molecule():
    return Molecule(
        atoms=[
            Atom("O", [0.0, 0.0, 0.0]),
            Atom("H", [0.0, 0.757 * ANGSTROM_TO_BOHR, -0.469 * ANGSTROM_TO_BOHR]),
            Atom("H", [0.0, -0.757 * ANGSTROM_TO_BOHR, -0.469 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
        name="H2O",
    )


def co_molecule():
    return Molecule(
        atoms=[
            Atom("C", [0.0, 0.0, 0.0]),
            Atom("O", [0.0, 0.0, 1.128 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
        name="CO",
    )


def finite_diff_tensor(builder, mol, atom_idx=1, coord_idx=2, h=DERIV_H):
    mol_p = _displaced(mol, atom_idx, coord_idx, +h)
    mol_m = _displaced(mol, atom_idx, coord_idx, -h)
    return (builder(STO3G, mol_p) - builder(STO3G, mol_m)) / (2.0 * h)


def test_overlap_derivative_h2():
    mol = h2_molecule()
    analytic = overlap_derivative(STO3G, mol)[1, 2]
    numerical = finite_diff_tensor(build_overlap, mol)
    assert np.max(np.abs(analytic - numerical)) < 1e-7


def test_hcore_derivative_h2():
    mol = h2_molecule()
    analytic = hcore_derivative(STO3G, mol)[1, 2]
    numerical = finite_diff_tensor(build_core_hamiltonian, mol)
    assert np.max(np.abs(analytic - numerical)) < 1e-7


def test_eri_derivative_h2():
    mol = h2_molecule()
    analytic = eri_derivative(STO3G, mol)[1, 2]
    numerical = finite_diff_tensor(build_eri, mol)
    assert np.max(np.abs(analytic - numerical)) < 1e-7


def assert_gradient_matches_numerical(mol):
    rhf = rhf_scf(mol, STO3G, verbose=False)
    analytic = rhf_gradient(mol, STO3G, rhf)
    numerical = numerical_gradient(mol, STO3G)
    assert analytic.shape == (mol.n_atoms, 3)
    assert np.max(np.abs(analytic - numerical)) < 1e-5


def test_h2_gradient_vs_numerical():
    assert_gradient_matches_numerical(h2_molecule())


def test_h2o_gradient_vs_numerical():
    assert_gradient_matches_numerical(h2o_molecule())


def test_co_gradient_vs_numerical():
    assert_gradient_matches_numerical(co_molecule())


def test_gradient_shape():
    mol = h2o_molecule()
    rhf = rhf_scf(mol, STO3G, verbose=False)
    assert rhf_gradient(mol, STO3G, rhf).shape == (3, 3)


def test_gradient_translational_invariance():
    mol = h2o_molecule()
    rhf = rhf_scf(mol, STO3G, verbose=False)
    grad = rhf_gradient(mol, STO3G, rhf)
    assert np.max(np.abs(grad.sum(axis=0))) < 1e-10


def test_optimizer_accepts_analytic_gradient():
    from molekul.optimizer import optimize_geometry

    mol = Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.7])],
        charge=0,
        multiplicity=1,
        name="H2",
    )
    result = optimize_geometry(
        mol, STO3G, grad_tol=1e-3, max_steps=20, use_analytic=True, verbose=False
    )
    assert result.converged
    assert result.energy_final < result.energy_initial
