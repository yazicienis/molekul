"""Tests for Phase 20a periodic infrastructure."""

import numpy as np

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.integrals import build_core_hamiltonian, build_overlap
from molekul.molecule import Molecule
from molekul.periodic import Crystal, bloch_hcore, bloch_overlap, monkhorst_pack


def _h2_molecule():
    return Molecule(
        atoms=[Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
        name="H2",
    )


def _h2_crystal():
    mol = _h2_molecule()
    return Crystal(
        lattice=np.diag([20.0, 20.0, 20.0]),
        atoms=mol.atoms,
        charge=0,
        multiplicity=1,
        name="H2 box",
    )


def test_bloch_overlap_gamma_matches_molecular():
    mol = _h2_molecule()
    crystal = _h2_crystal()
    k_gamma = np.zeros((1, 3))
    S_periodic = bloch_overlap(crystal, STO3G, k_gamma)[0]
    S_molecular = build_overlap(STO3G, mol)
    assert np.max(np.abs(S_periodic - S_molecular)) < 1e-10


def test_bloch_hcore_gamma_matches_molecular():
    mol = _h2_molecule()
    crystal = _h2_crystal()
    k_gamma = np.zeros((1, 3))
    H_periodic = bloch_hcore(crystal, STO3G, k_gamma)[0]
    H_molecular = build_core_hamiltonian(STO3G, mol)
    assert np.max(np.abs(H_periodic - H_molecular)) < 1e-8


def test_monkhorst_pack_1d_shape():
    lattice = np.array([[2.0, 0.0, 0.0]])
    kpts = monkhorst_pack(lattice, (4,))
    assert kpts.shape == (4, 3)


def test_monkhorst_pack_3d_shape():
    lattice = np.eye(3) * 5.0
    kpts = monkhorst_pack(lattice, (2, 2, 2))
    assert kpts.shape == (8, 3)


def test_monkhorst_pack_gamma_included():
    lattice = np.eye(3) * 5.0
    kpts = monkhorst_pack(lattice, (1, 1, 1))
    assert kpts.shape == (1, 3)
    assert np.allclose(kpts[0], np.zeros(3))


def test_reciprocal_lattice_orthogonality():
    a = 5.0
    crystal = Crystal(lattice=np.eye(3) * a, atoms=[])
    metric = crystal.lattice @ crystal.reciprocal_lattice.T
    assert np.allclose(metric, 2.0 * np.pi * np.eye(3))


def test_crystal_n_electrons():
    assert _h2_crystal().n_electrons == 2


def test_lattice_vectors_shell_count():
    crystal = Crystal(lattice=np.array([[2.0, 0.0, 0.0]]), atoms=[])
    vectors = crystal.lattice_vectors_in_shell(5.0)
    assert len(vectors) == 5
    assert np.allclose(np.sort(vectors[:, 0]), np.array([-4.0, -2.0, 0.0, 2.0, 4.0]))
