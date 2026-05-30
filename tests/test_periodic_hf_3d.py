"""Tests for Phase 20c 3D periodic HF on rock-salt LiH."""

from functools import lru_cache

import numpy as np

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.constants import BOHR_TO_ANGSTROM
from molekul.periodic import Crystal, ewald_energy, monkhorst_pack, periodic_hf

FALLBACK_GAMMA_ENERGY = -17.613976509694957
FALLBACK_KGRID_ENERGY = -18.521159535600447


def _lih_crystal():
    a = 7.608
    return Crystal(
        lattice=np.eye(3) * a,
        atoms=[Atom("Li", [0.0, 0.0, 0.0]), Atom("H", [a / 2.0, a / 2.0, a / 2.0])],
        charge=0,
        multiplicity=1,
        name="LiH rock-salt",
    )


@lru_cache(maxsize=None)
def _pyscf_reference(mesh):
    try:
        from pyscf.pbc import gto, scf
    except Exception:
        return None
    crystal = _lih_crystal()
    try:
        cell = gto.Cell()
        cell.atom = "\n".join(
            f"{atom.symbol} {atom.coords[0] * BOHR_TO_ANGSTROM:.12f} "
            f"{atom.coords[1] * BOHR_TO_ANGSTROM:.12f} {atom.coords[2] * BOHR_TO_ANGSTROM:.12f}"
            for atom in crystal.atoms
        )
        cell.a = (crystal.lattice * BOHR_TO_ANGSTROM).tolist()
        cell.unit = "Angstrom"
        cell.basis = "sto-3g"
        cell.verbose = 0
        cell.precision = 1e-4
        cell.mesh = [9, 9, 9]
        cell.build()
        kpts = cell.make_kpts(mesh)
        mf = scf.RHF(cell) if len(kpts) == 1 else scf.KRHF(cell, kpts=kpts)
        mf.conv_tol = 1e-8
        mf.max_cycle = 100
        energy = float(mf.kernel())
    except Exception:
        return None
    if not np.isfinite(energy):
        return None
    return energy


def _reference(mesh):
    runtime = _pyscf_reference(mesh)
    if runtime is not None:
        return runtime
    return FALLBACK_GAMMA_ENERGY if mesh == (1, 1, 1) else FALLBACK_KGRID_ENERGY


def _run(mesh):
    crystal = _lih_crystal()
    return periodic_hf(crystal, STO3G, monkhorst_pack(crystal.lattice, mesh))


def test_lih_gamma_energy():
    result = _run((1, 1, 1))
    assert abs(result.energy_per_cell - _reference((1, 1, 1))) < 1e-2


def test_lih_kgrid_energy():
    result = _run((2, 2, 2))
    assert abs(result.energy_per_cell - _reference((2, 2, 2))) < 1e-2


def test_ewald_nn_positive():
    assert ewald_energy(_lih_crystal()) > 0.0


def test_monkhorst_pack_3d_phase20c():
    kpts = monkhorst_pack(_lih_crystal().lattice, (2, 2, 2))
    assert kpts.shape == (8, 3)


def test_lih_converged():
    assert _run((2, 2, 2)).converged


def test_lih_density_real():
    result = _run((2, 2, 2))
    assert np.max(np.abs(np.imag(result.density_matrix))) < 1e-10
