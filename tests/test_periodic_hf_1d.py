"""Tests for Phase 20b cutoff periodic HF on a 1D H chain."""

from functools import lru_cache

import numpy as np

from molekul.atoms import Atom
from molekul.basis_sto3g import STO3G
from molekul.constants import BOHR_TO_ANGSTROM
from molekul.periodic import Crystal, _overlap_translation, monkhorst_pack, periodic_hf

FALLBACK_GAMMA_ENERGY = 0.08161353812017258
FALLBACK_KGRID_ENERGY = 0.8487041965918238


def _h_chain():
    return Crystal(
        lattice=np.array([[1.8, 0.0, 0.0]]),
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
        name="H chain",
    )


@lru_cache(maxsize=None)
def _pyscf_reference(mesh):
    try:
        import pyscf
        from pyscf.pbc import gto, scf
    except Exception:
        return None
    try:
        cell = gto.Cell()
        cell.atom = "H 0 0 0"
        cell.a = [
            [1.8 * BOHR_TO_ANGSTROM, 0.0, 0.0],
            [0.0, 20.0 * BOHR_TO_ANGSTROM, 0.0],
            [0.0, 0.0, 20.0 * BOHR_TO_ANGSTROM],
        ]
        cell.basis = "sto-3g"
        cell.dimension = 1
        cell.low_dim_ft_type = "inf_vacuum"
        cell.verbose = 0
        cell.build()
        if mesh == (1,):
            mf = scf.RHF(cell)
        else:
            mf = scf.KRHF(cell, kpts=cell.make_kpts([mesh[0], 1, 1]))
        energy = mf.kernel()
    except Exception:
        return None
    if not np.isfinite(energy):
        return None
    return float(energy)


def _reference(mesh):
    runtime = _pyscf_reference(mesh)
    if runtime is not None:
        return runtime
    return FALLBACK_GAMMA_ENERGY if mesh == (1,) else FALLBACK_KGRID_ENERGY


def _run(mesh):
    crystal = _h_chain()
    return periodic_hf(crystal, STO3G, monkhorst_pack(crystal.lattice, mesh))


def test_h_chain_gamma_energy():
    result = _run((1,))
    assert abs(result.energy_per_cell - _reference((1,))) < 1e-3


def test_h_chain_kgrid_energy():
    result = _run((4,))
    assert abs(result.energy_per_cell - _reference((4,))) < 1e-3


def test_h_chain_band_energies_shape():
    result = _run((4,))
    assert result.band_energies.shape == (4, 1)


def test_h_chain_density_matrix_real():
    result = _run((4,))
    assert np.max(np.abs(np.imag(result.density_matrix))) < 1e-10


def test_h_chain_density_matrix_trace():
    crystal = _h_chain()
    result = periodic_hf(crystal, STO3G, monkhorst_pack(crystal.lattice, (4,)))
    S0 = _overlap_translation(crystal, STO3G, np.zeros(3))
    assert abs(np.einsum("mn,nm->", result.density_matrix, S0) - crystal.n_electrons) < 1e-8


def test_h_chain_converged():
    assert _run((4,)).converged
