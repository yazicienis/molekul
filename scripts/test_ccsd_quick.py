"""Quick CCSD test vs PySCF for H2 and HeH+ (STO-3G)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.rhf import rhf_scf
from molekul.ccsd import ccsd_energy

# -----------------------------------------------------------------------
# H2  (STO-3G equilibrium geometry)
# -----------------------------------------------------------------------
h2 = Molecule(
    atoms=[Atom("H", [0.0, 0.0, 0.0]),
           Atom("H", [0.0, 0.0, 1.345919])],  # bohr
    charge=0, multiplicity=1
)
bas_h2 = STO3G
rhf_h2 = rhf_scf(h2, bas_h2, verbose=False)
print(f"H2 RHF  E = {rhf_h2.energy_total:.10f} Ha")

res_h2 = ccsd_energy(h2, bas_h2, rhf_h2, verbose=True)
print(f"H2 CCSD E_corr = {res_h2.energy_ccsd:.10f} Ha")
print(f"H2 CCSD E_tot  = {res_h2.energy_total:.10f} Ha")
print(f"H2 CCSD converged={res_h2.converged} in {res_h2.n_iter} iter")

# PySCF reference (computed separately at same geometry, STO-3G)
# pyscf: mf.CCSD().run() gives E_corr = -0.013413... (approx)
# Let me compare with PySCF if available
try:
    from pyscf import gto, scf, cc
    mol_ps = gto.Mole()
    mol_ps.atom = "H 0 0 0; H 0 0 0.71272"  # in Angstrom = 1.345919 bohr
    mol_ps.basis = "sto-3g"
    mol_ps.unit  = "Angstrom"
    mol_ps.build()
    mf = scf.RHF(mol_ps).run(verbose=0)
    mycc = cc.CCSD(mf).run(verbose=0)
    print(f"\nPySCF H2 CCSD E_corr = {mycc.e_corr:.10f} Ha")
    print(f"PySCF H2 CCSD E_tot  = {mf.e_tot + mycc.e_corr:.10f} Ha")
    diff = abs(res_h2.energy_ccsd - mycc.e_corr)
    print(f"Difference: {diff:.3e} Ha  ({'PASS' if diff < 1e-6 else 'FAIL'})")
except ImportError:
    print("(PySCF not available for comparison)")

print()
# -----------------------------------------------------------------------
# HeH+  (STO-3G)
# -----------------------------------------------------------------------
heh = Molecule(
    atoms=[Atom("He", [0.0, 0.0, 0.0]),
           Atom("H",  [0.0, 0.0, 1.4632])],  # bohr, approx equilibrium
    charge=1, multiplicity=1
)
bas_heh = STO3G
rhf_heh = rhf_scf(heh, bas_heh, verbose=False)
print(f"HeH+ RHF  E = {rhf_heh.energy_total:.10f} Ha")

res_heh = ccsd_energy(heh, bas_heh, rhf_heh, verbose=True)
print(f"HeH+ CCSD E_corr = {res_heh.energy_ccsd:.10f} Ha")
print(f"HeH+ CCSD E_tot  = {res_heh.energy_total:.10f} Ha")

try:
    from pyscf import gto, scf, cc
    mol_ps2 = gto.Mole()
    mol_ps2.atom = "He 0 0 0; H 0 0 0.77439"  # 1.4632 bohr → Angstrom
    mol_ps2.basis = "sto-3g"
    mol_ps2.charge = 1
    mol_ps2.unit = "Angstrom"
    mol_ps2.build()
    mf2 = scf.RHF(mol_ps2).run(verbose=0)
    mycc2 = cc.CCSD(mf2).run(verbose=0)
    print(f"\nPySCF HeH+ CCSD E_corr = {mycc2.e_corr:.10f} Ha")
    diff2 = abs(res_heh.energy_ccsd - mycc2.e_corr)
    print(f"Difference: {diff2:.3e} Ha  ({'PASS' if diff2 < 1e-6 else 'FAIL'})")
except ImportError:
    pass
