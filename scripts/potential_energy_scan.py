"""
scripts/potential_energy_scan.py
=================================
N2 potential energy scan: MOLEKUL vs PySCF at RHF/STO-3G and MP2/STO-3G.

The RHF scan covers 0.90–2.20 Å.  The MP2 scan is restricted to 0.90–1.70 Å
to avoid the well-known near-degeneracy divergence of MP2 at stretched
N≡N geometries.  Agreement with PySCF is quantified for each curve.

Outputs: outputs/figures/pes_n2.{pdf,png,json}

Usage
-----
    python scripts/potential_energy_scan.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "src")

from pyscf import gto, scf as pyscf_scf, mp as pyscf_mp

from molekul.atoms import Atom
from molekul.basis_sto3g import get_sto3g
from molekul.molecule import Molecule
from molekul.rhf import rhf_scf
from molekul.mp2 import mp2_energy

ANG2BOHR = 1.8897259886

R_RHF = np.array([0.90, 1.00, 1.05, 1.10, 1.15, 1.20,
                   1.30, 1.40, 1.55, 1.70, 1.85, 2.00, 2.20])
R_MP2 = np.array([0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.55, 1.70])


def _mk_mol(r_ang):
    half = r_ang * ANG2BOHR / 2.0
    return Molecule([Atom("N", np.array([-half, 0.0, 0.0])),
                     Atom("N", np.array([+half, 0.0, 0.0]))])


def _py_mol(r_ang):
    mol = gto.Mole()
    mol.atom = f"N 0 0 -{r_ang/2:.6f}; N 0 0 {r_ang/2:.6f}"
    mol.basis = "sto-3g"
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    return mol


print("N2 potential energy scan — STO-3G basis")
print(f"  RHF: {len(R_RHF)} points | MP2: {len(R_MP2)} points (R ≤ 1.70 Å)")

mk_rhf, py_rhf, mk_mp2, py_mp2 = [], [], [], []

print("  Computing RHF/STO-3G ...", flush=True)
for r in R_RHF:
    mol = _mk_mol(r)
    basis = get_sto3g()
    mk_rhf.append(rhf_scf(mol, basis, max_iter=300, verbose=False).energy_total)
    mf = pyscf_scf.RHF(_py_mol(r)); mf.max_cycle = 300
    py_rhf.append(mf.kernel())

print("  Computing MP2/STO-3G ...", flush=True)
for r in R_MP2:
    mol = _mk_mol(r)
    basis = get_sto3g()
    rhf_res = rhf_scf(mol, basis, max_iter=300, verbose=False)
    mk_mp2.append(mp2_energy(mol, basis, rhf_res).energy_total)
    mf = pyscf_scf.RHF(_py_mol(r)); mf.max_cycle = 300; mf.kernel()
    pt = pyscf_mp.MP2(mf); pt.kernel()
    py_mp2.append(pt.e_tot)

mk_rhf = np.array(mk_rhf)
py_rhf = np.array(py_rhf)
mk_mp2 = np.array(mk_mp2)
py_mp2 = np.array(py_mp2)

print("Done.")


def r_eq(R, E):
    return R[np.nanargmin(E)]


print("\nSummary:")
for tag, R, mk, py in [("RHF/STO-3G", R_RHF, mk_rhf, py_rhf),
                        ("MP2/STO-3G", R_MP2, mk_mp2, py_mp2)]:
    delta = np.abs(mk - py).max()
    print(f"  {tag}  r_e(MOLEKUL)={r_eq(R,mk):.3f} Å  "
          f"r_e(PySCF)={r_eq(R,py):.3f} Å  max|ΔE|={delta:.2e} Eh")

os.makedirs("outputs/figures", exist_ok=True)
with open("outputs/figures/pes_n2_data.json", "w") as f:
    json.dump({"R_RHF_ang": R_RHF.tolist(), "R_MP2_ang": R_MP2.tolist(),
               "MOLEKUL_RHF": mk_rhf.tolist(), "PySCF_RHF": py_rhf.tolist(),
               "MOLEKUL_MP2": mk_mp2.tolist(), "PySCF_MP2": py_mp2.tolist()}, f)

# ── figure ────────────────────────────────────────────────────────────────
def rel_kcal(E):
    return (E - np.nanmin(E)) * 627.509


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
plt.subplots_adjust(left=0.10, right=0.97, top=0.85, bottom=0.15, wspace=0.35)

C_MK, C_PY = "#1f77b4", "#d62728"

# Both panels share the same x-range (bonding region only, no clipping)
XLIM = (0.87, 1.75)

for ax, R, mk, py, title in [
    (ax1, R_RHF, mk_rhf, py_rhf, "RHF / STO-3G"),
    (ax2, R_MP2, mk_mp2, py_mp2, "MP2 / STO-3G"),
]:
    # Restrict data to XLIM
    mask = (R >= XLIM[0]) & (R <= XLIM[1])
    R_pl, mk_pl, py_pl = R[mask], mk[mask], py[mask]

    mk_r = rel_kcal(mk_pl)
    py_r = rel_kcal(py_pl)

    ax.plot(R_pl, mk_r, "o-",  color=C_MK, lw=2.0, ms=4.5,
            label="MOLEKUL", zorder=3)
    ax.plot(R_pl, py_r, "s--", color=C_PY, lw=1.6, ms=4.5,
            label="PySCF",   zorder=2, dashes=(6, 2.5))

    re_val = r_eq(R_pl, mk_pl)
    ax.axvline(re_val, color="gray", lw=0.8, ls=":", zorder=1)
    ymax = np.nanmax(mk_r)
    ax.annotate(f"grid min: {re_val:.3f} Å",
                xy=(re_val, 0), xytext=(re_val + 0.12, ymax * 0.45),
                fontsize=8, color="dimgray",
                arrowprops=dict(arrowstyle="-|>", color="dimgray",
                                lw=0.8, mutation_scale=9))

    ax.set_xlim(*XLIM)
    ax.set_ylim(-3, ymax * 1.15)
    ax.set_xlabel(r"$r_{\mathrm{N-N}}$ / Å", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.tick_params(labelsize=8.5)

ax1.set_ylabel(r"$\Delta E$ / kcal mol$^{-1}$", fontsize=10)

legend_elements = [
    Line2D([0], [0], color=C_MK, lw=2.0, marker="o", ms=4.5, label="MOLEKUL"),
    Line2D([0], [0], color=C_PY, lw=1.6, marker="s", ms=4.5,
           ls="--", dashes=(6, 2.5), label="PySCF"),
]
ax2.legend(handles=legend_elements, fontsize=9.5, framealpha=0.85,
           loc="upper right")
# No internal suptitle — caption carries all information

for fmt in ("pdf", "png"):
    path = f"outputs/figures/pes_n2.{fmt}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")

plt.close(fig)
