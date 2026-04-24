"""
scripts/validate_phase2.py
==========================
Human-readable validation of Phase 2: one-electron integrals (S, T, V, H_core)
in STO-3G basis using the McMurchie-Davidson recurrence.

Runs independently of the pytest suite.
Exits 0 if all checks pass, 1 if any fail.

Usage
-----
    python scripts/validate_phase2.py
"""

import sys
import traceback
from math import exp, sqrt, pi

import numpy as np

# Allow running from repo root without install
sys.path.insert(0, "src")

from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.basis_sto3g import STO3G
from molekul.basis import norm_primitive
from molekul.integrals import (
    build_overlap, build_kinetic, build_nuclear, build_core_hamiltonian,
    overlap_primitive, _E, _boys,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[32m  PASS\033[0m"
FAIL = "\033[31m  FAIL\033[0m"
WARN = "\033[33m  WARN\033[0m"

_failures = []

def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  [{detail}]" if detail else ""
    print(f"{status}  {label}{suffix}")
    if not condition:
        _failures.append(label)
    return condition


def check_close(label: str, value, ref, atol: float = 1e-5, detail: str = "") -> bool:
    err = abs(value - ref)
    ok = err <= atol
    suffix = f"  computed={value:.8f}  ref={ref:.8f}  |err|={err:.2e}"
    if detail:
        suffix += f"  [{detail}]"
    return check(label, ok, suffix.strip())


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Molecules
# ─────────────────────────────────────────────────────────────────────────────

H2 = Molecule(
    atoms=[Atom("H", [0, 0, 0]), Atom("H", [0, 0, 1.4])],
    charge=0, multiplicity=1, name="H2"
)

HEH = Molecule(
    atoms=[Atom("He", [0, 0, 0]), Atom("H", [0, 0, 1.4632])],
    charge=1, multiplicity=1, name="HeH+"
)

H2O = Molecule(
    atoms=[
        Atom.from_angstrom("O",  0.0,  0.000, 0.000),
        Atom.from_angstrom("H",  0.0,  0.757, -0.586),
        Atom.from_angstrom("H",  0.0, -0.757, -0.586),
    ],
    charge=0, multiplicity=1, name="H2O"
)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Primitive building blocks
# ─────────────────────────────────────────────────────────────────────────────

section("1. Primitive Building Blocks")

print("\n-- Boys function F_n(x) --")
check_close("F_0(0) = 1.0", _boys(0, 0), 1.0, atol=1e-14)
check_close("F_1(0) = 1/3", _boys(1, 0), 1.0 / 3.0, atol=1e-14)
check_close("F_2(0) = 1/5", _boys(2, 0), 1.0 / 5.0, atol=1e-14)
# Large-x limit: F_0(x) → sqrt(π/x)/2
x = 50.0
check_close(f"F_0({x}) ≈ sqrt(π/x)/2", _boys(0, x), sqrt(pi / x) / 2, atol=1e-6)
# Verify monotone decrease in n at fixed x
vals = [_boys(n, 2.0) for n in range(6)]
check("F_n(2) monotone decreasing in n",
      all(vals[i] > vals[i+1] for i in range(5)),
      f"values = {[f'{v:.6f}' for v in vals]}")

print("\n-- E^{{ij}}_t coefficients --")
# E(0,0,0,Qx) = exp(-mu*Qx^2)
for Qx in [0.0, 1.0, 2.5]:
    a, b = 1.2, 0.8
    mu = a * b / (a + b)
    expected = exp(-mu * Qx**2)
    check_close(f"E(0,0,0,Qx={Qx}) = exp(-mu*Qx²)", _E(0, 0, 0, Qx, a, b), expected, atol=1e-14)

# E(1,0,0) = XPA * E(0,0,0)
a, b, Qx = 1.2, 0.8, 1.5
p = a + b
XPA = -b * Qx / p
e00 = _E(0, 0, 0, Qx, a, b)
check_close("E(1,0,0) = X_PA·E(0,0,0)", _E(1, 0, 0, Qx, a, b), XPA * e00, atol=1e-14)

# E(0,1,0) = XPB * E(0,0,0)
XPB = a * Qx / p
check_close("E(0,1,0) = X_PB·E(0,0,0)", _E(0, 1, 0, Qx, a, b), XPB * e00, atol=1e-14)

# Out-of-range must be zero
check("E(i,j,t<0) = 0", _E(1, 1, -1, 0.5, 1.0, 1.0) == 0.0)
check("E(i,j,t>i+j) = 0", _E(1, 1, 3, 0.5, 1.0, 1.0) == 0.0)

# Swap symmetry: E^{i,j}_t(Qx,a,b) = E^{j,i}_t(-Qx,b,a)
for (i, j, t) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 1)]:
    v1 = _E(i, j, t, 0.7, 1.3, 0.9)
    v2 = _E(j, i, t, -0.7, 0.9, 1.3)
    check_close(f"E({i},{j},{t}) center-swap symmetry", v1, v2, atol=1e-13)

print("\n-- Primitive overlap normalisation --")
A = np.zeros(3)
for (lx, ly, lz) in [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0), (1,1,0), (0,1,1)]:
    for alpha in [0.5, 1.0, 5.0]:
        N = norm_primitive(alpha, lx, ly, lz)
        raw = overlap_primitive(lx, ly, lz, A, alpha, lx, ly, lz, A, alpha)
        check_close(f"N²<g|g>=1  l=({lx},{ly},{lz})  α={alpha}",
                    N**2 * raw, 1.0, atol=1e-10)

print("\n-- Primitive overlap parity zeros --")
a = 1.5
check_close("<s|px> same center = 0", overlap_primitive(0,0,0, A, a, 1,0,0, A, a), 0.0, atol=1e-14)
check_close("<px|py> same center = 0", overlap_primitive(1,0,0, A, a, 0,1,0, A, a), 0.0, atol=1e-14)
check_close("<px|pz> same center = 0", overlap_primitive(1,0,0, A, a, 0,0,1, A, a), 0.0, atol=1e-14)
check_close("<s|dxy> same center = 0", overlap_primitive(0,0,0, A, a, 1,1,0, A, a), 0.0, atol=1e-14)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Matrix properties (molecule-independent invariants)
# ─────────────────────────────────────────────────────────────────────────────

section("2. Matrix-Level Mathematical Properties")

for mol in [H2, HEH, H2O]:
    name = mol.name
    n = STO3G.n_basis(mol)
    S = build_overlap(STO3G, mol)
    T = build_kinetic(STO3G, mol)
    V = build_nuclear(STO3G, mol)
    H = T + V
    eig_S = np.linalg.eigvalsh(S)
    eig_T = np.linalg.eigvalsh(T)

    print(f"\n  -- {name} (n_basis={n}) --")

    check(f"{name}: S shape = ({n},{n})", S.shape == (n, n))
    check(f"{name}: T shape = ({n},{n})", T.shape == (n, n))
    check(f"{name}: V shape = ({n},{n})", V.shape == (n, n))

    check(f"{name}: S symmetric", np.allclose(S, S.T, atol=1e-10))
    check(f"{name}: T symmetric", np.allclose(T, T.T, atol=1e-10))
    check(f"{name}: V symmetric", np.allclose(V, V.T, atol=1e-10))
    check(f"{name}: H symmetric", np.allclose(H, H.T, atol=1e-10))

    check(f"{name}: S diagonal ≈ 1",
          np.allclose(np.diag(S), np.ones(n), atol=1e-4),
          f"max|S_ii-1|={max(abs(np.diag(S)-1)):.2e}")

    check(f"{name}: Tr(S) = {n}",
          abs(np.trace(S) - n) < 1e-3,
          f"Tr(S)={np.trace(S):.6f}")

    check(f"{name}: S positive definite",
          eig_S.min() > 0,
          f"min eigval={eig_S.min():.4e}")

    # Cauchy-Schwarz: |S_ij|² ≤ S_ii * S_jj
    diag_S = np.diag(S)
    cs_bound = np.outer(diag_S, diag_S)
    cs_ok = np.all(S**2 <= cs_bound + 1e-7)
    check(f"{name}: S Cauchy-Schwarz |S_ij|²≤S_ii·S_jj", cs_ok,
          f"max violation={(S**2 - cs_bound).max():.2e}")

    check(f"{name}: T diagonal > 0",
          np.all(np.diag(T) > 0),
          f"min T_ii={np.diag(T).min():.4e}")

    check(f"{name}: T positive definite",
          eig_T.min() > 0,
          f"min eigval={eig_T.min():.4e}")

    check(f"{name}: V diagonal < 0",
          np.all(np.diag(V) < 0),
          f"max V_ii={np.diag(V).max():.4e}")

    check(f"{name}: H_core diagonal < 0",
          np.all(np.diag(H) < 0),
          f"max H_ii={np.diag(H).max():.4e}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: H2 regression (Szabo & Ostlund + PySCF)
# ─────────────────────────────────────────────────────────────────────────────

section("3. H2 STO-3G at R=1.4 bohr — Regression (Szabo & Ostlund / PySCF)")

S2 = build_overlap(STO3G, H2)
T2 = build_kinetic(STO3G, H2)
V2 = build_nuclear(STO3G, H2)
H2m = T2 + V2

print(f"\n  Nuclear repulsion: {H2.nuclear_repulsion_energy():.8f}  (exact: {1/1.4:.8f})")
check_close("E_nuc = 1/1.4", H2.nuclear_repulsion_energy(), 1.0 / 1.4, atol=1e-10)

# S&O values
check_close("S12 = 0.6593182 [S&O]", S2[0, 1], 0.6593182, atol=5e-5)
check_close("T11 = 0.7600319 [S&O]", T2[0, 0], 0.7600319, atol=5e-5)
check_close("T12 = 0.2364547 [S&O]", T2[0, 1], 0.2364547, atol=5e-5)
check_close("V11 = -1.8804409 [PySCF]", V2[0, 0], -1.8804409, atol=5e-5)
check_close("V12 = -1.1948346 [PySCF]", V2[0, 1], -1.1948346, atol=5e-5)
check_close("H11 = -1.1204090 [S&O]", H2m[0, 0], -1.1204090, atol=5e-5)
check_close("H12 = -0.9583800 [S&O]", H2m[0, 1], -0.9583800, atol=5e-5)

print(f"\n  Full matrices:")
print(f"  S =\n{S2}")
print(f"  T =\n{T2}")
print(f"  V =\n{V2}")
print(f"  H =\n{H2m}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: HeH+ regression (PySCF reference)
# ─────────────────────────────────────────────────────────────────────────────

section("4. HeH+ STO-3G at R=1.4632 bohr — Regression (PySCF)")

Sh = build_overlap(STO3G, HEH)
Th = build_kinetic(STO3G, HEH)
Vh = build_nuclear(STO3G, HEH)
Hh = Th + Vh

print(f"\n  Nuclear repulsion: {HEH.nuclear_repulsion_energy():.8f}  (exact: {2/1.4632:.8f})")
check_close("E_nuc = 2/1.4632", HEH.nuclear_repulsion_energy(), 2.0 / 1.4632, atol=1e-6)

check_close("S11 ≈ 1.0 (He norm)", Sh[0, 0], 1.0, atol=1e-4)
check_close("S22 ≈ 1.0 (H norm)", Sh[1, 1], 1.0, atol=1e-4)
check_close("S12 = 0.5368194 [PySCF]", Sh[0, 1], 0.5368194, atol=5e-5)
check_close("T11 = 1.4117632 [PySCF]", Th[0, 0], 1.4117632, atol=5e-5)
check_close("T22 = 0.7600319 [PySCF]", Th[1, 1], 0.7600319, atol=5e-5)
check_close("T12 = 0.1974432 [PySCF]", Th[0, 1], 0.1974432, atol=5e-5)
check_close("V11 = -4.0100462 [PySCF]", Vh[0, 0], -4.0100462, atol=5e-5)
check_close("V22 = -2.4918576 [PySCF]", Vh[1, 1], -2.4918576, atol=5e-5)
check_close("V12 = -1.6292717 [PySCF]", Vh[0, 1], -1.6292717, atol=5e-5)
check_close("H11 = -2.5982830 [PySCF]", Hh[0, 0], -2.5982830, atol=5e-5)
check_close("H22 = -1.7318257 [PySCF]", Hh[1, 1], -1.7318257, atol=5e-5)
check_close("H12 = -1.4318285 [PySCF]", Hh[0, 1], -1.4318285, atol=5e-5)

print(f"\n  Full matrices:")
print(f"  S =\n{Sh}")
print(f"  T =\n{Th}")
print(f"  V =\n{Vh}")
print(f"  H =\n{Hh}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: H2O regression (PySCF reference)
# ─────────────────────────────────────────────────────────────────────────────

section("5. H2O STO-3G at C2v geometry — Regression (PySCF)")

print("  Basis ordering: 1s_O, 2s_O, 2px_O, 2py_O, 2pz_O, 1s_H1, 1s_H2")

Sw = build_overlap(STO3G, H2O)
Tw = build_kinetic(STO3G, H2O)
Vw = build_nuclear(STO3G, H2O)
Hw = Tw + Vw

check(f"n_basis = 7", STO3G.n_basis(H2O) == 7)

# Structural zeros (molecular symmetry)
print("\n  -- Symmetry-mandated zeros in S --")
check_close("<px_O | 1s_H1> = 0  [parity in x]", Sw[2, 5], 0.0, atol=1e-7)
check_close("<px_O | 1s_H2> = 0  [parity in x]", Sw[2, 6], 0.0, atol=1e-7)

# py overlaps with H1 (+y) and H2 (-y) have opposite signs
check_close("|<py_O|H1>| = |<py_O|H2>|  [C2 symmetry]",
            abs(Sw[3, 5]), abs(Sw[3, 6]), atol=1e-8)
check("sign(<py_O|H1>) = −sign(<py_O|H2>)",
      Sw[3, 5] * Sw[3, 6] < 0)
check_close("<py_O|H1> = 0.31109 [PySCF]", Sw[3, 5], 0.31109395, atol=1e-4)

# pz overlaps with H1 and H2 are equal
check_close("<pz_O|H1> = <pz_O|H2>  [C2 symmetry]",
            Sw[4, 5], Sw[4, 6], atol=1e-8)
check_close("<pz_O|H1> = -0.24082 [PySCF]", Sw[4, 5], -0.24082042, atol=1e-4)

# Diagonal kinetic values
print("\n  -- Kinetic diagonal [PySCF reference] --")
ref_T_diag = np.array([29.00319995, 0.80812795, 2.52873120, 2.52873120,
                        2.52873120, 0.76003188, 0.76003188])
for i, (got, ref) in enumerate(zip(np.diag(Tw), ref_T_diag)):
    check_close(f"T[{i},{i}]", got, ref, atol=1e-3)

# Diagonal Hcore values
print("\n  -- H_core diagonal [PySCF reference] --")
ref_H_diag = np.array([-32.72080378, -9.33471026, -7.45756728,
                        -7.61384064, -7.55121310, -5.07626089, -5.07626089])
for i, (got, ref) in enumerate(zip(np.diag(Hw), ref_H_diag)):
    check_close(f"H[{i},{i}]", got, ref, atol=1e-2)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Basis set data integrity
# ─────────────────────────────────────────────────────────────────────────────

section("6. Basis Set (STO-3G) Data Integrity")

for el in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]:
    check(f"Element {el} present in STO-3G", el in STO3G.shells_by_element)

check("H: 1 shell, l=0, 3 primitives",
      len(STO3G.shells_by_element["H"]) == 1
      and STO3G.shells_by_element["H"][0].l == 0
      and STO3G.shells_by_element["H"][0].n_primitives == 3)

check("O: 3 shells (1s+2s+2p)",
      len(STO3G.shells_by_element["O"]) == 3
      and STO3G.shells_by_element["O"][0].l == 0
      and STO3G.shells_by_element["O"][1].l == 0
      and STO3G.shells_by_element["O"][2].l == 1)

check("H2 → 2 basis functions", STO3G.n_basis(H2) == 2)
check("HeH+ → 2 basis functions", STO3G.n_basis(HEH) == 2)
check("H2O → 7 basis functions", STO3G.n_basis(H2O) == 7)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

section("SUMMARY")

total_checks = sum(
    1 for line in open(__file__)
    if "check(" in line or "check_close(" in line
)

_all_ok = len(_failures) == 0

if _failures:
    print(f"\n\033[31mFAILED {len(_failures)} checks:\033[0m")
    for f in _failures:
        print(f"  ✗  {f}")
    print()
else:
    print(f"\n\033[32mAll checks passed.\033[0m")
    print()

# --- Dual-format log to outputs/logs/ ---------------------------------------
try:
    sys.path.insert(0, "src")
    from molekul.logging_utils import ExperimentLogger
    import numpy as _np

    _exp = ExperimentLogger("phase2", "integrals")

    # Key regression values (H2 at R=1.4 bohr, STO-3G)
    _S2 = build_overlap(STO3G, H2)
    _T2 = build_kinetic(STO3G, H2)
    _V2 = build_nuclear(STO3G, H2)
    _H2m = _T2 + _V2

    _exp.metric("h2_S12",  float(_S2[0, 1]))
    _exp.metric("h2_T11",  float(_T2[0, 0]))
    _exp.metric("h2_T12",  float(_T2[0, 1]))
    _exp.metric("h2_H11",  float(_H2m[0, 0]))
    _exp.metric("h2_H12",  float(_H2m[0, 1]))
    _exp.metric("h2_S12_ref_so",    0.6593182)
    _exp.metric("h2_S12_err",       abs(float(_S2[0, 1]) - 0.6593182))
    _exp.metric("h2_T11_ref_so",    0.7600319)
    _exp.metric("h2_T11_err",       abs(float(_T2[0, 0]) - 0.7600319))
    _exp.metric("n_checks_run",     total_checks)
    _exp.metric("n_checks_failed",  len(_failures))
    _exp.metric("h2o_n_basis",      STO3G.n_basis(H2O))

    for label, ok in [(lbl, lbl not in _failures) for lbl in
                      ["S12 = 0.6593182 [S&O]", "T11 = 0.7600319 [S&O]",
                       "T12 = 0.2364547 [S&O]", "H11 = -1.1204090 [S&O]",
                       "H12 = -0.9583800 [S&O]"]]:
        _exp.check(label, ok)

    _exp.line(f"Checks run: {total_checks}  passed: {total_checks - len(_failures)}  failed: {len(_failures)}")
    _exp.line("Molecules: H2 (R=1.4 bohr), HeH+ (R=1.4632 bohr), H2O (C2v)  — STO-3G")
    _exp.line("References: Szabo & Ostlund / PySCF 2.x")
    _exp.artifact("outputs/phase2_integrals.txt")

    _txt_path, _json_path = _exp.save()
    print(f"Structured log : {_txt_path}")
    print(f"               : {_json_path}")
except Exception as _e:
    print(f"[logging skipped: {_e}]")

sys.exit(0 if _all_ok else 1)
