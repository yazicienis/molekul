"""
6-31G* (Pople split-valence polarized) basis set.

Data source: Hehre, Ditchfield, Pople, J. Chem. Phys. 56, 2257 (1972);
             Hariharan & Pople, Theor. Chim. Acta 28, 213 (1973).
Exponents and coefficients taken directly from PySCF's built-in BSE data
to ensure exact agreement with PySCF reference calculations.

Supported elements: H, He, C, N, O, F

Contraction schemes  (primitives → contracted)
----------------------------------------------
H   : (4s)       → [2s]        =  2 basis functions  (no d on H)
He  : (4s)       → [2s]        =  2 basis functions
C   : (10s4p1d)  → [3s2p1d]   = 15 basis functions  (6 Cartesian d)
N   : (10s4p1d)  → [3s2p1d]   = 15 basis functions
O   : (10s4p1d)  → [3s2p1d]   = 15 basis functions
F   : (10s4p1d)  → [3s2p1d]   = 15 basis functions

Notes
-----
• 6-31G* adds d-polarization shells only to non-hydrogen atoms.
• Pople SP shells (shared exponents for s and p) are expanded into
  separate S and P Shell objects — algebraically identical.
• Use cart=True in PySCF for exact n_basis and energy agreement.
"""

from .basis import BasisSet, Shell


def get_631gstar() -> BasisSet:
    bs = BasisSet(name="6-31G*")

    # ------------------------------------------------------------------
    # Hydrogen   [2s]  (no polarization on H in 6-31G*)
    # ------------------------------------------------------------------
    bs.shells_by_element["H"] = [
        Shell(l=0,
              exponents=[18.731137, 2.8253937, 0.6401217],
              coefficients=[0.0334946, 0.23472695, 0.81375733]),
        Shell(l=0,
              exponents=[0.1612778],
              coefficients=[1.0]),
    ]

    # ------------------------------------------------------------------
    # Helium   [2s]
    # ------------------------------------------------------------------
    bs.shells_by_element["He"] = [
        Shell(l=0,
              exponents=[38.421634, 5.77803, 1.241774],
              coefficients=[0.023766, 0.154679, 0.46963]),
        Shell(l=0,
              exponents=[0.297964],
              coefficients=[1.0]),
    ]

    # ------------------------------------------------------------------
    # Carbon   [3s 2p 1d]  (SP shells expanded)
    # ------------------------------------------------------------------
    _C_sp_exps = [7.8682724, 1.8812885, 0.5442493]
    bs.shells_by_element["C"] = [
        Shell(l=0,
              exponents=[3047.5249, 457.36951, 103.94869,
                         29.210155, 9.286663, 3.163927],
              coefficients=[0.0018347, 0.0140373, 0.0688426,
                            0.2321844, 0.4679413, 0.362312]),
        Shell(l=0, exponents=_C_sp_exps,
              coefficients=[-0.1193324, -0.1608542, 1.1434564]),
        Shell(l=1, exponents=_C_sp_exps,
              coefficients=[0.0689991, 0.316424, 0.7443083]),
        Shell(l=0, exponents=[0.1687144], coefficients=[1.0]),
        Shell(l=1, exponents=[0.1687144], coefficients=[1.0]),
        Shell(l=2, exponents=[0.8],       coefficients=[1.0]),
    ]

    # ------------------------------------------------------------------
    # Nitrogen   [3s 2p 1d]
    # ------------------------------------------------------------------
    _N_sp_exps = [11.626358, 2.71628, 0.772218]
    bs.shells_by_element["N"] = [
        Shell(l=0,
              exponents=[4173.511, 627.4579, 142.9021,
                         40.23433, 12.82021, 4.390437],
              coefficients=[0.0018348, 0.013995, 0.068587,
                            0.232241,  0.46907,  0.360455]),
        Shell(l=0, exponents=_N_sp_exps,
              coefficients=[-0.114961, -0.169118, 1.145852]),
        Shell(l=1, exponents=_N_sp_exps,
              coefficients=[0.06758, 0.323907, 0.740895]),
        Shell(l=0, exponents=[0.2120313], coefficients=[1.0]),
        Shell(l=1, exponents=[0.2120313], coefficients=[1.0]),
        Shell(l=2, exponents=[0.8],       coefficients=[1.0]),
    ]

    # ------------------------------------------------------------------
    # Oxygen   [3s 2p 1d]
    # ------------------------------------------------------------------
    _O_sp_exps = [15.539616, 3.5999336, 1.0137618]
    bs.shells_by_element["O"] = [
        Shell(l=0,
              exponents=[5484.6717, 825.23495, 188.04696,
                         52.9645, 16.89757, 5.7996353],
              coefficients=[0.0018311, 0.0139501, 0.0684451,
                            0.2327143, 0.470193,  0.3585209]),
        Shell(l=0, exponents=_O_sp_exps,
              coefficients=[-0.1107775, -0.1480263, 1.130767]),
        Shell(l=1, exponents=_O_sp_exps,
              coefficients=[0.0708743, 0.3397528, 0.7271586]),
        Shell(l=0, exponents=[0.2700058], coefficients=[1.0]),
        Shell(l=1, exponents=[0.2700058], coefficients=[1.0]),
        Shell(l=2, exponents=[0.8],       coefficients=[1.0]),
    ]

    # ------------------------------------------------------------------
    # Fluorine   [3s 2p 1d]
    # ------------------------------------------------------------------
    _F_sp_exps = [20.8479528, 4.80830834, 1.34406986]
    bs.shells_by_element["F"] = [
        Shell(l=0,
              exponents=[7001.71309, 1051.36609, 239.28569,
                         67.3974453, 21.5199573, 7.4031013],
              coefficients=[0.0018196169, 0.0139160796, 0.0684053245,
                            0.23318576,   0.471267439,  0.356618546]),
        Shell(l=0, exponents=_F_sp_exps,
              coefficients=[-0.108506975, -0.146451658, 1.12868858]),
        Shell(l=1, exponents=_F_sp_exps,
              coefficients=[0.0716287243, 0.345912103, 0.722469957]),
        Shell(l=0, exponents=[0.358151393], coefficients=[1.0]),
        Shell(l=1, exponents=[0.358151393], coefficients=[1.0]),
        Shell(l=2, exponents=[0.8],         coefficients=[1.0]),
    ]

    return bs


#: Module-level singleton
G631Star = get_631gstar()
