"""
Physical constants and conversion factors.
All values in atomic units (Hartree, Bohr) unless noted.
"""

# Conversion factors
ANGSTROM_TO_BOHR = 1.8897259886  # 1 Angstrom = 1.8897... Bohr
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
HARTREE_TO_EV = 27.211396132
HARTREE_TO_KCAL_MOL = 627.5094740631

# Periodic table: symbol -> atomic number
SYMBOL_TO_Z = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
}

# Atomic number -> symbol
Z_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_Z.items()}

# Standard atomic masses (u)
ATOMIC_MASS = {
    1: 1.00794, 2: 4.00260,
    3: 6.941, 4: 9.01218, 5: 10.811, 6: 12.011,
    7: 14.007, 8: 15.999, 9: 18.9984, 10: 20.180,
}
