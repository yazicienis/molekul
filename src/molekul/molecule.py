"""
Molecule data structure.

Holds a list of Atom objects plus charge and spin multiplicity.
Provides convenience properties for nuclear repulsion energy, etc.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

from .atoms import Atom
from .constants import BOHR_TO_ANGSTROM


@dataclass
class Molecule:
    atoms: List[Atom] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1  # 2S+1; 1 = singlet (closed shell)
    name: str = ""

    def __post_init__(self):
        n_electrons = self.n_electrons
        unpaired = self.multiplicity - 1
        if (n_electrons - unpaired) % 2 != 0:
            raise ValueError(
                f"Charge {self.charge} and multiplicity {self.multiplicity} "
                f"are inconsistent with {n_electrons} electrons."
            )

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_electrons(self) -> int:
        return sum(a.Z for a in self.atoms) - self.charge

    @property
    def n_alpha(self) -> int:
        """Number of alpha electrons."""
        unpaired = self.multiplicity - 1
        return (self.n_electrons + unpaired) // 2

    @property
    def n_beta(self) -> int:
        """Number of beta electrons."""
        return self.n_electrons - self.n_alpha

    @property
    def coords_bohr(self) -> np.ndarray:
        """All atomic coordinates, shape (n_atoms, 3), in Bohr."""
        return np.stack([a.coords for a in self.atoms])

    @property
    def coords_angstrom(self) -> np.ndarray:
        """All atomic coordinates, shape (n_atoms, 3), in Angstrom."""
        return self.coords_bohr * BOHR_TO_ANGSTROM

    @property
    def atomic_numbers(self) -> List[int]:
        return [a.Z for a in self.atoms]

    def nuclear_repulsion_energy(self) -> float:
        """
        Compute nuclear repulsion energy in Hartree.
        E_nuc = sum_{A<B} Z_A * Z_B / |R_A - R_B|
        Coordinates in Bohr, result in Hartree.
        """
        e = 0.0
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                Za = self.atoms[i].Z
                Zb = self.atoms[j].Z
                r = np.linalg.norm(self.atoms[i].coords - self.atoms[j].coords)
                e += Za * Zb / r
        return e

    def __repr__(self) -> str:
        lines = [f"Molecule(name='{self.name}', charge={self.charge}, mult={self.multiplicity})"]
        for a in self.atoms:
            lines.append(f"  {a}")
        lines.append(f"  n_electrons={self.n_electrons}, n_alpha={self.n_alpha}, n_beta={self.n_beta}")
        return "\n".join(lines)
