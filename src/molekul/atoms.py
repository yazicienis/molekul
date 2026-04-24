"""
Atom data structure.

An Atom holds:
- symbol: element symbol (e.g. "H", "O")
- Z: atomic number
- coords: (x, y, z) in Bohr (internal units)

All coordinates stored internally in Bohr.
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

from .constants import SYMBOL_TO_Z, Z_TO_SYMBOL, ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM


@dataclass
class Atom:
    symbol: str
    coords: np.ndarray  # shape (3,), in Bohr

    def __post_init__(self):
        if self.symbol not in SYMBOL_TO_Z:
            raise ValueError(f"Unknown element symbol: '{self.symbol}'")
        self.coords = np.asarray(self.coords, dtype=float)
        if self.coords.shape != (3,):
            raise ValueError(f"coords must be shape (3,), got {self.coords.shape}")

    @property
    def Z(self) -> int:
        return SYMBOL_TO_Z[self.symbol]

    @classmethod
    def from_angstrom(cls, symbol: str, x: float, y: float, z: float) -> "Atom":
        """Create an Atom from Angstrom coordinates."""
        coords = np.array([x, y, z]) * ANGSTROM_TO_BOHR
        return cls(symbol=symbol, coords=coords)

    def coords_angstrom(self) -> np.ndarray:
        """Return coordinates in Angstrom."""
        return self.coords * BOHR_TO_ANGSTROM

    def __repr__(self) -> str:
        c = self.coords_angstrom()
        return f"Atom({self.symbol}, Z={self.Z}, xyz=[{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}] Å)"
