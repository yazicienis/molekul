"""
Gaussian basis set data structures.

A BasisSet maps each element symbol to a list of contracted Gaussian shells.
Each Shell carries:
  - angular momentum l  (0=s, 1=p, 2=d)
  - primitive exponents alpha_i
  - raw contraction coefficients c_i  (for NORMALIZED primitives)

Normalization constants for primitives are computed on demand.
Coordinates are NOT stored in shells; they come from the Molecule.
"""

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helper: double factorial  n!! = n*(n-2)*...  with (-1)!! = 1
# ---------------------------------------------------------------------------
def _dfact(n: int) -> int:
    if n <= 0:
        return 1
    r = 1
    while n > 0:
        r *= n
        n -= 2
    return r


def norm_primitive(alpha: float, lx: int, ly: int, lz: int) -> float:
    """
    Normalization constant N for one primitive Cartesian Gaussian:
        g(r) = N (x-Ax)^lx (y-Ay)^ly (z-Az)^lz exp(-alpha |r-A|^2)
    such that <g|g> = 1.

    Formula (Szabo & Ostlund, appendix A):
        N = (2*alpha/pi)^(3/4) * (4*alpha)^(L/2) / sqrt( (2lx-1)!! (2ly-1)!! (2lz-1)!! )
    where L = lx + ly + lz.
    """
    L = lx + ly + lz
    prefac = (2.0 * alpha / pi) ** 0.75
    num = (4.0 * alpha) ** (L / 2.0)
    denom = sqrt(_dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1))
    return prefac * num / denom


# Angular momentum component lists
_ANG_COMPONENTS: Dict[int, List[Tuple[int, int, int]]] = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2),
        (1, 1, 0), (1, 0, 1), (0, 1, 1)],
}
_L_LABEL = {0: "S", 1: "P", 2: "D"}


@dataclass
class Shell:
    """
    A contracted Gaussian shell.

    Attributes
    ----------
    l           : angular momentum (0=s, 1=p, 2=d)
    exponents   : primitive exponents, shape (n_prim,)
    coefficients: raw contraction coefficients (for normalized primitives)
    """
    l: int
    exponents: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        self.exponents = np.asarray(self.exponents, dtype=float)
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if self.exponents.shape != self.coefficients.shape:
            raise ValueError("exponents and coefficients must have the same shape")
        if self.l not in _ANG_COMPONENTS:
            raise NotImplementedError(f"Angular momentum l={self.l} not yet supported")

    @property
    def n_primitives(self) -> int:
        return len(self.exponents)

    @property
    def angular_components(self) -> List[Tuple[int, int, int]]:
        """All (lx, ly, lz) tuples for this shell's angular momentum."""
        return _ANG_COMPONENTS[self.l]

    @property
    def n_functions(self) -> int:
        return len(self.angular_components)

    def norms(self, lx: int, ly: int, lz: int) -> np.ndarray:
        """Per-primitive normalization constants for angular component (lx,ly,lz)."""
        return np.array([norm_primitive(a, lx, ly, lz) for a in self.exponents])

    def __repr__(self) -> str:
        lbl = _L_LABEL.get(self.l, str(self.l))
        return (f"Shell({lbl}, n_prim={self.n_primitives}, "
                f"alpha=[{self.exponents[0]:.4f}..{self.exponents[-1]:.4f}])")


@dataclass
class BasisSet:
    """
    Maps element symbols to lists of Shell objects.

    Usage
    -----
    shells = basis.assign_to_molecule(mol)  # [(atom_idx, Shell), ...]
    bfs   = basis.basis_functions(mol)      # [(atom_idx, lx, ly, lz, Shell), ...]
    """
    name: str
    shells_by_element: Dict[str, List[Shell]] = field(default_factory=dict)

    def assign_to_molecule(self, molecule) -> List[Tuple[int, "Shell"]]:
        """Flat list of (atom_index, Shell) for every shell in the molecule."""
        result = []
        for i, atom in enumerate(molecule.atoms):
            sym = atom.symbol
            if sym not in self.shells_by_element:
                raise ValueError(f"Element '{sym}' not in basis '{self.name}'")
            for shell in self.shells_by_element[sym]:
                result.append((i, shell))
        return result

    def basis_functions(self, molecule) -> List[Tuple[int, int, int, int, "Shell"]]:
        """
        Flat list of (atom_index, lx, ly, lz, Shell) for every basis function.
        This is the primary indexing scheme used by integral routines.
        """
        result = []
        for i, atom in enumerate(molecule.atoms):
            sym = atom.symbol
            if sym not in self.shells_by_element:
                raise ValueError(f"Element '{sym}' not in basis '{self.name}'")
            for shell in self.shells_by_element[sym]:
                for (lx, ly, lz) in shell.angular_components:
                    result.append((i, lx, ly, lz, shell))
        return result

    def n_basis(self, molecule) -> int:
        """Total number of basis functions for the molecule."""
        return len(self.basis_functions(molecule))

    def __repr__(self) -> str:
        els = list(self.shells_by_element.keys())
        return f"BasisSet('{self.name}', elements={els})"
