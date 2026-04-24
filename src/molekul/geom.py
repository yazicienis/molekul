"""
Internal coordinate utilities.

Functions for computing bond lengths, bond angles, and dihedral angles
from Molecule objects.  All inputs and outputs in atomic units (bohr)
unless otherwise noted.

These are thin wrappers around numpy; no caching or special handling.
"""

from __future__ import annotations

import numpy as np

from .molecule import Molecule


def bond_length(mol: Molecule, i: int, j: int) -> float:
    """
    Return the distance between atoms i and j in bohr.

    Parameters
    ----------
    mol : Molecule
    i, j : atom indices (0-based)

    Returns
    -------
    float : |R_i − R_j| in bohr
    """
    return float(np.linalg.norm(mol.atoms[i].coords - mol.atoms[j].coords))


def bond_angle(mol: Molecule, i: int, j: int, k: int) -> float:
    """
    Return the bond angle i–j–k in degrees.

    The angle is measured at the central atom j.

    Parameters
    ----------
    mol  : Molecule
    i, j, k : atom indices (0-based); j is the vertex atom

    Returns
    -------
    float : angle in degrees, range [0, 180]
    """
    rji = mol.atoms[i].coords - mol.atoms[j].coords
    rjk = mol.atoms[k].coords - mol.atoms[j].coords
    norm_ji = np.linalg.norm(rji)
    norm_jk = np.linalg.norm(rjk)
    if norm_ji < 1e-12 or norm_jk < 1e-12:
        raise ValueError(f"Degenerate bond at atom {j}: zero bond length.")
    cos_theta = np.dot(rji, rjk) / (norm_ji * norm_jk)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def dihedral_angle(mol: Molecule, i: int, j: int, k: int, l: int) -> float:
    """
    Return the dihedral angle i–j–k–l in degrees.

    Uses the standard IUPAC convention: the angle between the i–j–k plane
    and the j–k–l plane.

    Parameters
    ----------
    mol     : Molecule
    i,j,k,l : atom indices (0-based)

    Returns
    -------
    float : dihedral in degrees, range (−180, 180]
    """
    b1 = mol.atoms[j].coords - mol.atoms[i].coords
    b2 = mol.atoms[k].coords - mol.atoms[j].coords
    b3 = mol.atoms[l].coords - mol.atoms[k].coords

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-12 or norm_n2 < 1e-12:
        raise ValueError("Degenerate dihedral: collinear atoms.")
    n1 /= norm_n1
    n2 /= norm_n2
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.degrees(np.arctan2(y, x)))
