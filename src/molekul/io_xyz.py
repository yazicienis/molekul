"""
XYZ file reader and writer.

XYZ format:
  Line 1: number of atoms (int)
  Line 2: comment (used for name/metadata)
  Lines 3+: symbol x y z   (coordinates in Angstrom)

We always read coordinates as Angstrom and convert to Bohr internally.
We always write Angstrom for interoperability with viewers (Avogadro, VESTA, VMD).
"""

from pathlib import Path
from typing import Union
import numpy as np

from .atoms import Atom
from .molecule import Molecule
from .constants import BOHR_TO_ANGSTROM


def read_xyz(path: Union[str, Path], charge: int = 0, multiplicity: int = 1) -> Molecule:
    """
    Parse an XYZ file and return a Molecule.

    Args:
        path: path to .xyz file
        charge: molecular charge (not stored in XYZ format, must be provided)
        multiplicity: spin multiplicity (not stored in XYZ format, must be provided)

    Returns:
        Molecule with atoms in Bohr coordinates
    """
    path = Path(path)
    lines = path.read_text().strip().splitlines()

    if len(lines) < 2:
        raise ValueError(f"XYZ file too short: {path}")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"XYZ line 1 must be integer atom count, got: '{lines[0]}'")

    comment = lines[1].strip()
    name = path.stem if not comment else comment

    atom_lines = lines[2:]
    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atoms but found only {len(atom_lines)} coordinate lines."
        )

    atoms = []
    for i, line in enumerate(atom_lines[:n_atoms]):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Atom line {i+3} malformed: '{line}'")
        symbol = parts[0].capitalize()
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"Non-numeric coordinates on line {i+3}: '{line}'")
        atoms.append(Atom.from_angstrom(symbol, x, y, z))

    return Molecule(atoms=atoms, charge=charge, multiplicity=multiplicity, name=name)


def write_xyz(mol: Molecule, path: Union[str, Path], comment: str = "") -> None:
    """
    Write a Molecule to an XYZ file (coordinates in Angstrom).

    Args:
        mol: Molecule to write
        path: output file path
        comment: optional comment line (defaults to mol.name)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comment_line = comment if comment else mol.name
    lines = [str(mol.n_atoms), comment_line]

    for atom in mol.atoms:
        c = atom.coords_angstrom()
        lines.append(f"{atom.symbol:<4s}  {c[0]:14.8f}  {c[1]:14.8f}  {c[2]:14.8f}")

    path.write_text("\n".join(lines) + "\n")


def read_xyz_trajectory(path: Union[str, Path], charge: int = 0, multiplicity: int = 1):
    """
    Read a multi-frame XYZ file (concatenated frames).
    Returns a list of Molecule objects.
    """
    path = Path(path)
    text = path.read_text().strip()
    lines = text.splitlines()

    molecules = []
    i = 0
    frame = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n_atoms = int(lines[i].strip())
        except ValueError:
            raise ValueError(f"Expected atom count at line {i+1}, got: '{lines[i]}'")
        block = lines[i: i + 2 + n_atoms]
        # Write to a temporary string buffer and reuse read_xyz logic
        from io import StringIO
        buf = "\n".join(block)
        tmp_path = path.parent / f"__tmp_frame_{frame}.xyz"
        tmp_path.write_text(buf)
        mol = read_xyz(tmp_path, charge=charge, multiplicity=multiplicity)
        mol.name = f"frame_{frame}"
        tmp_path.unlink()
        molecules.append(mol)
        i += 2 + n_atoms
        frame += 1

    return molecules


def write_xyz_trajectory(molecules, path: Union[str, Path]) -> None:
    """Write a list of Molecules as a multi-frame XYZ trajectory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks = []
    for mol in molecules:
        lines = [str(mol.n_atoms), mol.name]
        for atom in mol.atoms:
            c = atom.coords_angstrom()
            lines.append(f"{atom.symbol:<4s}  {c[0]:14.8f}  {c[1]:14.8f}  {c[2]:14.8f}")
        blocks.append("\n".join(lines))

    path.write_text("\n".join(blocks) + "\n")
