"""Tests for io_xyz.py: parser, writer, roundtrip."""

import tempfile
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molekul.io_xyz import read_xyz, write_xyz, read_xyz_trajectory, write_xyz_trajectory
from molekul.atoms import Atom
from molekul.molecule import Molecule
from molekul.constants import ANGSTROM_TO_BOHR


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class TestReadXYZ:
    def test_read_h2(self):
        mol = read_xyz(EXAMPLES / "h2.xyz")
        assert mol.n_atoms == 2
        assert all(a.symbol == "H" for a in mol.atoms)

    def test_read_h2o(self):
        mol = read_xyz(EXAMPLES / "h2o.xyz")
        assert mol.n_atoms == 3
        assert mol.atoms[0].symbol == "O"
        assert mol.atoms[1].symbol == "H"
        assert mol.atoms[2].symbol == "H"

    def test_read_heh_plus(self):
        mol = read_xyz(EXAMPLES / "heh_plus.xyz", charge=1, multiplicity=1)
        assert mol.charge == 1
        assert mol.n_electrons == 2  # Z_He + Z_H - charge = 2 + 1 - 1 = 2
        assert mol.atoms[0].symbol == "He"
        assert mol.atoms[1].symbol == "H"

    def test_coordinates_converted_to_bohr(self):
        mol = read_xyz(EXAMPLES / "h2.xyz")
        # H2 bond in file is 0.74 Ang
        dz = abs(mol.atoms[1].coords[2] - mol.atoms[0].coords[2])
        expected = 0.740000 * ANGSTROM_TO_BOHR
        np.testing.assert_allclose(dz, expected, rtol=1e-6)

    def test_bad_atom_count(self, tmp_path):
        bad = tmp_path / "bad.xyz"
        bad.write_text("3\ncomment\nH 0 0 0\nH 0 0 1\n")  # says 3 but only 2
        with pytest.raises(ValueError, match="Expected 3 atoms"):
            read_xyz(bad)

    def test_non_numeric_coords(self, tmp_path):
        bad = tmp_path / "bad2.xyz"
        bad.write_text("1\ncomment\nH 0 0 x\n")
        with pytest.raises(ValueError, match="Non-numeric"):
            read_xyz(bad)

    def test_too_short_file(self, tmp_path):
        bad = tmp_path / "short.xyz"
        bad.write_text("1\n")
        with pytest.raises(ValueError, match="too short"):
            read_xyz(bad)


class TestWriteXYZ:
    def test_write_then_read(self, tmp_path):
        mol = read_xyz(EXAMPLES / "h2.xyz")
        out = tmp_path / "h2_out.xyz"
        write_xyz(mol, out)
        mol2 = read_xyz(out)
        np.testing.assert_allclose(mol.coords_bohr, mol2.coords_bohr, atol=1e-8)

    def test_write_preserves_symbols(self, tmp_path):
        mol = read_xyz(EXAMPLES / "h2o.xyz")
        out = tmp_path / "h2o_out.xyz"
        write_xyz(mol, out)
        mol2 = read_xyz(out)
        assert [a.symbol for a in mol2.atoms] == ["O", "H", "H"]

    def test_write_creates_parent_dirs(self, tmp_path):
        mol = read_xyz(EXAMPLES / "h2.xyz")
        deep = tmp_path / "a" / "b" / "c" / "h2.xyz"
        write_xyz(mol, deep)
        assert deep.exists()

    def test_comment_line(self, tmp_path):
        mol = read_xyz(EXAMPLES / "h2.xyz")
        out = tmp_path / "h2_comm.xyz"
        write_xyz(mol, out, comment="my custom comment")
        lines = out.read_text().splitlines()
        assert lines[1] == "my custom comment"


class TestTrajectory:
    def test_trajectory_roundtrip(self, tmp_path):
        mol1 = read_xyz(EXAMPLES / "h2.xyz")
        mol2 = read_xyz(EXAMPLES / "h2.xyz")
        mol2.atoms[1].coords[2] += 0.1  # small displacement
        mol1.name = "frame_0"
        mol2.name = "frame_1"

        traj_path = tmp_path / "traj.xyz"
        write_xyz_trajectory([mol1, mol2], traj_path)
        frames = read_xyz_trajectory(traj_path)

        assert len(frames) == 2
        np.testing.assert_allclose(frames[0].coords_bohr, mol1.coords_bohr, atol=1e-8)
        np.testing.assert_allclose(frames[1].coords_bohr, mol2.coords_bohr, atol=1e-8)
