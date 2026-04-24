# Visualizing MOLEKUL CUBE Files

MOLEKUL exports volumetric data (electron density, molecular orbitals) in the
[Gaussian CUBE format](https://gaussian.com/cubegen/).  CUBE files are
supported by every major molecular visualization package.

---

## Generated Files

| File | Contents | Units |
|------|----------|-------|
| `outputs/h2o_density.cube` | Total electron density ρ(r) | e/bohr³ |
| `outputs/h2o_homo.cube` | HOMO amplitude ψ_HOMO(r) | bohr^{−3/2} |
| `outputs/h2o_lumo.cube` | LUMO amplitude ψ_LUMO(r) | bohr^{−3/2} |
| `outputs/h2o_mo1.cube` | MO 1 amplitude (O 1s core) | bohr^{−3/2} |

Run `python scripts/export_cube_h2o.py` to regenerate all files.

---

## Opening in VMD

[VMD](https://www.ks.uiuc.edu/Research/vmd/) (Visual Molecular Dynamics) is
the most capable free viewer for CUBE files.

```bash
vmd outputs/h2o_density.cube
```

**Rendering an isosurface:**

1. Open **Graphics → Representations**
2. Change *Drawing Method* to **Isosurface**
3. Set *Isovalue*:
   - Density:  `0.05` (shows ~90% electron density)
   - Orbital:  `+0.05` and `−0.05` (positive and negative lobes)
4. Set *Draw* to **Solid Surface** and *Coloring Method* to **ColorID**

**Showing both orbital lobes in two colours:**

1. Create two representations with the same CUBE file loaded
2. Rep 1: Isosurface, isovalue `+0.05`, color blue
3. Rep 2: Isosurface, isovalue `−0.05`, color red

**Tip:** For electron density, use *Coloring Method → Volume* with the
`RWB` colour scale to show high-density regions in red/warm colours.

---

## Opening in VESTA

[VESTA](https://jp-minerals.org/vesta/en/) provides excellent publication-
quality rendering and is very easy to use.

1. **File → Open** and select the `.cube` file (or drag and drop)
2. VESTA automatically detects the molecular structure and volumetric data
3. Go to **Objects → Isosurfaces**
4. Click **New** to add an isosurface; set the level:
   - Density:  `0.05` e/bohr³
   - Orbital:  `+0.05` and `−0.05` bohr^{−3/2}
5. For orbitals: add a second isosurface with negative value and a different
   colour to show both lobes

---

## Opening in Avogadro

[Avogadro](https://avogadro.cc/) (v1 or v2) supports CUBE files directly.

```bash
avogadro outputs/h2o_homo.cube
```

1. The molecule appears automatically
2. Select **Extensions → Surfaces → Add Surface**
3. Choose the CUBE data source from the dropdown
4. Adjust the isovalue (default 0.05 is usually reasonable)

---

## Opening in Jmol

[Jmol](https://jmol.sourceforge.net/) is a Java-based viewer that runs in
a browser or standalone.

**Standalone:**
```bash
jmol outputs/h2o_density.cube
```

**Scripting isosurfaces:**
```
load "outputs/h2o_density.cube"
isosurface cutoff 0.05 "outputs/h2o_density.cube"
color isosurface red blue
```

For orbital lobes:
```
isosurface ID "pos" cutoff  0.05 "outputs/h2o_homo.cube" color blue
isosurface ID "neg" cutoff -0.05 "outputs/h2o_homo.cube" color red
```

---

## Recommended Isovalue Guide

| Field | Isovalue | What it shows |
|-------|----------|---------------|
| Electron density | 0.001 | Van der Waals surface |
| Electron density | 0.05  | ~90% electron count enclosed |
| Electron density | 0.10  | Tight bonding region |
| MO amplitude | ±0.03 | Diffuse orbital (virtual/LUMO) |
| MO amplitude | ±0.05 | Standard orbital lobe |
| MO amplitude | ±0.10 | Compact orbital (core 1s) |

---

## CUBE Format Reference

The CUBE format stores data on a regular Cartesian 3-D grid (all units bohr):

```
Comment line 1
Comment line 2
NATOMS  XORIG  YORIG  ZORIG
NX  DX  0.0  0.0
NY  0.0  DY  0.0
NZ  0.0  0.0  DZ
Z  charge  x  y  z    (one line per atom)
...
val1  val2  val3  val4  val5  val6    (6 values per line, x-outer z-inner)
...
```

MOLEKUL uses:

| Parameter | Value |
|-----------|-------|
| Default step | 0.2 bohr (~0.106 Å) |
| Default margin | 4.0 bohr (~2.12 Å) beyond outermost atoms |
| Grid order | x-outer, z-inner (standard) |
| Coordinates | bohr (atomic units) |
| Density units | e/bohr³ |
| Orbital units | bohr^{−3/2} |

---

## Validation

The export script checks:

- `∫ρ dr` within 2% of the true electron count (N_electrons)
- `∫|ψ_i|² dr` within 2% of 1.0 for each exported orbital
- `ρ ≥ 0` everywhere (negative density indicates a bug)

Run with a coarser grid for quick validation:

```bash
python scripts/export_cube_h2o.py --step 0.4 --margin 3.0
```

Run with the production grid (may take several minutes):

```bash
python scripts/export_cube_h2o.py --step 0.2 --margin 4.0
```
