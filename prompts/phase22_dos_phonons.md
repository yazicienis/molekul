# Phase 22: Yoğunluk Fonksiyonu (DOS) + Fononlar

## Context

Phase 21 ile `band_structure()` native olarak tamamlandı. Elinde:
- `BandStructureResult`: `band_energies (n_kpts, n_bands)`, `x_coords`, `tick_positions`, `n_occ`
- `periodic_hf()`: 1D H zinciri için SCF enerjileri ve band enerjileri
- `rhf_gradient()` / `numerical_gradient()`: moleküler kuvvetler (Phase 18)

Bu faz iki bağımsız ama ilgili ekleme yapar: **DOS** ve **fononlar**.

## Part A: Durum Yoğunluğu (DOS)

### Objective

`band_structure()` sonucundan Gaussian genişletme ile DOS hesapla.

### Theory

```
DOS(E) = Σ_{n,k} w_k · δ(E − ε_{nk})
```

Dirac delta → Gaussian genişletme:
```
DOS(E) = Σ_{n,k} w_k · (1/(σ√(2π))) exp(−(E−ε_{nk})²/(2σ²))
```

Eşit ağırlıklı k-noktaları: `w_k = 1/N_k`.

### Implementation

`src/molekul/periodic.py`'ye ekle:

```python
@dataclass
class DOSResult:
    energies: np.ndarray    # E grid, shape (n_grid,)
    dos: np.ndarray         # DOS(E) değerleri, shape (n_grid,)
    e_fermi: float          # Fermi seviyesi (n_occ. band'ın max enerjisi)

def dos(
    band_result: BandStructureResult,
    n_grid: int = 500,
    sigma: float = 0.02,    # Ha cinsinden Gaussian genişlik
    e_min: float | None = None,
    e_max: float | None = None,
) -> DOSResult:
```

Fermi seviyesi: tüm k-noktalarındaki `n_occ`. band'ın maksimum özdeğeri.

### Tests (`tests/test_dos.py`)

```python
def test_dos_shape()           # energies ve dos shape (n_grid,)
def test_dos_non_negative()    # dos >= 0 her yerde
def test_dos_normalizable()    # integral(dos) * dE > 0
def test_dos_fermi_in_range()  # e_fermi, energies aralığında
def test_dos_h_chain_peak()    # DOS'un en yüksek noktası band içinde
```

---

## Part B: Harmonik Fononlar (1D H Zinciri)

### Objective

Periyodik 1D H zinciri için dinamik matrisi finite-difference kuvvet sabiteleriyle kur, fonon dispersiyonunu hesapla.

### Theory

Birim hücredeki A atomu için kuvvet sabiti tensörü:
```
Φ_{Aα, Bβ}(R) = d²E / (du_{Aα}(0) du_{Bβ}(R))
```

Central finite differences ile (h=0.01 Bohr):
```
Φ_{Aα, Bβ}(R) = -dF_{Aα}(0) / du_{Bβ}(R)
```

Dinamik matris:
```
D_{Aα, Bβ}(q) = (1/√(m_A m_B)) Σ_R e^{iq·R} Φ_{Aα, Bβ}(R)
```

Fonon frekansları: `D(q)` özdeğerleri ω²(q).

1D H zinciri için uygulanabilir boyut: a=1.8 Bohr, 1 H/hücre, 1D q-yolu.

### Implementation

`src/molekul/periodic.py`'ye ekle (veya `src/molekul/phonons.py` yeni dosya):

```python
@dataclass
class PhononResult:
    frequencies: np.ndarray      # shape (n_qpts, n_modes)
    qpoints: np.ndarray          # shape (n_qpts, 3) Cartesian Bohr^-1
    x_coords: list[float]        # çizim x ekseni
    tick_positions: list[float]
    tick_labels: list[str]

def phonon_band_structure(
    crystal: Crystal,
    basis_fn: BasisSet,
    special_points: dict[str, np.ndarray],
    path: str,
    n_points: int = 30,
    h: float = 0.01,
    r_max_factor: float = 4.0,
) -> PhononResult:
```

**Scope kısıtı:** Yalnızca 1D kristaller. 3D için `NotImplementedError` ver.

Kuvvet sabitlerini hesaplamak için `periodic_hf()` 1D kullan (SCF enerjileri gerekli).
Kuvvetler: `−dE_total/dR` → bunun için `_periodic_nuclear_repulsion_cutoff` türevi yeterli (electronic kuvvetler Phase 18 kapsamında değil — nükleer itme kuvveti ile sınırlı tut ve dokümante et).

**Not:** Bu sadece nükleer itme katkısından gelen fonon hesabıdır. Elektronik katkı olmadan fiziksel bir dispersiyon bekleme; eğitsel amaçla altyapıyı göster.

### Tests (`tests/test_phonons.py`)

```python
def test_phonon_shape()           # frequencies.shape = (n_qpts, n_modes)
def test_phonon_gamma_acoustic()  # q=0'da akustik mod ≈ 0 (translasyonel)
def test_phonon_3d_not_impl()     # 3D kristal → NotImplementedError
def test_phonon_tick_labels()     # tick_labels doğru
```

---

## Files

- `src/molekul/periodic.py` — `DOSResult`, `dos()`, `PhononResult`, `phonon_band_structure()`
  *(ya da fononlar için `src/molekul/phonons.py` yeni dosya — senin tercihin)*
- `tests/test_dos.py`
- `tests/test_phonons.py`
- `scripts/validate_dos_phonons.py`
- `outputs/logs/phase22_dos_phonons.json/.txt`

## SCIENCE.md

- DOS Gaussian sigma = 0.02 Ha
- Fonon finite-difference h = 0.01 Bohr
- Kuvvet sabitinin "nükleer itme only" kısıtını belgele

## Acceptance Criteria

- DOS shape, non-negative, integrable ✓
- Phonon shape doğru; q=0 akustik mod < 1e-3 Ha/Bohr² ✓
- 3D phonon → NotImplementedError ✓
- `pytest tests/ -x` no regressions
- Commit: Phase 22 files only

## Notes

- DOS ve fononlar birbirinden bağımsız — istersen ayrı alt fazlar olarak yap.
- Fononlarda elektronik katkı yoksa dispersiyon gerçekçi değil; bu bir öğretim noktası: "Gerçek fononlar için Hellmann-Feynman kuvvetleri (analitik gradient) gerekir."
- Phase 23 bu temeli genişletebilir.
