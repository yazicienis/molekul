# Phase 21: Band Structure (H chain + LiH, native)

## Context

Part II periyodik altyapısı tamamlandı:
- `Crystal`, `monkhorst_pack`, `bloch_overlap`, `bloch_hcore` — native (Phase 20a)
- 1D periyodik HF SCF döngüsü — native (Phase 20b)
- Ewald E_nn ve `ewald_hcore()` — native (Phase 20c)

`periodic_hf()` yalnızca 1D destekler; 3D için `NotImplementedError` verir çünkü
3D periyodik J/K integralleri Ewald-screened Coulomb gerektirir ve bu kodun
eğitsel kapsamı dışındadır.

Phase 21'in amacı: **band yapısı**. Yüksek-simetri k-yolları boyunca H_core(k)
diagonalize et, özdeğerleri çizdir. Bu tamamen native, tamamen izlenebilir ve
öğrenciye Bloch teoreminin fiziksel sonucunu — band yapısını — gösterir.

## Objective

`src/molekul/periodic.py`'ye `band_structure()` ekle. 1D H zinciri ve 3D LiH
üzerinde doğrula.

## Theory

Band yapısı: k-yolu boyunca her k noktasında

```
H_core(k) C(k) = S(k) C(k) E(k)
```

çöz. E(k) band enerjileri; k boyunca çizilince band diyagramı elde edilir.

Bu **tek-elektron (tight-binding seviyesi)** band yapısıdır — J/K terimleri
içermez. Gerçek HF/DFT band yapısı için 2-elektron terimler gerekir (bunlar
VASP/Quantum ESPRESSO tarafından yapılır). MOLEKUL burada tam şeffaflıkla
sınırını gösterir.

### k-yolu üretimi

```python
def kpath(crystal: Crystal, special_points: dict[str, np.ndarray],
          path: str, n_points: int) -> tuple[np.ndarray, list[float], list[str]]:
    """
    Yüksek-simetri noktalar arasında lineer interpolasyon.

    special_points: {"G": [0,0,0], "X": [pi/a, 0, 0], ...}  Cartesian Bohr^-1
    path: "G-X-M-G"
    n_points: her segment için nokta sayısı

    Returns:
        kpoints: (N, 3) Cartesian Bohr^-1
        x_coords: k-yolu boyunca kümülatif mesafe (çizim için x ekseni)
        tick_labels: yüksek-simetri noktaların etiketleri
    """
```

### band_structure fonksiyonu

```python
@dataclass
class BandStructureResult:
    band_energies: np.ndarray   # shape (n_kpts, n_bands)
    kpoints: np.ndarray         # shape (n_kpts, 3) Cartesian Bohr^-1
    x_coords: list[float]       # kümülatif mesafe (çizim x ekseni)
    tick_positions: list[float] # yüksek-simetri noktaların x konumları
    tick_labels: list[str]      # yüksek-simetri nokta isimleri ("Γ", "X", ...)
    n_occ: int                  # dolu band sayısı

def band_structure(
    crystal: Crystal,
    basis_fn: BasisSet,
    special_points: dict[str, np.ndarray],
    path: str,
    n_points: int = 50,
) -> BandStructureResult:
    """
    Yüksek-simetri k-yolu boyunca tek-elektron band yapısı.

    H_core(k) diagonalize edilir — J/K yok, tight-binding seviyesi.
    """
```

## Test systems

### 1D H chain (a = 1.8 Bohr)

k-yolu: Γ (k=0) → X (k=π/a)

Beklentiler:
- Tek band (1 basis fonksiyonu/hücre)
- Γ'dan X'e monoton değişim (cosine dispersion)
- Band genişliği > 0

### 3D LiH (a = 7.608 Bohr, rock-salt)

k-yolu: Γ → X → M → Γ

```python
a = 7.608  # Bohr
special_points = {
    "G": np.zeros(3),
    "X": np.array([np.pi/a, 0, 0]),
    "M": np.array([np.pi/a, np.pi/a, 0]),
}
```

Beklentiler:
- n_basis = 6 (Li: 1s,2s,2px,2py,2pz; H: 1s)
- n_occ = 2 (4 elektron / spin-2 = 2)
- En düşük 2 band dolu

## Tests

File: `tests/test_band_structure.py`

```python
def test_h_chain_band_1d_shape():
    # k-yolu N nokta, 1 band → (N, 1)

def test_h_chain_band_dispersion():
    # Γ'dan X'e band enerjisi monoton artar veya azalır

def test_h_chain_band_gamma_lt_x():
    # STO-3G H zinciri: Γ noktası X'ten düşük enerjili
    # (cosine band: alt dolu)

def test_lih_band_3d_shape():
    # LiH Γ→X→M→Γ, 50 nokta/segment → band_energies.shape = (150, 6)

def test_lih_n_occ():
    # LiH n_occ == 2

def test_band_structure_tick_labels():
    # tick_labels içinde "Γ" var

def test_band_structure_x_coords_monotone():
    # x_coords monoton artar
```

## Validation Script

File: `scripts/validate_band_structure.py`

Output:
- `outputs/logs/phase21_band_structure.json`
- `outputs/logs/phase21_band_structure.txt`

Log: H chain 1D band genişliği, LiH band gap (en düşük boş band − en yüksek dolu),
k-yolu tick_labels.

## Acceptance Criteria

- 1D H chain: tek band, Γ→X monoton dispersiyon
- 3D LiH: band_energies.shape = (n_kpts_total, 6), n_occ = 2
- `pytest tests/ -x` no regressions
- Commit: Phase 21 files only

## Notes

- Bu phase `ewald_hcore()` kullanır (Phase 20c native Ewald H_core).
- Docstring'de açıkça yaz: "Bu tight-binding seviyesi band yapısıdır.
  Gerçek quasiparticle band yapısı için HF exchange veya DFT XC katkıları
  gerekir (VASP, Quantum ESPRESSO)."
- Çizim için matplotlib gereksizdir — sadece veriyi döndür.
  İsterse kullanıcı `outputs/figures/` altına PNG kaydedebilir.
