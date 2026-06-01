# Phase 23: Tam Fononlar (Elektronik + Nükleer Kuvvetler)

## Context

Phase 22 fonon altyapısını kurdu ama yalnızca nükleer itme kuvvet sabitleri kullandı.
Bu faz, `periodic_hf()` SCF enerjisinden finite-difference ile **tam kuvvet
sabitleri** hesaplayarak gerçek (elektronik + nükleer) fonon dispersiyonunu verir.

Phase 18'de moleküler sistemler için `numerical_gradient()` nasıl yazıldıysa, bu
fazda periyodik sistemler için aynı mantık uygulanır.

## Amaç

`src/molekul/periodic.py`'ye `periodic_force_constants()` ve
`phonon_band_structure_full()` ekle. 1D H zinciri üzerinde doğrula; Phase 22
nükleer-only sonuçla karşılaştır.

## Theory

### Kuvvet Sabiti

```
Φ_{Aα, Bβ}(R) = -∂F_{Aα}(0) / ∂u_{Bβ}(R)
              = ∂²E_total / (∂u_{Aα}(0) ∂u_{Bβ}(R))
```

4-noktalı sonlu fark (h = 0.01 Bohr, Phase 22 ile aynı):

```
Φ_{Aα,Bβ}(R) = [E(+A,+B) − E(+A,−B) − E(−A,+B) + E(−A,−B)] / (4h²)
```

Burada `E(±A, ±B)` = atom A'nın α yönünde ±h, atom B'nin (R hücresinde) β
yönünde ±h ötelendiği kristalde `periodic_hf()` toplam enerjisi.

### Akustik Toplam Kuralı

Phase 22 ile aynı: `Φ(R=0) = −Σ_{R≠0} Φ(R)`.

### Dinamik Matris ve Frekanslar

Phase 22 ile identik; sadece kuvvet sabitleri farklı (tam SCF, nükleer-only değil).

```
D_{Aα,Bβ}(q) = (1/√(m_A m_B)) Σ_R e^{iq·R} Φ_{Aα,Bβ}(R)
ω²(q) = eigenvalues(D(q))
```

## Implementasyon

### `periodic_force_constants()`

```python
def periodic_force_constants(
    crystal: Crystal,
    basis_fn: BasisSet,
    h: float = 0.01,
    r_max_factor: float = 4.0,
    scf_kwargs: dict | None = None,
) -> dict[tuple[float, float, float], np.ndarray]:
    """
    Finite-difference force constants from periodic_hf() total energy.

    Returns force constant blocks Φ(R), shape (3*n_atoms, 3*n_atoms) per R.
    Acoustic sum rule applied: Φ(R=0) = -Σ_{R≠0} Φ(R).

    Only 1D crystals are supported (periodic_hf constraint).
    """
```

**Adımlar:**
1. `crystal.lattice_vectors_in_shell(r_max_factor * max_lattice_len)` ile R vektörlerini bul
2. Her R ve her (A,α,B,β) çifti için 4 SCF hesabı:
   - `_displaced_crystal(crystal, atom_B, R, beta, +h)` ve `-h` ile ±A, ±B kombinasyonları
   - Her seferinde `periodic_hf()` çağır, `result.energy_per_cell` kullan
3. 4-noktalı formülle Φ hesapla
4. Akustik toplam kuralını uygula

### `_displaced_crystal()`

Yardımcı fonksiyon: `crystal`'ın atom `atom_idx` (hücre R'deki) α yönünde `delta` Bohr
kadar ötelenmiş kopyasını döndürür.

1D H zincirinde hücre R'deki atom B'nin ötelerken: B'nin koordinatı `coords[B] + R`
olduğundan, bunu sabit tutup crystal kopyasında sadece birim hücre koordinatını değiştirme
**yanlış** olur. Bunun yerine: crystal kopyasında B atomunun koordinatını `+delta` yap,
ama `periodic_hf()` çağrısında aynı R ile hesabı yap. Matematiksel olarak bu B'nin R
hücresindeki ötelemesiyle eşdeğerdir çünkü kuvvet sabitleri translasyonel değişmezdir.

### `phonon_band_structure_full()`

```python
def phonon_band_structure_full(
    crystal: Crystal,
    basis_fn: BasisSet,
    special_points: dict[str, np.ndarray],
    path: str,
    n_points: int = 30,
    h: float = 0.01,
    r_max_factor: float = 4.0,
) -> PhononResult:
    """
    Full phonon band structure from periodic SCF force constants.

    Same PhononResult as phonon_band_structure() but uses
    periodic_force_constants() instead of _nuclear_force_constant_blocks().
    """
```

**Scope kısıtı:** Yalnızca 1D kristaller (`periodic_hf` kısıtı). 3D için `NotImplementedError`.

## Testler

Dosya: `tests/test_phonons_full.py`

```python
def test_full_phonon_shape():
    # frequencies.shape == (n_qpts, 3) for H chain (1 atom × 3 DOF)

def test_full_phonon_gamma_acoustic():
    # q=0 akustik mod < 1e-3 (akustik toplam kuralı)

def test_full_phonon_3d_not_impl():
    # 3D crystal → NotImplementedError

def test_full_vs_nuclear_different():
    # Tam fononlar ≠ nükleer-only fononlar (X noktasında fark > 1e-3)
    # Elektronik katkının etkisini gösterir

def test_full_phonon_finite_freq_at_X():
    # X noktasında en az bir mod > 0 (optik veya akustik dispersiyon)
```

## Validation Script

Dosya: `scripts/validate_phonons_full.py`

Çıktı:
- `outputs/logs/phase23_phonons_full.json`
- `outputs/logs/phase23_phonons_full.txt`

Log: H zinciri Γ acoustic frekansı, X frekansları, Phase 22 nükleer-only ile karşılaştırma.

## SCIENCE.md

Ekle:
- Full phonon FD h = 0.01 Bohr (Phase 22 ile tutarlı)
- "Nükleer + elektronik kuvvetler dahil; Phase 22 nükleer-only ile karşılaştır"

## Acceptance Criteria

- H chain: Γ acoustic < 1e-3 (akustik toplam kuralı) ✓
- Tam fononlar ≠ nükleer-only fononlar (X noktasında fark > 1e-3) ✓
- 3D → NotImplementedError ✓
- `pytest tests/ -x` no regressions
- Log'da Phase 22 ile sayısal karşılaştırma var
- Commit: Phase 23 files only

## Pedagojik Not

Bu fazın eğitsel değeri açık şekilde gösterilmeli:
- Elektronik katkı olmadan (Phase 22): fononlar fiziğin uzağında
- Elektronik katkıyla (Phase 23): daha gerçekçi dispersiyon
- "VASP/QE'de analitik gradient var; MOLEKUL'de semi-numerical — aynı fizik, farklı maliyet"

Bu MOLEKUL'ün doğal bitiş noktası: her adım sıfırdan, her katkı izlenebilir.
