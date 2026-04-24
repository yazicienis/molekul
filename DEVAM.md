# Devam Noktası — 2026-04-24

## Şu an yapılacaklar (Claude gerekmez)

1. github.com/new → repo adı: `molekul` → Public → README ekleme → Create
2. Terminalde:
   ```
   git remote add origin https://github.com/yazicienis/molekul.git
   git push -u origin master
   git tag v0.1.0 -m "MOLEKUL v0.1.0 — initial JOSS submission"
   git push origin v0.1.0
   ```

## Bir sonraki Claude oturumunda söylenecek

> "JOSS submit öncesi: cc-pVDZ ve 6-31G* hatalarını düzelt, paper.md'ye MP2 validation ve known limitations ekle."

## Bilinen hatalar (düzeltilmesi gereken)

- `src/molekul/basis_ccpvdz.py` — O, F, C için yanlış exponent değerleri
- `src/molekul/basis_631gstar.py` — O valans s katsayısı yanlış
- `paper.md` — MP2 için validation satırı yok
- `paper.md` — Known limitations bölümü yok

## Proje durumu

- 606 test, hepsi geçiyor
- RHF/STO-3G: PySCF ile max ΔE < 5×10⁻⁸ Eh (14 molekül)
- LICENSE, CONTRIBUTING.md, paper.md, paper.bib hazır
- Git: 2 commit, master branch, remote henüz yok
