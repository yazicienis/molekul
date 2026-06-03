# NEXT_AGENT

**Next:** Human
**Task:** Tüm fazlar tamamlandı — sıradaki adıma karar ver

## Durum

✅ **Phases 1–23 eksiksiz tamamlandı.** 682 test geçiyor.

### Tamamlanan her şey

**Part I — Moleküler kuantum kimyası (Phases 1–17):**
RHF, MP2, CCSD/CCSD(T), KS-DFT, CIS, EOM-CCSD, UHF, TD-DFT,
geometri optimizasyonu, harmonik frekanslar, cube dosyaları,
popülasyon analizi, 3 basis seti (STO-3G, 6-31G*, cc-pVDZ)

**Part II — Periyodik sistemler + GPU (Phases 18–23):**
- 18: Semi-numerical RHF gradient
- 19: CuPy GPU backend
- 20a/b/c: Periyodik HF altyapısı (Crystal, Bloch, 1D SCF, Ewald)
- 21: Band yapısı (H zinciri + LiH, native)
- 22: DOS + nükleer-only fononlar
- 23: Tam fononlar (elektronik + nükleer, SCF enerjisinden FD)

## Olası sonraki adımlar

1. **Notebook yazımı** — PHASES.md'de planlanan, her faz için Jupyter notebook.
   Bu tamamen senin (human) sesinin eseri olacak — ajan yazamaz.

2. **Commit + release** — Bekleyen tüm değişiklikleri commit et, v0.2.0 tag at.

3. **SoftwareX revision** — Hakem raporu geldiyse `paper_corrections_pending.txt`
   uygulanır (4 madde hazır).
