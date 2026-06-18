# Handoff to the team (for writing the report)

All analysis/code/data/dashboard tasks from the full-fix brief are done. The
report is **out of scope** here (team rewrites it); below are the final numbers
and artefacts to write it from.

## Regenerated artefacts (in repo)
- `deliverables/rq1_metrics.json`, `rq2_metrics.json`, `rq3_metrics.json`
- `deliverables/figures/*.png` (10 figures)
- `deliverables/rq{1,2,3}_results.md` (per-RQ write-ups)
- Dashboard data: `src/dashboard/data/*.parquet`

## Key parameters & data provenance
- **OCR full-text column:** `text` in `data/silver/onerge_text.parquet`
  (also exposed as `text` in Silver `yazili_soru_clean`). **Join key:** `guid`.
- **OCR coverage:** **100%** (44,484 / 44,484 questions; 1 metadata row with an
  empty MP name is dropped, so analysis N = 44,484). Method split: 28,893
  text-layer (pypdf) + 15,591 PaddleOCR; ~108.6 M characters.
- **LDA k:** RQ1 = **15** (finer topic words); RQ2 = **12** (lower-dim province
  vectors → better K-Means separation).
- **Silhouettes:** RQ2 K-Means = **0.259** (up from 0.166 subject-line);
  RQ3 DeepWalk/UMAP party = **−0.26** (negative — reported honestly).
- Text pipeline: Turkish lower + **ASCII fold** (merges OCR diacritic loss),
  stop-words extended with letter/legal boilerplate + party-identity tokens,
  CountVectorizer `maxDF=0.4`.

## Headline numbers
**RQ1** — Adalet 6,140 / İçişleri 5,571 lead. Full-text LDA themes:
detention/justice, health, environment/energy, education/disaster, child/women,
economy. Party predictable from full text **0.97** (identity leakage; **0.76** on
subject line is the clean topical figure). MinHash near-duplicate full documents:
**8,573** pairs (4,077 cross-MP, **2** cross-party) → coordination is intra-party.

**RQ2** — **59,761** province mentions, **84%** of questions name ≥1 province.
attention~population Pearson **0.59** / Spearman **0.69**; per-capita most
attended = small eastern provinces (Tunceli, Kars, Bitlis), least = İstanbul.
Metro vs rural top topics now differ. *Caveat:* full-text mentions inflate Ankara
(ministry addresses) and prolific MPs' home provinces.

**RQ3 (redefined)** — edges = **ministry × province × month co-attention**
(independent of RQ1's wording signal). **304 nodes, 14,908 edges**. Louvain
party-alignment **ARI 0.13** (was 0.46 for the old identical-summary graph) →
*attention coordination is far more cross-party than wording coordination*; only
DEM stays internally cohesive (58 MPs, 83%). PageRank top: Gergerlioğlu (DEM).
Bridges: Mustafa Bilici (CHP), Ekmen (CHP), Aykut Kaya (İYİ).

## Verification
- **Dashboard runs:** verified via Streamlit AppTest (3 tabs, 0 exceptions);
  `python -m streamlit run src/dashboard/app.py`.
- **OCR code committed:** `notebooks/03_hybrid_extract.ipynb` (text-layer + PaddleOCR).
- **Dataset committed:** `data/silver/onerge_text.parquet` (67 MB) + bronze meta →
  pipeline reproduces from a clean checkout.
- **D1** cross-party dup directionality is guid-ordered, not double-counted
  (moot: only 2 cross-party pairs on full text). **D2** N = 44,484 confirmed.
  **D3** PageRank no longer clique-inflated under the co-attention definition.

## Re-run order
`prep.py → rq1_*.py → rq2_*.py → rq3_network_spark.py → export_dashboard.py → streamlit run app.py`
(seed = 42 throughout). JDK 17 + `pip install -r requirements.txt` required.

## Notes / honest limitations to carry into the report
- Full-text party classifier (0.97) reflects identity leakage; cite 0.76.
- RQ2 mentions carry capital/signature noise (Ankara); lean on per-capita.
- FP-Growth on full text surfaces legal-citation boilerplate; LDA is the topic lens.
- A lemmatizer (zeyrek) was not used (time); maxDF + stop-words + minTokenLength
  used instead — a future improvement for agglutinative inflections.
