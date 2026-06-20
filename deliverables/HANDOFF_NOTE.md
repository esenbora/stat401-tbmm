# Handoff to the team (for writing the report)

All tasks from the fix brief **and** the post-full-text addendum are done. The
report is out of scope (team rewrites it); below are the final numbers and
artefacts. (Latest substantive change: letterhead/signature stripped to `govde`,
self-province removed from mentions — this corrected the RQ2 headline.)

## Regenerated artefacts (in repo)
- `deliverables/rq{1,2,3}_metrics.json`, `rq{1,2,3}_results.md`
- `deliverables/figures/*.png`; dashboard data `src/dashboard/data/*.parquet`

## Data provenance & parameters
- **Analysis text:** `govde` (OCR full text with party letterhead + closing
  signature stripped in `prep.py`). Raw OCR is `text`; `govde` is what all NLP
  uses. **Join key:** `guid`. **OCR coverage:** 100% (44,484 docs; 28,893
  text-layer + 15,591 PaddleOCR; ~108.6 M chars). Analysis population **N = 44,484**.
- **LDA k:** RQ1 = 15, RQ2 = 12. **Silhouettes:** RQ2 K-Means = **0.270**;
  RQ3 DeepWalk/UMAP = **−0.26**.
- Pipeline: Turkish lower + ASCII-fold, stop-words (boilerplate + filler + party
  abbreviations; content words like "halk" kept), CountVectorizer `maxDF=0.4`.

## §8 one-liners (what the audit asked for)
- **New RQ2 total mention count:** **19,342** (was 59,761 before stripping the
  letterhead + removing the MP's own province — both were signature artifacts).
- **New RQ2 top-5 per-capita provinces** (Ankara excluded as a ministry-address
  artifact): **Hakkari (200), Tunceli (110), Kilis (97), Burdur (86), Kars (79)**
  — all driven by *other* MPs' attention. Validation: **Kocaeli** (Gergerlioğlu's
  home) moved over→under-attended (8/100k) once self-mentions were removed.
- **Honest classifier accuracy:** **0.957** on `govde` (party letterhead removed;
  no `0.76` cited anywhere — it isn't produced by the current code).
- **Refreshed RQ3:** **265 nodes / 6,973 edges**; Louvain **ARI 0.325** (was 0.46
  for the old identical-summary graph → attention coordination is moderately less
  party-aligned than wording). PageRank top-5: **Gergerlioğlu (DEM)**, then other
  DEM MPs + Mustafa Bilici (CHP). Bridges: Bilici, Avşar, Tanrıkulu, Ekmen (CHP).

## Other headline numbers
- **RQ1:** Adalet 6,140 / İçişleri 5,571 lead. LDA themes: detention/justice,
  earthquake/energy, environment/mining, agriculture, health, education, finance,
  child/women. MinHash near-duplicate **bodies**: 50,812 pairs (4,608 cross-MP,
  138 cross-party) → coordination overwhelmingly intra-party.
- **RQ2:** attention~population Pearson **0.65 (Ankara excluded — the headline)**
  / 0.40 with Ankara (Ankara is a ministry-address outlier). Either way attention
  is politically targeted, not purely size-driven. The dashboard computes the
  Ankara-excluded r in-app; `rq2_metrics.json` keeps the raw with-Ankara 0.4017.

## Verification
- Dashboard: Streamlit AppTest passes (3 tabs, 0 exceptions). Captions read from
  metrics (auto-refresh). Sidebar: 592 total / 307 filing / 265 in RQ3 graph.
- OCR code committed (`notebooks/03_hybrid_extract.ipynb`); corpus committed
  (`data/silver/onerge_text.parquet`, 67 MB) → reproduces from clean checkout.
- **D1** cross-party dup directionality guid-ordered, not double-counted.
  **D2** N = 44,484 (one empty-MV row dropped). **D3** PageRank clique inflation
  reduced (not eliminated) under co-attention.

## Honest limitations to carry into the report
- RQ2 Ankara raw count = ministry-location artifact (excluded from interpretation).
- RQ1 classifier 0.957 still carries closing-signature identity signal.
- FP-Growth → legal-citation boilerplate; LDA is the topic lens.
- OCR de-spacing (`GrupBaşkanlığı`) drops some run-together tokens — known limit;
  no lemmatizer used (maxDF + stop-words + minTokenLength instead).

## Re-run order
`prep.py → rq1_*.py → rq2_*.py → rq3_network_spark.py → export_dashboard.py → streamlit run app.py`
(seed = 42; JDK 17 + `pip install -r requirements.txt`).
