# STAT 401 — TBMM Parliamentary Big Data: Final Report

*Bora Esen et al. — 28th legislative term · Semester 8*

## 1. Introduction

We analyse Türkiye's parliamentary activity in the 28th term of the Grand National Assembly (TBMM) using big-data tooling covered in BIG DATA 401 (Apache Spark 3.5, Delta Lake, Spark MLlib). Three research questions, approved by the instructor:

- **RQ1** — How are written parliamentary questions distributed across ministries, parties, topics, and time? Are there duplicate (coordinated) questions?
- **RQ2** — Which provinces dominate parliamentary discourse, on which topics, and can provinces be clustered into latent "attention profiles"?
- **RQ3** — What is the structure of MP co-signing networks, do PageRank centralities align with party boundaries, and which MPs bridge communities?

## 2. Data Collection

Source: official TBMM public web pages (no API). Three endpoint families:

| Endpoint | Use |
|---|---|
| `tbmm.gov.tr/milletvekili/AllList` | enumerate all 592 28th-term MPs (province, party, GUID) |
| `tbmm.gov.tr/milletvekili/MilletvekiliDetay?Id=<GUID>` | per-MP profile (commissions, biography) |
| `tbmm.gov.tr/milletvekili/UyeninSahibiOlduguYaziliSoruOnergeleri?...` | list written-question GUIDs per MP |
| `tbmm.gov.tr/Denetim/Yazili-Soru-Onergesi-Detay/<GUID>` | per-question structured metadata + PDF links |
| `cdn.tbmm.gov.tr/KKBSPublicFile/.../*.pdf` | önerge + cevap full text (scanned) |

Custom scraper (`src/scraper.py`, `src/mp_crawler.py`) — rate-limited single-thread (1 req/s), `Retry-After` aware, `tenacity` exponential backoff, persistent SQLite-style append-only Parquet writes for crash resilience.

**Volume collected:** *(filled at run end)*
- 592 MP profiles
- ~10 000 written-question metadata rows
- ~20 000 PDFs (önerge + cevap)
- ~120 MB compressed raw layer

## 3. Data Preprocessing

**Bronze → Silver pipeline (Delta Lake Medallion architecture, syllabus Week 1).**

- **OCR**: TBMM önerge PDFs are scanned. We ran PaddleOCR 2.x (`lang=latin`) on a Colab Pro T4 GPU. Validation POC: 28 lines/page, mean confidence **0.944**, 1.55 s/page. Batch processing with 500-PDF checkpoints survives session timeouts.
- **Text normalisation**: NFKC unicode normalisation, whitespace collapse, ASCII-fold control characters, Turkish stop-word removal.
- **Entity extraction**: closed-dictionary regex for 81 Turkish provinces (no NER required because they're a closed set). Ministry names are normalised against the cabinet roster.
- **Joins**: written-question GUID ↔ submitting MP ↔ party ↔ province.

**Storage layout**

```
data/bronze/       raw HTML metadata + binary PDFs
data/silver/       OCR-emitted text + cleaned + joined Delta tables
data/gold/         analysis-ready aggregates (per-RQ partitions)
```

## 4. Methodology

| Component | Tool | Lecture |
|---|---|---|
| Distributed compute | PySpark 3.5 + Delta Lake | W1, W2 |
| Frequent itemset | `pyspark.ml.fpm.FPGrowth` | W6 |
| Near-duplicate detection | `pyspark.ml.feature.MinHashLSH` | W3 |
| Topic modelling | `pyspark.ml.clustering.LDA` (k=20) | W4 (extension) |
| Distinct cardinality | `approx_count_distinct` (HyperLogLog) | W4 |
| Clustering | `pyspark.ml.clustering.KMeans` (k-means‖) | W8 |
| Matrix factorisation | `pyspark.ml.recommendation.ALS` (RQ2 latent factors) | W10 |
| Centrality | `networkx.pagerank` + Louvain | W5 + W11 |
| Embedding | UMAP + PCA | W12 |

**Scalability (syllabus 20 %)**: Delta partition pruning by `year`, broadcast joins for MP→party lookup, `OPTIMIZE` + `ZORDER BY (muhatap_bakanlık, year)` on Gold ministry table, MinHash LSH with 8 hash tables for sub-second similarity search across ~10K önerge.

## 5. Results

*(populated post-execution; placeholders below)*

### 5.1 RQ1 — Ministry × Topic × Party × Time
- Top ministries by question volume: TBD.
- LDA topics: TBD (20 topics described by 10 words each).
- FP-Growth association rules: notable (party, ministry, topic) triples with lift > 1.5.
- MinHash duplicates: N candidate pairs with Jaccard ≥ 0.8 — evidence of coordinated agenda-setting?

### 5.2 RQ2 — Provincial Attention
- Bar chart of top-30 provinces by mention count.
- K-Means clusters: 5 attention profiles (earthquake-impacted, agricultural, metropolitan, border-security, average).
- PCA scatter for dashboard.

### 5.3 RQ3 — MP Network
- 592 nodes, ~K co-signing edges, average degree ~D.
- Louvain modularity Q = T (good partition).
- Top-20 MPs by PageRank.
- Community–party purity ≈ X % — quantifies cross-party bridging.

## 6. Dashboard

Single Streamlit app (`src/dashboard/app.py`), three tabs (Ministry, Provinces, Network), reads Gold Delta tables. Global sidebar filters for year-range and parties. Plotly + Matplotlib charts; clickable scatter plots with hover tooltips. Run locally with `streamlit run src/dashboard/app.py`.

## 7. Conclusion

Three findings answering the approved research questions. Pipeline ships as a reproducible repo (`stat401-tbmm/`) with notebooks numbered for execution order, a Streamlit demo, and Delta-Lake gold tables suitable for re-querying.

**Limitations**
- 28th-term only; broader historical comparison would require additional scraping for previous terms.
- LDA topic count fixed at k=20 (sensitivity analysis future work).
- PaddleOCR `latin` model occasionally misses Turkish diacritics; OCR confidence flagged where < 0.6.

**Improvements (future work)**
- Add Genel-Kurul tutanak NLP for cross-validating online discourse versus written questions.
- Sentiment / stance classification per party.
- Real-time streaming ingestion as new önergeler are submitted (Apache Kafka, syllabus Week 4 extension).

## Team Contributions

| Member | Contribution |
|---|---|
| Bora Esen | Pipeline architecture, scraping, OCR, RQ1 |
| *TBD* | *TBD* |
| *TBD* | *TBD* |
| *TBD* | *TBD* |
| *TBD* | *TBD* |
