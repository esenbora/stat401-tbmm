# STAT 401 — TBMM Parliamentary Big Data Project

End-to-end big data pipeline on Turkish parliamentary activities (TBMM 28th term).

## Research Questions (approved)

- **RQ1** — Ministry × Topic × Party × Time: how are written questions distributed and how does the distribution evolve?
- **RQ2** — Provincial attention map: which provinces dominate parliamentary discourse, on which topics, and how do they cluster?
- **RQ3** — MP co-signing network + influence: who are the central actors, and do communities mirror party lines?

## Stack

| Layer | Tool |
|-------|------|
| Scraping | `requests` + `BeautifulSoup` + rate limiter |
| OCR | PaddleOCR (Colab Pro GPU) |
| Lakehouse | Delta Lake (Bronze → Silver → Gold) |
| Processing | PySpark 3.5 + Spark MLlib |
| Algorithms | FP-Growth, MinHash/LSH, K-Means++, ALS, PageRank, Louvain, PCA/UMAP |
| Dashboard | Streamlit + Plotly + pyvis + folium |

## Repo Layout

```
stat401-tbmm/
├── notebooks/        # Jupyter / Colab notebooks (numbered execution order)
├── src/              # Reusable Python modules
│   ├── scraper.py    # TBMM HTML + PDF Bronze ingestion
│   ├── spark_utils.py
│   └── dashboard/    # Streamlit app
├── data/             # Delta tables (gitignored)
│   ├── bronze/       # raw HTML metadata + PDF binary
│   ├── silver/       # OCR'd text + cleaned + joined
│   └── gold/         # analysis-ready aggregates
├── docs/             # design notes, RQ approval, methodology
└── requirements.txt
```

## Execution Order

Bronze scrape is already complete (`data/bronze/yazili_soru_meta.parquet`, 44,485
questions). The analysis pipeline runs on Spark MLlib + Delta Lake from the repo
root, in order:

```bash
# 0. Prereqs: JDK 17 (brew install openjdk@17) + pip install -r requirements-pipeline.txt
.venv/bin/python src/analysis/prep.py                     # Bronze -> Silver (enrich)
.venv/bin/python src/analysis/rq1_ministry_topic_party.py # RQ1 -> gold + figures + metrics
.venv/bin/python src/analysis/rq2_province_attention.py   # RQ2 -> gold + figures + metrics
.venv/bin/python src/analysis/rq3_network_spark.py        # RQ3 -> gold + figures + metrics
.venv/bin/python src/analysis/export_dashboard.py         # gold Delta -> dashboard parquet
.venv/bin/python -m streamlit run src/dashboard/app.py    # interactive 3-tab dashboard
```

Results land in `deliverables/` (`rq{1,2,3}_results.md`, `rq{1,2,3}_metrics.json`,
`figures/`) and `data/gold/` (Delta tables). The original networkx RQ3 (pre-port)
is kept under `notebooks/rq3_network/` for provenance.

**Spark MLlib used:** LDA, FP-Growth, CountVectorizer, IDF, LogisticRegression,
MinHashLSH, KMeans (k-means++), Correlation, plus DataFrame-based PageRank +
Label Propagation. Louvain / betweenness / DeepWalk embeddings run in
networkx/gensim (no Spark equivalent).

## Live Dashboard (Streamlit Community Cloud)

The dashboard reads the committed parquet/JSON snapshots, so it runs with no Spark
and only the light root `requirements.txt` (streamlit, pandas, pyarrow, plotly).

Deploy (one-time, free): go to **share.streamlit.io** → *New app* → pick this repo,
branch `main`, **main file path `src/dashboard/app.py`** → *Deploy*. You get a public
`https://<app>.streamlit.app` URL. (Streamlit Cloud installs the root
`requirements.txt`; the heavier analysis pipeline lives in `requirements-pipeline.txt`.)

Run locally: `python -m streamlit run src/dashboard/app.py`.

## Setup

```bash
cd stat401-tbmm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-pipeline.txt

# Smoke tests
python src/spark_utils.py     # → "Spark + Delta ready"
python src/scraper.py --test  # → 1 sample önerge metadata
```

## Rate Limiting

Single thread, 1 req/sec, exponential backoff on 429. **No multi-thread scraping** (TBMM IP-bans aggressive scrapers — see `koezgen/Turkish_MP_Prediction` warning).

## Data Volumes (est.)

| Layer | Size |
|-------|------|
| Bronze metadata Parquet | ~5 MB (10K rows) |
| Bronze PDF binary | ~10-30 GB (20K PDFs) |
| Silver OCR text | ~500 MB (compressed Parquet) |
| Gold aggregates | ~10 MB |

## Evaluation Alignment (syllabus Term Project)

- Technical depth (40%): 6+ Spark MLlib algos from Weeks 3, 5, 6, 8, 10, 11
- Scalability (20%): Spark partitioning + Delta `OPTIMIZE/ZORDER` + Medallion
- Documentation (20%): this README + inline docstrings + report
- Presentation (20%): 20 min demo with live dashboard
