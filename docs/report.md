# Mapping Parliamentary Behaviour in Türkiye: A Big-Data Analysis of TBMM 28th-Term Written Questions

**STAT / BIG DATA 401 — Final Project Report**
Bora Esen et al. · Spring, Semester 8 · Repository: `github.com/esenbora/stat401-tbmm`

---

## Abstract

We build an end-to-end big-data pipeline over **44,484 written parliamentary
questions** (*yazılı soru önergesi*) filed in the 28th term of the Turkish Grand
National Assembly (TBMM, 2023–2026). Crucially, the analysis substrate is the
**full OCR'd text** of every question (≈ **108.6 million characters**; 65% read
from the PDF text layer, 35% recovered with PaddleOCR), not merely the one-line
official subject. Using Apache Spark, Delta Lake and Spark MLlib we answer how
questions distribute across ministries, parties and time (RQ1), where attention
falls geographically (RQ2), and who coordinates with whom (RQ3). Headline
findings: the instrument is overwhelmingly an opposition tool (96% of questions
from three opposition parties); a question's party is predictable from its full
text with **97% accuracy** (vs. 76% from the subject line alone — the gap is
identity leakage, discussed below); **8,685** near-duplicate full-document pairs
reveal heavy within-party templating; provincial attention correlates moderately
with population (Pearson **0.59**, Spearman **0.69**) and is strongly skewed
toward small eastern provinces per capita; and the co-signing network splits into
a tight DEM Parti clique versus a merged CHP/İYİ/MHP bloc (Louvain ARI = 0.46),
bridged almost single-handedly by one MP.

---

## 1. Introduction

Parliamentary written questions are a formal oversight instrument: any MP may ask
a minister a written question that must receive a written answer. Their text and
metadata make them an ideal substrate for studying legislative behaviour at
scale. We pose three instructor-approved research questions:

- **RQ1 — Ministry × Topic × Party × Time.** How are questions distributed across
  ministries, topics and parties, and how does this evolve? Do coordinated
  (duplicate) questions exist?
- **RQ2 — Provincial attention.** Which provinces draw the most attention, on
  which topics, do they cluster into latent "attention profiles", and how does
  attention relate to population and representation?
- **RQ3 — Co-signing network.** What is the structure of the MP co-signing
  network; do PageRank influence and community detection align with party lines;
  who bridges communities?

## 2. Data Collection

The source is the official TBMM website (no public API). A custom rate-limited
scraper (`src/scraper.py`, `src/mp_crawler.py`) — single thread, 1 req/s,
`Retry-After`-aware exponential backoff, append-only Parquet checkpoints —
traversed the MP roster, each MP's question GUIDs, per-question structured
metadata, and the önerge PDFs.

**Collected:** 592 MP profiles; **44,485** written-question metadata rows (every
field populated: *esas no, geliş tarihi, soru sahibi (il + MP), muhatap bakanlık,
durum, özet*, PDF links); time span 2023–2026 (7,854 / 14,448 / 16,144 / 6,038
questions per legislative year).

## 3. Preprocessing — Medallion Architecture + OCR

We use a Delta Lake **Bronze → Silver → Gold** layout.

- **Bronze** — raw scraped metadata Parquet.
- **Full-text extraction (the core big-data step).** Each önerge PDF was
  processed on a Colab GPU: where a PDF carried a usable text layer (**28,893**
  docs, 65%) we read it with `pypdf`; the remaining scanned PDFs (**15,591**,
  35%) were OCR'd with **PaddleOCR** (Turkish, mean confidence ≈ 0.94). Output:
  `onerge_text.parquet` — **100% coverage** (44,484 docs), ~108.6 M characters,
  mean ≈ 2,441 chars/doc. *This full text — not the 78-char subject — is what all
  NLP below runs on.*
- **Silver** (`src/analysis/prep.py`) — one enriched row per question: full text
  attached by GUID, party + electoral province via a broadcast join (307/307 MPs
  matched), year parsed from the date, and provinces *mentioned in the body*
  detected with an 81-province gazetteer.
- **Text normalisation for NLP.** Turkish lower-casing **+ ASCII folding** so the
  35% OCR'd text (which often drops diacritics: *çocuk→cocuk*) merges with the
  text-layer 65%; a Turkish stop-word list extended with parliamentary
  letter/legal boilerplate and party-identifier tokens.
- **Gold** — 10 analysis-ready Delta tables.

The central structural fact: question volume by party is **CHP 20,902 · DEM
15,416 · İYİ 6,545** vs. MHP 645, Yeniden Refah 355, TİP 341, **AK Parti 254**.
Written questions are an *opposition* instrument; the governing party barely uses
them. This conditions every result.

## 4. Methodology

| Component | Tool (Spark MLlib unless noted) | Used in |
|---|---|---|
| Tokenisation | RegexTokenizer + Turkish lower/ASCII-fold + StopWordsRemover | RQ1, RQ2 |
| Vectorisation | CountVectorizer, IDF | RQ1, RQ2 |
| Topic modelling | LDA (k = 12) | RQ1, RQ2 |
| Frequent itemsets | FP-Growth | RQ1 |
| Classification | Multinomial LogisticRegression | RQ1 |
| Near-duplicate detection | MinHashLSH | RQ1 |
| Clustering | KMeans (parallel k-means‖) | RQ2 |
| Correlation | `ml.stat.Correlation` + Spearman | RQ2 |
| Centrality | PageRank (DataFrame power iteration) | RQ3 |
| Community detection | Label Propagation (Spark) + Louvain (networkx) | RQ3 |
| Bridges / embeddings | betweenness, DeepWalk + UMAP (networkx/gensim) | RQ3 |

Data is modest in volume (~70 MB text), so Spark here demonstrates the
distributed *methodology* (medallion, MLlib, partitioning, broadcast joins,
LSH bucketing) rather than being performance-critical.

## 5. Results

### 5.1 RQ1 — Ministry × Topic × Party × Time

**Volume.** Justice (6,140) and Interior (5,571) lead, then Environment (4,587),
Agriculture (4,584), Health (3,520), Education (3,508); top-ministry ranks are
stable across all four legislative years.

**Topics (LDA on full text).** With boilerplate removed, the 12 topics are
coherent policy themes: **detention/justice** (*cezaevi, ceza, infaz, adalet*),
**health** (*sağlık, hastane, hasta*), **environment/energy** (*orman, tarım,
çevre, maden, enerji, ÇED*), **disaster** (*deprem, Hatay, KYK, yurt*),
**child/women/social** (*çocuk, kadın, eğitim, engelli*). Full-text LDA is far
richer than the subject-line version. FP-Growth on full text mostly surfaces
*legal-citation* templating (*"… inci maddesi gereğince …"*, date/number
phrasing) — itself evidence of formulaic drafting — while thematic co-occurrence
is better captured by LDA.

**Party is encoded in the text.** A multinomial logistic regression on TF-IDF of
the **full text** predicts the filing party (CHP/DEM/İYİ) at **97.3% accuracy**
(F1 = 0.973) vs. a 48.8% baseline. *Caveat — leakage:* full text contains MP
names, provinces and signature/letterhead cues that identify the party, so this
over-states topical distinctiveness. The cleaner, leakage-free measure is the
**subject-line model at 76% accuracy** (+27 pts over baseline): parties pursue
genuinely distinguishable agendas, and the full text additionally carries strong
identity signals.

**Coordinated agenda-setting.** MinHashLSH (Jaccard ≥ 0.8) on full documents
finds **8,685** near-duplicate question pairs (**4,080** across different MPs, 2
across parties). Because these are *whole-document* near-identities (not short
subject collisions), they are strong, conservative evidence of **within-party
templated campaigns** — most coordination is intra-party.

### 5.2 RQ2 — Provincial Attention

Attention = a province named in the question's full text, detected with a
Turkish case-suffix-aware 81-province gazetteer (e.g. *Diyarbakır'da →
Diyarbakır*; "Kars" inside *karşılaşılan* is not matched). **59,761** mentions;
84% of questions name ≥ 1 province (vs. 47% from the subject line).

**Distribution.** Top raw counts: Ankara 10,971, Kocaeli 5,518, İstanbul 4,013,
İzmir 2,832, Diyarbakır 2,182, Van 1,557. *Caveat:* full-text counts are inflated
for **Ankara** (ministries/addresses are in the capital) and for the home
provinces of prolific MPs (signature blocks) — a known cost of mining the body
rather than the subject line; we therefore lean on the **per-capita** view.

**Metro vs. rural.** Metropolitan provinces skew toward infrastructure/capital
topics; rural / eastern provinces skew toward education-access, detention and
disaster topics.

**Attention profiles (K-Means++).** k = 5 on standardised province×topic vectors,
silhouette = **0.28** (improved from the subject-line run): a small-eastern
disaster/service cluster, an İstanbul/Ankara metro cluster, etc.

**Attention vs. population.** Pearson r(attention, population) = **0.593**
(Spearman 0.692); MP count is collinear with population (r = 0.999) and adds no
independent signal. Population explains ~35% of attention variance. Per 100k,
the **most over-attended** provinces are Tunceli (375), Kars (329), Artvin (280),
Bitlis (256), Burdur (222) — small eastern/Black-Sea provinces — while the metros
are **under-attended** per capita (İstanbul ≈ 25). Attention is politically
targeted, not proportional to size.

### 5.3 RQ3 — MP Co-signing Network

Written questions are single-authored, so "co-signing" is a **proxy**: two MPs
linked if they filed questions with an *identical subject* (a campaign). This
yields **276 MPs, 7,720 weighted edges**, built in Spark; the network is an
*opposition* network by construction (AK Parti absent).

- **Centrality (Spark PageRank).** DEM MPs dominate, led by Ömer Faruk
  Gergerlioğlu. This is *partly a clique artifact* — DEM mass-campaigns inject
  large cliques that inflate centrality.
- **Communities.** Spark Label Propagation is degenerate here (collapses to one
  community — a known LPA weakness on clique-heavy graphs); **Louvain** recovers
  a 213-MP opposition bloc (CHP+İYİ+MHP), a near-pure **DEM clique (59 MPs,
  98%)**, and a 4-MP AK Parti pocket (ARI = 0.461, NMI = 0.551). The fault line
  is *DEM vs. the rest of the opposition*.
- **Embeddings.** Weighted DeepWalk + UMAP; party silhouette = **−0.22**
  (negative) — party labels do not form clean clusters, though party is the
  relatively strongest organising signal.
- **Bridges.** **Mustafa Sezgin Tanrıkulu** (CHP, Diyarbakır) is the dominant
  cross-party broker (~3× the next MP), linking the CHP bloc to the DEM clique.

## 6. Dashboard

A single Streamlit app (`src/dashboard/app.py`, three tabs) presents all results
interactively — filterable ministry volume/timeline, party prediction, an
**interactive province choropleth**, per-capita targeting, K-Means profiles, the
PageRank bar, DeepWalk embedding, Louvain composition and the bridge table. It
reads small Parquet snapshots, so it starts instantly without Spark.

## 7. Conclusion

Across three lenses a consistent picture emerges: TBMM written questions in the
28th term are an **opposition-coordinated** instrument with a strongly
text-encoded party signature; attention is **politically targeted** rather than
proportional to population, concentrating per-capita on small eastern provinces;
and coordination clusters around a **DEM core** linked to the wider opposition by
a few brokers. The lenses reinforce each other.

**Limitations.**
- Full-text province mentions include capital/ministry and signature-block noise
  (Ankara and prolific MPs' home provinces inflated); per-capita and clustering
  views mitigate this, and the subject-line signal is a cleaner but sparser
  alternative.
- The 97% full-text party classifier reflects identity leakage; 76% (subject
  line) is the honest topical-distinctiveness figure.
- The co-signing graph is a same-subject proxy; DEM centrality is partly a clique
  artifact; the network excludes the governing party.
- OCR (35% of docs) introduces residual noise despite ASCII folding.

**Future work.** Strip signatures before mention/identity analysis; second-order
Node2Vec with held-out link prediction; transcript (Genel Kurul) NLP to compare
spoken vs. written agendas; per-party stance classification.

## Reproducibility

```bash
brew install openjdk@17 && pip install -r requirements.txt
python src/analysis/prep.py                     # Bronze + full text -> Silver
python src/analysis/rq1_ministry_topic_party.py
python src/analysis/rq2_province_attention.py
python src/analysis/rq3_network_spark.py
python src/analysis/export_dashboard.py
python -m streamlit run src/dashboard/app.py
```

All randomised steps are seeded (seed = 42). Results, metrics JSON and figures
land in `deliverables/`; Gold Delta tables in `data/gold/`.

## Team Contributions

| Member | Contribution |
|---|---|
| Bora Esen | Pipeline architecture, scraping, OCR/full-text, Spark medallion, RQ1, RQ2, dashboard |
| *(teammate)* | RQ3 network analysis (initial networkx implementation) |
| *(teammate)* | *…* |
