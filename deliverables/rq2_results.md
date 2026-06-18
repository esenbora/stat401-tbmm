# RQ2 — Provincial Attention Map (Results)

Computed on the **full OCR'd text** of all 44,484 written questions with Apache
Spark MLlib (LDA, K-Means++ , Correlation). "Attention" = a province named in the
question body, detected with a Turkish case-suffix-aware 81-province gazetteer
(*Diyarbakır'da → Diyarbakır*; "Kars" inside *karşılaşılan* is not matched). Gold
tables `data/gold/{province_mentions, province_topic, province_cluster,
province_correlates}`; figures `rq2_*`.

**59,761** province mentions; **84%** of questions name ≥1 province (vs. 47% from
the subject line — full text is far richer).

## S1 — Geographic distribution

| Rank | Province | Mentions |
|---|---|---|
| 1 | Ankara | 10,971 |
| 2 | Kocaeli | 5,518 |
| 3 | İstanbul | 4,013 |
| 4 | İzmir | 2,832 |
| 5 | Diyarbakır | 2,182 |
| 6 | Van | 1,557 |

> **Caveat:** full-text counts are inflated for **Ankara** (ministries/addresses
> sit in the capital) and for the home provinces of prolific MPs (signature
> blocks). Raw rank is therefore read together with the per-capita view below.

## S2 — Topics per province; metro vs. rural

Metropolitan provinces skew toward infrastructure/capital and
public-administration topics; rural / eastern provinces skew toward education
access, detention and disaster topics.

## S3 — Attention profiles (K-Means++)

k = 5 on standardised (province × 12-topic) vectors, silhouette = **0.259**
(improved over the subject-line run's 0.166; topic k kept at 12 — k=15 raises
vector dimensionality and lowers separation to 0.17): clusters separate a
small-eastern
disaster/service profile, an İstanbul/Ankara metro profile, and agriculture /
tourism profiles.

## S4 — Attention vs. population & representation

| Pair | Pearson | Spearman |
|---|---|---|
| attention ~ population | **0.593** | 0.692 |
| attention ~ MP count | 0.592 | 0.691 |
| population ~ MP count | 0.999 | — |

Population explains ~35% of attention variance; MP count is collinear with
population (seats are population-allocated) and adds no independent signal.
Per 100k inhabitants:

- **Most over-attended:** Tunceli (375), Kars (329), Artvin (280), Bitlis (256),
  Burdur (222) — small eastern / Black-Sea provinces.
- **Most under-attended:** İstanbul (~25), Çanakkale, Kastamonu, Sinop — large /
  western provinces.

Parliamentary attention is **politically targeted**, not proportional to size:
small eastern provinces draw far more attention per capita than the metros.

---

### Limitations
- Full-text mentions include capital/ministry and signature-block noise (Ankara,
  prolific MPs' home provinces); per-capita and clustering views mitigate this.
- Topic clusters inherit summary-level coarseness and OCR noise.
- GDP not included; population + MP count cover "size" and "representation" and
  are mutually collinear.
