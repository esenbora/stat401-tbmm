# RQ1 — Ministry × Topic × Party × Time (Results)

How are written parliamentary questions distributed across ministries, topics,
and parties in the TBMM 28th term, and how does the distribution evolve over
time? Computed on the full **44,484** written questions with **Apache Spark
MLlib** (LDA, FP-Growth, CountVectorizer/IDF, multinomial Logistic Regression,
MinHash/LSH). Gold tables written to `data/gold/{ministry_year, ministry_topic,
party_ministry, duplicate_pairs}`; figures in `deliverables/figures/rq1_*`.

---

## S1 — Ministry volume and ranking shift over time

Top ministries by written-question volume (28th term):

| Rank | Ministry | Questions |
|---|---|---|
| 1 | Adalet (Justice) | 6,140 |
| 2 | İçişleri (Interior) | 5,571 |
| 3 | Çevre, Şehircilik ve İklim | 4,587 |
| 4 | Tarım ve Orman | 4,584 |
| 5 | Sağlık (Health) | 3,520 |
| 6 | Milli Eğitim | 3,508 |
| 7 | Ulaştırma ve Altyapı | 2,685 |
| 8 | Çalışma ve Sosyal Güvenlik | 2,158 |

Justice and Interior dominate — driven by the opposition's focus on detention
conditions, rule-of-law and security (see the topic and duplicate findings
below). The `ministry_year` matrix (figure `rq1_s1_ministry_time.png`) shows the
top ministries are stable in rank across the four legislative years; total
volume rises 2023 → 2025 (7,854 → 16,144) before the partial 2026 (6,038).

## S2 — Latent topics (LDA) and term co-occurrence (FP-Growth)

A 12-topic LDA over the question summaries (log-perplexity ≈ 6.58) recovers
coherent themes, e.g.:

- **T06 — detention / prisons:** *cezaevinde, kapalı, mahkumun, tipi* (the
  single most distinctive theme; concentrated in Justice).
- **T00 — Feb-2023 earthquake:** *şubat, kahramanmaraş, hatay, depremler*.
- **T05 — agriculture / municipal:** *tarım, istanbul, belediye*.
- **T07 — infrastructure projects:** *proje, ankara, planlanan*.

FP-Growth on the per-question token sets confirms the prison cluster as the
strongest co-occurrence (highest-support itemsets all involve
`cezaevinde / kapalı / mahkumun / bulunan`, support > 1,600). High-confidence
association rules (confidence = 1.0) include
`{yüksek, cezaevinde} ⇒ {güvenlikli}` and `{mücadeleye, ilinin, ilçesinde} ⇒ {şap}`
(animal-disease outbreak), showing tightly templated question wording.

## S3 — Party differences & predicting party from text

Parties differ sharply in ministerial focus (figure `rq1_s3_party_ministry.png`,
row-normalised). A **multinomial logistic regression** trained on TF-IDF of the
question text predicts the submitting party among the three high-volume
opposition parties (CHP / DEM / İYİ):

| Metric | Value |
|---|---|
| Accuracy | **0.761** |
| Weighted F1 | **0.754** |
| Majority-class baseline | 0.488 |

The model beats the baseline by +27 points — **a question's party affiliation is
strongly encoded in its text**, i.e. parties pursue distinguishable agendas. The
confusion matrix (`rq1_s3_confusion.png`) shows DEM is the most separable (its
prison/human-rights vocabulary is highly characteristic); CHP↔İYİ is the main
confusion (overlapping municipal/economic agendas).

> Context: 96% of all written questions come from CHP (20,902), DEM (15,416) and
> İYİ (6,545); the governing AK Parti filed only 254. Written questions are an
> *opposition* instrument, so RQ1's party axis is effectively an
> opposition-agenda comparison.

## S4 — Near-duplicate / coordinated questions (MinHash/LSH)

MinHash/LSH (5 hash tables) over question token sets, retaining pairs with
**Jaccard ≥ 0.8**:

| Measure | Pairs |
|---|---|
| Near-duplicate pairs | **109,443** |
| …across different MPs | 52,482 |
| …across different parties | 14,461 |

Near-duplication is pervasive — strong evidence of **coordinated, templated
agenda-setting**, not just within a party but across the opposition. The most
frequent cross-party near-duplicate coordination is **DEM ↔ CHP** (≈ 8,170
pairs), then İYİ ↔ CHP and İYİ ↔ DEM. This independently corroborates the RQ3
network finding that CHP and DEM coordinate closely (and that Sezgin Tanrıkulu
bridges the two).

---

### Limitations

- Topics are modelled on the official one-line *summary* (`özet`, median ~78
  chars), not the full PDF body, so LDA themes are coarse.
- The party classifier is restricted to CHP/DEM/İYİ (the only parties with
  enough volume to learn); MHP/AKP are too sparse for reliable supervised
  learning.
- "Near-duplicate" counts pairs, so a single 54-MP campaign contributes
  C(54,2) ≈ 1,431 pairs; the pair count signals coordination intensity, not the
  number of distinct campaigns (32,772 distinct summaries exist; 2,060 are exact
  multi-MP campaigns).
