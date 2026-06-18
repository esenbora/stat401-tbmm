# RQ1 — Ministry × Topic × Party × Time (Results)

Computed on the **full OCR'd text** of all 44,484 written questions (≈108.6 M
chars; 65% PDF text-layer, 35% PaddleOCR) — not the one-line subject — with
Apache Spark MLlib (LDA, FP-Growth, CountVectorizer/IDF, multinomial
LogisticRegression, MinHashLSH). Gold tables in `data/gold/{ministry_year,
ministry_topic, party_ministry, duplicate_pairs}`; figures `rq1_*`.

## S1 — Ministry volume & ranking over time

| Rank | Ministry | Questions |
|---|---|---|
| 1 | Adalet (Justice) | 6,140 |
| 2 | İçişleri (Interior) | 5,571 |
| 3 | Çevre, Şehircilik ve İklim | 4,587 |
| 4 | Tarım ve Orman | 4,584 |
| 5 | Sağlık | 3,520 |
| 6 | Milli Eğitim | 3,508 |

Top ministries are stable in rank across all four legislative years; total volume
rises 2023→2025 (7,854→16,144).

## S2 — Topics (LDA) and co-occurrence (FP-Growth)

12-topic LDA on full text (after removing parliamentary letter/legal boilerplate
and ASCII-folding OCR diacritics) yields coherent policy themes:

- **Detention / justice** — *cezaevi, ceza, infaz, adalet, mahpus*
- **Health** — *sağlık, hastane, hasta, devlet*
- **Environment / energy** — *orman, tarım, çevre, maden, enerji, ÇED, elektrik*
- **Disaster** — *deprem, Hatay, KYK, yurt*
- **Child / women / social** — *çocuk, kadın, eğitim, engelli, soruşturma*

FP-Growth on full text mostly surfaces **legal-citation templating** (*"… inci
maddesi gereğince …"*, date/number phrasing) — evidence of formulaic drafting;
thematic co-occurrence is better captured by LDA above.

## S3 — Party differences & predicting party from text

A multinomial logistic regression on TF-IDF of the **full text** predicts the
filing party (CHP/DEM/İYİ):

| Model | Accuracy | F1 | Baseline |
|---|---|---|---|
| Full text | **0.973** | 0.973 | 0.488 |
| Subject line only (clean) | **0.76** | 0.754 | 0.488 |

> **Leakage caveat:** full text contains MP names, provinces and
> signature/letterhead cues that identify the party, inflating accuracy. The
> **76%** subject-line figure is the honest measure of *topical* distinctiveness
> (+27 pts over baseline); parties pursue genuinely distinct agendas, and the
> full text additionally carries strong identity signals.

96% of all questions come from CHP/DEM/İYİ; AK Parti filed only 254 — the party
axis is effectively an opposition-agenda comparison.

## S4 — Near-duplicate / coordinated questions (MinHash/LSH)

MinHashLSH (Jaccard ≥ 0.8) over **full-document** token sets:

| Measure | Pairs |
|---|---|
| Near-duplicate pairs | **8,685** |
| …across different MPs | 4,080 |
| …across different parties | 2 |

Because these are *whole-document* near-identities (not short subject
collisions), they are strong, conservative evidence of **within-party templated
campaigns**; near-identical coordination across parties is essentially absent at
the full-document level.

---

### Limitations
- FP-Growth on full text is dominated by legal-citation boilerplate; LDA is the
  better topical lens.
- The 97% full-text classifier reflects identity leakage; 76% (subject) is the
  clean topical figure.
- OCR (35% of docs) adds residual noise despite Turkish ASCII folding.
