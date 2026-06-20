# RQ2 — Provincial Attention Map (Results)

Computed on the question body (`govde`) of all 44,484 questions with Apache Spark
MLlib (LDA, K-Means++, Correlation). "Attention" = a province named in the body,
detected with a Turkish case-suffix-aware 81-province gazetteer. **Two
bias-removal steps** (vs. the earlier full-text run):

1. mentions are read from `govde` (party letterhead + "<MP> / <il> Milletvekili"
   stripped), and
2. the MP's **own electoral province is removed** from their mentions —
   otherwise every question trivially "mentions" the MP's home province via the
   signature, making the per-capita map a signature artifact.

Result: **19,342** genuine cross-province mentions (down from the inflated 59,761).

## S1 — Geographic distribution

Raw top: **Ankara 6,098**, İstanbul 798, Hakkari 556, Van 519, Diyarbakır 437.

> **Ankara caveat (important):** Ankara's raw count is a *capital/ministry*
> artifact — question bodies constantly reference ministries located in Ankara
> ("Ankara'daki … Bakanlığı"), not Ankara-as-subject. Ankara is therefore
> excluded from the attention interpretation below; the genuine signal is the
> eastern/Anatolian provinces.

## S2 — Topics per province; metro vs. rural

Metro-skewed vs rural-skewed topic shares now differ ([3,11,5] vs [7,1,0]):
metros skew to infrastructure/finance topics, rural/eastern provinces to
detention, education-access and disaster topics.

## S3 — Attention profiles (K-Means++)

k = 5 on standardised (province × 12-topic) vectors, silhouette = **0.270**
(> the 0.166 subject-line baseline).

## S4 — Attention vs. population & representation

| Pair | Pearson | Spearman |
|---|---|---|
| attention ~ population (Ankara excluded) | **0.65** | — |
| attention ~ population (with Ankara) | 0.40 | 0.63 |
| population ~ MP count | 0.999 | — |

Headline = **Pearson 0.65 with Ankara excluded** (Ankara's mention count is a
ministry-address artifact, ~1/3 of all mentions, so it's a low-attention/
high-population outlier that drags the raw correlation down to 0.40). Excluding
it, population explains ~43% of attention variance — moderate. Either way, after
removing the self-province artifact the relationship is well below the inflated
0.59 of the buggy run: attention is **substantially politically targeted**, not
just population-driven.

Per 100k inhabitants (Ankara excluded as a ministry artifact):

- **Most over-attended:** Hakkari (200), Tunceli (110), Kilis (97), Burdur (86),
  Kars (79), Erzincan (78) — small eastern/Anatolian provinces, now driven by
  **other** MPs' attention (not the province's own MPs).
- **Most under-attended:** Çanakkale (10), Tekirdağ (9), **Kocaeli (8)**, İzmir
  (8), Bursa (8) — western metros.

**Validation of the fix:** Kocaeli — home of the single most prolific MP
(Gergerlioğlu) — moved from *over-attended* (in the buggy run) to *under-attended*
(8/100k) once self-mentions were removed, confirming the old ranking was a
signature artifact.

---

### Limitations
- Ankara raw mentions reflect ministry locations, not subject attention
  (excluded from interpretation).
- Mentions read from `govde`; residual closing signatures may leak a few
  self-province mentions despite the `array_except` removal.
- GDP not included; population + MP count cover size/representation (collinear).
