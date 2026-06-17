# RQ2 — Provincial Attention Map (Results)

Which provinces receive the most parliamentary attention, on which topics, and
can provinces be clustered into latent "attention profiles"? Computed with
**Apache Spark MLlib** (LDA, K-Means with parallel k-means++ init, Correlation)
over the **44,484** written questions. "Attention" = a province being *mentioned*
in a question summary, detected with an 81-province gazetteer (Turkish
case-suffix aware; e.g. `Diyarbakır'da → Diyarbakır`, no false hit for `Kars`
inside `karşılaşılan`). Gold tables: `data/gold/{province_mentions,
province_topic, province_cluster, province_correlates}`; figures `rq2_*`.

**21,519** province mentions span all **81** provinces.

---

## S1 — Geographic distribution of attention

Most-mentioned provinces:

| Rank | Province | Mentions |
|---|---|---|
| 1 | Kocaeli | 1,206 |
| 2 | Diyarbakır | 1,110 |
| 3 | Şanlıurfa | 1,080 |
| 4 | İstanbul | 952 |
| 5 | İzmir | 928 |
| 6 | Van | 688 |
| 7 | Ankara | 656 |
| 8 | Bitlis | 590 |
| 9 | Antalya | 541 |
| 10 | Hakkari | 534 |

The raw ranking mixes large metros (İstanbul, İzmir, Ankara) with south-eastern
provinces (Diyarbakır, Şanlıurfa, Van, Bitlis, Hakkari) that are far smaller —
already hinting that attention is **not** purely population-driven (quantified in
S4). Figure: `rq2_s1_province_mentions.png`.

## S2 — Topics per province; metropolitan vs rural

Using the per-question LDA dominant topic exploded over mentioned provinces, the
generic "local-issue" topics dominate everywhere, so the informative signal is
the **share difference** between the 14 metropolitan provinces and the rest
(figure `rq2_s2_metro_rural.png`):

- **Metropolitan-skewed:** infrastructure/regional projects (T05, T07) and
  public-administration/headcount questions (T08).
- **Rural / eastern-skewed:** education & service access (T03), **detention /
  prisons** (T06), and **earthquake / disabled-affected** (T11).

So metros draw project- and service-delivery questions, while smaller eastern
provinces draw rights-, disaster- and access-focused questions.

## S3 — Attention profiles (K-Means++)

K-Means++ (parallel `k-means||` init) on standardised (province × 12-topic)
share vectors, k = 5 (silhouette = **0.166** — modest but positive structure):

| Cluster | n | Dominant topics | Example provinces |
|---|---|---|---|
| 0 | 31 | T6, T10, T9 | Ankara, Balıkesir, Edirne, Elazığ, Afyon… |
| 1 | 15 | T4, T10, T9 | Hakkari, Bitlis, Kars, Ardahan, Artvin, Kocaeli… |
| 2 | 19 | T10, T5, T4 | Diyarbakır, Adana, Bursa, Aydın, Ağrı… |
| 3 | 9 | T8, T10, T6 | Konya, Mersin, Denizli, Batman, Karaman… |
| 4 | 7 | T7, T10, T5 | Antalya, Trabzon, Nevşehir, Çorum, Aksaray… |

Clusters capture distinct discourse signatures (e.g. cluster 1 = small eastern
provinces with a disaster/service profile; cluster 4 = tourism/agriculture
provinces with a projects profile). Figure: `rq2_s3_clusters.png`.

## S4 — Attention vs population and political representation

| Pair | Pearson | Spearman |
|---|---|---|
| attention ~ population | **0.517** | 0.525 |
| attention ~ MP count | 0.515 | 0.516 |
| population ~ MP count | 0.999 | — |

Two clear results:

1. **Population only moderately explains attention (r ≈ 0.52, ~27% of variance).**
   Half the variation in provincial attention is *not* about size — it is
   political targeting.
2. **MP count is almost perfectly collinear with population (r = 0.999)** because
   parliamentary seats are allocated by population; it adds no independent
   signal beyond population.

Normalising by population (mentions per 100k) makes the targeting explicit
(figure `rq2_s4_correlation.png`):

- **Over-attended (per capita):** Tunceli (199.7), Hakkari (191.9), Burdur
  (188.9), Artvin (185.2), Bitlis (167.5), Kars, Ardahan, Bingöl — small
  eastern/Black-Sea provinces.
- **Under-attended (per capita):** İstanbul (6.0), Samsun (5.9), Balıkesir,
  Bursa, Manisa, Konya — the large western metros.

Parliamentary attention in the 28th term is **strongly skewed toward small
eastern provinces and away from large western metros on a per-capita basis**,
consistent with the opposition (especially DEM) agenda surfaced in RQ1.

---

### Limitations

- Attention = province *mentions in the summary*; a question with no province in
  its one-line summary (≈ 53%) contributes no signal even if its full text is
  local.
- Population is TÜİK 2022 (via Wikipedia, `data/reference/province_population.json`);
  GDP was not included — population + MP count already cover "size" and
  "representation", and the two are collinear.
- Topic clusters inherit the coarseness of summary-level LDA (S2 caveat).
