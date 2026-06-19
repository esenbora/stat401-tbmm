# RQ1 — Ministry × Topic × Party × Time (Results)

Computed on the **question body** (`govde` = OCR full text with the party
letterhead/signature stripped) of all 44,484 written questions, with Apache
Spark MLlib (LDA, FP-Growth, CountVectorizer/IDF, multinomial LogisticRegression,
MinHashLSH). Gold tables in `data/gold/{ministry_year, ministry_topic,
party_ministry, duplicate_pairs}`; figures `rq1_*`.

## S1 — Ministry volume & ranking over time

Adalet (Justice) 6,140 · İçişleri (Interior) 5,571 · Çevre/Şehircilik 4,587 ·
Tarım/Orman 4,584 · Sağlık 3,520 · Milli Eğitim 3,508. Top-ministry ranks are
stable across all four legislative years; volume rises 2023→2025 (7,854→16,144).

## S2 — Topics (LDA, k=15) and co-occurrence (FP-Growth)

LDA on `govde` (letterhead removed, Turkish ASCII-folded, `maxDF=0.4`) gives
clean policy themes:

- **Detention / justice** — *cezaevi, infaz, mahpus, ceza, tipi, adalet*
- **Earthquake / energy** — *deprem, Hatay, elektrik, enerji, konut*
- **Environment / mining** — *maden, çevre, belediye, denetim, yol*
- **Agriculture** — *tarım, orman, sulama, hayvan, şap, gıda*
- **Health** — *sağlık, hastane, hasta*; **Education** — *eğitim, okul, milli*
- **Finance** — *maliye, hazine, SGK*; **Child/women/social** — *çocuk, kadın,
  koruma, engelli, soruşturma*

FP-Growth on `govde` surfaces legal-citation templating (*"… inci maddesi
gereğince …"*) — evidence of formulaic drafting; LDA is the thematic lens.

## S3 — Party differences & predicting party from text

A multinomial logistic regression on TF-IDF of `govde` predicts the filing party
(CHP/DEM/İYİ) at **0.957 accuracy** (F1 = 0.956) vs. a 0.488 baseline.

> Run on `govde`, not the raw text: the party **letterhead** ("Cumhuriyet Halk
> Partisi", "CHP" …) is stripped, so the model no longer reads the label off the
> page. The residual accuracy reflects genuine content + writing-style/identity
> signal (MP names in some closing blocks survive). Parties pursue measurably
> distinct agendas. *(The earlier 0.76 "subject-line" figure is not produced by
> the current pipeline and is not cited.)*

## S4 — Near-duplicate / coordinated questions (MinHash/LSH)

MinHashLSH (Jaccard ≥ 0.8) over **`govde` bodies**:

| Measure | Pairs |
|---|---|
| Near-duplicate pairs | **50,812** |
| …across different MPs | 4,608 |
| …across different parties | 138 |

The pair count is much higher than the subject-line run because stripping the
unique MP header makes templated **bodies** comparable — i.e. this measures
genuine content templating. Coordination is overwhelmingly intra-party (138 of
50,812 pairs cross party lines).

---

### Limitations
- Classifier on `govde` (0.957) still carries some identity signal from closing
  signatures that aren't "saygılarımla"-terminated.
- FP-Growth surfaces legal-citation boilerplate; LDA is the topic lens.
- OCR (35% of docs) sometimes drops word spaces (`halklailiskilerbinasi`),
  producing run-together tokens that fall below `minDF` and are dropped — a known
  OCR limitation.
