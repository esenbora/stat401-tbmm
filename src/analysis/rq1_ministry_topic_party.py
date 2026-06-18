"""RQ1 — Ministry x Topic x Party x Time (Spark MLlib).

Answers the four approved sub-questions:

  S1. Which ministries receive the most written questions, and how does the
      ranking shift across legislative years?
  S2. What are the dominant latent topics (LDA) and term co-occurrences
      (FP-Growth) directed at each ministry?
  S3. How do parties differ in topical/ministerial focus, and can a question's
      party be predicted from its text? (multinomial logistic regression)
  S4. Do near-duplicate questions (MinHash/LSH, Jaccard >= 0.8) exist across
      MPs / parties, indicating coordinated agenda-setting?

Spark MLlib used: RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, LDA,
FP-Growth, LogisticRegression, MinHashLSH.

Outputs: Delta gold tables under data/gold, figures under deliverables/figures,
and a results report + metrics JSON under deliverables/.

Run:  .venv/bin/python src/analysis/rq1_ministry_topic_party.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, MinHashLSH, StringIndexer,
)
from pyspark.ml.clustering import LDA
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from spark_utils import get_spark, write_delta, attach_sources, SILVER, GOLD
from analysis._common import TURKISH_STOPWORDS
from analysis._text import tokenize as _tokenize

FIG = ROOT / "deliverables" / "figures"
DELIV = ROOT / "deliverables"
N_TOPICS = 12
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def short_ministry(name: str) -> str:
    """Trim 'X Bakanlığı' -> 'X' for readable axis labels."""
    return (name or "?").replace(" Bakanlığı", "").replace(" Ve ", "/").strip()


# ==========================================================================
def sub1_ministry_time(df, metrics: dict) -> None:
    print("\n=== S1: ministry volume x legislative year ===")
    by_min = (
        df.groupBy("bakanlik").agg(F.count("*").alias("n"))
        .orderBy(F.desc("n"))
    )
    pdf = by_min.toPandas()
    pdf["label"] = pdf["bakanlik"].map(short_ministry)
    metrics["top_ministries"] = pdf.head(10)[["bakanlik", "n"]].to_dict("records")

    # ministry x year matrix for ranking shift
    grid = (
        df.groupBy("bakanlik", "yil").agg(F.count("*").alias("n"))
        .where(F.col("yil").isNotNull())
    ).toPandas()
    write_delta(
        df.groupBy("bakanlik", "yil").agg(F.count("*").alias("n")),
        GOLD / "ministry_year",
    )
    pivot = grid.pivot(index="bakanlik", columns="yil", values="n").fillna(0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).head(12).drop(columns="total")
    pivot.index = [short_ministry(i) for i in pivot.index]

    # figure: top ministries bar + per-year rank heatmap
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    top = pdf.head(12).iloc[::-1]
    ax[0].barh(top["label"], top["n"], color="#3b6ea5")
    ax[0].set_title("Bakanlığa göre toplam yazılı soru (Top 12)")
    ax[0].set_xlabel("Soru sayısı")
    rank = pivot.rank(ascending=False, axis=0)  # rank within each year
    im = ax[1].imshow(rank.values, cmap="RdYlGn_r", aspect="auto")
    ax[1].set_xticks(range(len(pivot.columns)))
    ax[1].set_xticklabels([int(c) for c in pivot.columns])
    ax[1].set_yticks(range(len(pivot.index)))
    ax[1].set_yticklabels(pivot.index)
    ax[1].set_title("Yıllara göre bakanlık sıralaması (1 = en çok)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax[1].text(j, i, int(rank.values[i, j]), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax[1], label="sıra")
    plt.tight_layout()
    plt.savefig(FIG / "rq1_s1_ministry_time.png")
    plt.close()
    print(pdf.head(8)[["label", "n"]].to_string(index=False))


# ==========================================================================
def sub2_topics(df, spark, metrics: dict):
    print("\n=== S2: LDA topics + FP-Growth co-occurrence ===")
    toks = _tokenize(df, in_col="text").where(F.size("toks") > 0).cache()

    cv = CountVectorizer(inputCol="toks", outputCol="tf", vocabSize=2000, minDF=10.0)
    cv_model = cv.fit(toks)
    vocab = cv_model.vocabulary
    feat = cv_model.transform(toks)

    lda = LDA(k=N_TOPICS, maxIter=30, featuresCol="tf", seed=42)
    lda_model = lda.fit(feat)
    metrics["lda_perplexity"] = float(lda_model.logPerplexity(feat))

    topics = lda_model.describeTopics(8)
    topic_words = []
    for row in topics.collect():
        words = [vocab[i] for i in row["termIndices"]]
        topic_words.append({"topic": int(row["topic"]), "words": words})
    metrics["lda_topics"] = topic_words
    print("LDA topics (top words):")
    for t in topic_words:
        print(f"  T{t['topic']:02d}: {', '.join(t['words'])}")

    # per-doc dominant topic -> ministry x topic
    transformed = lda_model.transform(feat)
    to_dom = F.udf(lambda v: int(np.argmax(v.toArray())), "int")
    transformed = transformed.withColumn("dom_topic", to_dom(F.col("topicDistribution")))
    mt = (
        transformed.groupBy("bakanlik", "dom_topic").agg(F.count("*").alias("n"))
    )
    write_delta(mt, GOLD / "ministry_topic")

    # ministry x topic heatmap (row-normalised)
    mt_pd = mt.toPandas()
    piv = mt_pd.pivot(index="bakanlik", columns="dom_topic", values="n").fillna(0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).head(12).index]
    piv.index = [short_ministry(i) for i in piv.index]
    norm = piv.div(piv.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(norm.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(N_TOPICS)); ax.set_xticklabels([f"T{i}" for i in range(N_TOPICS)])
    ax.set_yticks(range(len(norm.index))); ax.set_yticklabels(norm.index)
    ax.set_title("Bakanlık × LDA konu dağılımı (satır-normalize)")
    plt.colorbar(im, ax=ax, label="oran")
    plt.tight_layout(); plt.savefig(FIG / "rq1_s2_ministry_topic.png"); plt.close()

    # FP-Growth: co-occurring terms. Full-text transactions are long (~200
    # tokens) so we (a) restrict items to the CountVectorizer vocabulary (top
    # terms) to bound transaction length and (b) use a higher minSupport — both
    # avoid the combinatorial blow-up that OOMs the FP-tree on long documents.
    vocab_set = set(vocab)
    keep_vocab = F.udf(lambda arr: [t for t in set(arr) if t in vocab_set],
                       T.ArrayType(T.StringType()))
    tx = (toks.withColumn("items", keep_vocab(F.col("toks")))
          .where(F.size("items") >= 2).select("items"))
    fp = FPGrowth(itemsCol="items", minSupport=0.03, minConfidence=0.4)
    fp_model = fp.fit(tx)
    freq = fp_model.freqItemsets.where(F.size("items") >= 2).orderBy(F.desc("freq"))
    rules = fp_model.associationRules.orderBy(F.desc("confidence"))
    top_sets = [
        {"items": r["items"], "freq": int(r["freq"])}
        for r in freq.limit(15).collect()
    ]
    top_rules = [
        {"antecedent": r["antecedent"], "consequent": r["consequent"],
         "confidence": round(float(r["confidence"]), 3), "lift": round(float(r["lift"]), 2)}
        for r in rules.limit(15).collect()
    ]
    metrics["fp_top_itemsets"] = top_sets
    metrics["fp_top_rules"] = top_rules
    print("\nFP-Growth frequent term-pairs:")
    for s in top_sets[:8]:
        print(f"  {s['items']}  (freq={s['freq']})")

    toks.unpersist()
    return cv_model  # reuse vocab for S4 not needed; returned for clarity


# ==========================================================================
def sub3_party_focus(df, metrics: dict):
    print("\n=== S3: party focus + predict party from text ===")
    # party x ministry focus (row-normalised share)
    pm = (
        df.groupBy("party", "bakanlik").agg(F.count("*").alias("n"))
    )
    write_delta(pm, GOLD / "party_ministry")
    pm_pd = pm.toPandas()
    main_parties = ["CHP", "DEM Parti", "İYİ Parti", "MHP", "AK Parti"]
    piv = (pm_pd[pm_pd.party.isin(main_parties)]
           .pivot(index="party", columns="bakanlik", values="n").fillna(0))
    piv = piv[piv.sum().sort_values(ascending=False).head(12).index]
    piv.columns = [short_ministry(c) for c in piv.columns]
    norm = piv.div(piv.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(norm.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(norm.columns))); ax.set_xticklabels(norm.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(norm.index))); ax.set_yticklabels(norm.index)
    ax.set_title("Parti × Bakanlık odağı (satır-normalize pay)")
    plt.colorbar(im, ax=ax, label="pay")
    plt.tight_layout(); plt.savefig(FIG / "rq1_s3_party_ministry.png"); plt.close()

    # classifier: predict party from question text (TF-IDF -> multinomial LR)
    clf_parties = ["CHP", "DEM Parti", "İYİ Parti"]  # enough volume to learn
    cdf = _tokenize(df.where(F.col("party").isin(clf_parties)), in_col="text")
    cdf = cdf.where(F.size("toks") > 0)
    cv = CountVectorizer(inputCol="toks", outputCol="tf", vocabSize=5000, minDF=5.0).fit(cdf)
    idf = IDF(inputCol="tf", outputCol="features")
    feat = idf.fit(cv.transform(cdf)).transform(cv.transform(cdf))
    lab = StringIndexer(inputCol="party", outputCol="label").fit(feat)
    feat = lab.transform(feat)
    train, test = feat.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50,
                            regParam=0.05, elasticNetParam=0.0)
    model = lr.fit(train)
    pred = model.transform(test)
    acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(pred)
    f1 = MulticlassClassificationEvaluator(metricName="f1").evaluate(pred)
    metrics["party_clf"] = {
        "classes": clf_parties, "accuracy": round(acc, 4), "f1": round(f1, 4),
        "baseline_majority": round(
            cdf.groupBy("party").count().toPandas()["count"].max()
            / cdf.count(), 4),
    }
    print(f"Party prediction: acc={acc:.3f} f1={f1:.3f} "
          f"(majority baseline={metrics['party_clf']['baseline_majority']})")

    # confusion matrix
    cm = (pred.groupBy("label", "prediction").count().toPandas())
    labels = lab.labels
    M = np.zeros((len(labels), len(labels)))
    for _, r in cm.iterrows():
        M[int(r["label"]), int(r["prediction"])] = r["count"]
    Mn = M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title(f"Parti tahmini — confusion (acc={acc:.2f})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                    color="white" if Mn[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(FIG / "rq1_s3_confusion.png"); plt.close()


# ==========================================================================
def sub4_duplicates(df, metrics: dict):
    print("\n=== S4: MinHash/LSH near-duplicate detection ===")
    toks = _tokenize(df.select("guid", "mv", "party", "text"), in_col="text")
    toks = toks.withColumn("items", F.array_distinct("toks")).where(F.size("items") >= 3)
    cv = CountVectorizer(inputCol="items", outputCol="vec", vocabSize=8000, minDF=2.0).fit(toks)
    # A question whose tokens are all rare (below minDF) becomes an all-zero
    # vector, which MinHashLSH rejects. Materialise an explicit non-zero filter
    # so the self-join only ever sees valid vectors.
    nnz = F.udf(lambda v: int(v.numNonzeros()), "int")
    vec = (cv.transform(toks)
           .select("guid", "mv", "party", "vec")
           .withColumn("nnz", nnz(F.col("vec")))
           .where(F.col("nnz") > 0)
           .persist())
    print(f"  Non-zero vectors for LSH: {vec.count()}")
    mh = MinHashLSH(inputCol="vec", outputCol="hashes", numHashTables=5, seed=42)
    mh_model = mh.fit(vec)
    # Jaccard >= 0.8  <=>  distance <= 0.2
    joined = mh_model.approxSimilarityJoin(vec, vec, 0.2, distCol="dist") \
        .where(F.col("datasetA.guid") < F.col("datasetB.guid"))
    joined = joined.select(
        F.col("datasetA.guid").alias("a"), F.col("datasetB.guid").alias("b"),
        F.col("datasetA.mv").alias("mv_a"), F.col("datasetB.mv").alias("mv_b"),
        F.col("datasetA.party").alias("party_a"), F.col("datasetB.party").alias("party_b"),
        (1 - F.col("dist")).alias("jaccard"),
    ).cache()
    n_pairs = joined.count()
    cross_mp = joined.where(F.col("mv_a") != F.col("mv_b")).count()
    cross_party = joined.where(F.col("party_a") != F.col("party_b")).count()
    write_delta(joined, GOLD / "duplicate_pairs")
    metrics["duplicates"] = {
        "near_dup_pairs": n_pairs, "cross_mp_pairs": cross_mp,
        "cross_party_pairs": cross_party,
    }
    print(f"Near-dup pairs (Jaccard>=0.8): {n_pairs} | cross-MP: {cross_mp} | cross-party: {cross_party}")
    # which party-pairs coordinate most
    cp = (joined.where(F.col("party_a") != F.col("party_b"))
          .groupBy("party_a", "party_b").count().orderBy(F.desc("count")))
    metrics["duplicates"]["top_cross_party"] = [
        {"a": r["party_a"], "b": r["party_b"], "n": r["count"]}
        for r in cp.limit(8).collect()
    ]
    joined.unpersist()


# ==========================================================================
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    spark = get_spark("tbmm-rq1", memory="8g")
    attach_sources(spark)
    df = spark.read.format("delta").load(str(SILVER / "yazili_soru_clean")) \
        .where(F.col("ozet").isNotNull()).cache()
    print(f"Loaded silver: {df.count()} questions")

    metrics: dict = {}
    sub1_ministry_time(df, metrics)
    sub2_topics(df, spark, metrics)
    sub3_party_focus(df, metrics)
    sub4_duplicates(df, metrics)

    (DELIV / "rq1_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved metrics -> {DELIV / 'rq1_metrics.json'}")
    print(f"Figures -> {FIG}")
    spark.stop()


if __name__ == "__main__":
    main()
