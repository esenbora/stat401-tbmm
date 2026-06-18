"""RQ2 — Provincial attention map (Spark MLlib).

Answers the four approved sub-questions:

  S1. Geographic distribution of province *mentions* across the written
      questions (which provinces draw the most parliamentary attention).
  S2. Which topics dominate each province, and how metropolitan vs. rural
      provinces differ (LDA dominant topic x province).
  S3. K-Means++ clustering of (province x topic) frequency vectors into latent
      "attention profiles".
  S4. How provincial attention correlates with population and political
      representation (number of MPs).  (Pearson + Spearman)

Spark MLlib used: RegexTokenizer, StopWordsRemover, CountVectorizer, LDA,
KMeans (k-means|| init = parallel k-means++), Correlation.

Outputs: Delta gold tables under data/gold, figures under deliverables/figures,
results + metrics JSON under deliverables/.

Run:  .venv/bin/python src/analysis/rq2_province_attention.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizer, VectorAssembler, StandardScaler
from pyspark.ml.clustering import LDA, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors, VectorUDT

from spark_utils import get_spark, write_delta, attach_sources, SILVER, GOLD
from analysis._common import normalize_province, PROVINCES_81
from analysis._text import tokenize

FIG = ROOT / "deliverables" / "figures"
DELIV = ROOT / "deliverables"
REF = ROOT / "data" / "reference"
N_TOPICS = 12
N_CLUSTERS = 5
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

# Metropolitan provinces (TÜİK: büyükşehir + largest populations) for the
# metro-vs-rural contrast in S2.
METRO = {"İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya",
         "Gaziantep", "Şanlıurfa", "Kocaeli", "Mersin", "Diyarbakır", "Kayseri",
         "Samsun"}


def mp_counts_by_province() -> dict[str, int]:
    """Number of TBMM 28th-term MPs elected per province (Wikipedia parse)."""
    mps = json.loads((REF / "tbmm28_mps_wikipedia.json").read_text(encoding="utf-8"))
    c = Counter(normalize_province(m.get("province")) for m in mps)
    c.pop("Bilinmiyor", None)
    return dict(c)


def population_by_province() -> dict[str, int]:
    raw = json.loads((REF / "province_population.json").read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ==========================================================================
def sub1_attention(df, metrics: dict):
    print("\n=== S1: province mention distribution ===")
    exploded = (df.select(F.explode("mentioned_provinces").alias("il"))
                .groupBy("il").agg(F.count("*").alias("mentions"))
                .orderBy(F.desc("mentions")))
    write_delta(exploded, GOLD / "province_mentions")
    pdf = exploded.toPandas()
    metrics["total_mentions"] = int(pdf["mentions"].sum())
    metrics["provinces_mentioned"] = int(len(pdf))
    metrics["top_provinces"] = pdf.head(15).to_dict("records")
    print(pdf.head(12).to_string(index=False))

    top = pdf.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(top["il"], top["mentions"], color="#2a7f62")
    ax.set_title("İl bazlı parlamento ilgisi — yazılı soruda anılma (Top 25)")
    ax.set_xlabel("Anılma sayısı")
    plt.tight_layout(); plt.savefig(FIG / "rq2_s1_province_mentions.png"); plt.close()
    return pdf


def build_province_topic(df, spark):
    """LDA dominant topic per question, exploded over mentioned provinces ->
    (province x topic) count matrix. Returns (matrix_pdf, topic_words)."""
    toks = tokenize(df.select("guid", "mentioned_provinces", "text"), in_col="text") \
        .where(F.size("toks") > 0)
    cv = CountVectorizer(inputCol="toks", outputCol="tf", vocabSize=2000, minDF=10.0).fit(toks)
    vocab = cv.vocabulary
    feat = cv.transform(toks)
    lda = LDA(k=N_TOPICS, maxIter=30, featuresCol="tf", seed=42)
    lda_model = lda.fit(feat)
    topic_words = [
        {"topic": int(r["topic"]), "words": [vocab[i] for i in r["termIndices"]]}
        for r in lda_model.describeTopics(8).collect()
    ]
    to_dom = F.udf(lambda v: int(np.argmax(v.toArray())), "int")
    doc = lda_model.transform(feat).withColumn("dom_topic", to_dom("topicDistribution"))
    pt = (doc.where(F.size("mentioned_provinces") > 0)
          .select(F.explode("mentioned_provinces").alias("il"), "dom_topic")
          .groupBy("il", "dom_topic").agg(F.count("*").alias("n")))
    write_delta(pt, GOLD / "province_topic")
    return pt.toPandas(), topic_words


def sub2_topics_by_province(pt_pd, topic_words, metrics: dict):
    print("\n=== S2: topics per province + metro vs rural ===")
    metrics["lda_topics"] = topic_words
    piv = pt_pd.pivot(index="il", columns="dom_topic", values="n").fillna(0)
    piv = piv.reindex(columns=range(N_TOPICS), fill_value=0)
    share = piv.div(piv.sum(axis=1), axis=0)

    # metro vs rural average topic profile
    metro_rows = [i for i in share.index if i in METRO]
    rural_rows = [i for i in share.index if i not in METRO]
    metro_prof = share.loc[metro_rows].mean()
    rural_prof = share.loc[rural_rows].mean()
    # Top topics are dominated by generic ones everywhere, so the *difference*
    # in share is the informative contrast: which topics skew metro vs rural.
    diff = (metro_prof - rural_prof).sort_values()
    metrics["metro_vs_rural"] = {
        "metro_top_topics": [int(t) for t in metro_prof.sort_values(ascending=False).head(3).index],
        "rural_top_topics": [int(t) for t in rural_prof.sort_values(ascending=False).head(3).index],
        "metro_skewed_topics": [
            {"topic": int(t), "delta": round(float(diff[t]), 4)} for t in diff.tail(3).index[::-1]],
        "rural_skewed_topics": [
            {"topic": int(t), "delta": round(float(diff[t]), 4)} for t in diff.head(3).index],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_TOPICS); w = 0.4
    ax.bar(x - w / 2, metro_prof.values, w, label="Metropol", color="#c0392b")
    ax.bar(x + w / 2, rural_prof.values, w, label="Kırsal/diğer", color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels([f"T{i}" for i in range(N_TOPICS)])
    ax.set_xlabel("LDA konu"); ax.set_ylabel("Ortalama pay")
    ax.set_title("Metropol vs kırsal il — konu profili"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG / "rq2_s2_metro_rural.png"); plt.close()
    print("Metro-skewed topics:", metrics["metro_vs_rural"]["metro_skewed_topics"])
    print("Rural-skewed topics:", metrics["metro_vs_rural"]["rural_skewed_topics"])
    return share


def sub3_kmeans(spark, share: pd.DataFrame, metrics: dict):
    print("\n=== S3: K-Means++ attention profiles ===")
    provinces = list(share.index)
    rows = [(p, Vectors.dense(share.loc[p].values.tolist())) for p in provinces]
    sdf = spark.createDataFrame(rows, ["il", "features_raw"])
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True).fit(sdf)
    sdf = scaler.transform(sdf)
    km = KMeans(k=N_CLUSTERS, featuresCol="features", seed=42,
                initMode="k-means||", maxIter=50)  # parallel k-means++
    model = km.fit(sdf)
    pred = model.transform(sdf)
    sil = ClusteringEvaluator(featuresCol="features").evaluate(pred)
    metrics["kmeans_silhouette"] = round(float(sil), 4)

    assign = {r["il"]: int(r["prediction"]) for r in pred.select("il", "prediction").collect()}
    clusters: dict[int, list[str]] = {}
    for il, c in assign.items():
        clusters.setdefault(c, []).append(il)

    # characterise each cluster by its dominant topics (mean share)
    cluster_profile = {}
    for c, members in clusters.items():
        prof = share.loc[members].mean()
        cluster_profile[c] = {
            "size": len(members),
            "members": sorted(members),
            "top_topics": [int(t) for t in prof.sort_values(ascending=False).head(3).index],
        }
    metrics["kmeans_clusters"] = cluster_profile
    write_delta(
        spark.createDataFrame([(il, c) for il, c in assign.items()], ["il", "cluster"]),
        GOLD / "province_cluster",
    )
    for c, info in sorted(cluster_profile.items()):
        print(f"  Cluster {c} (n={info['size']}, topics {info['top_topics']}): "
              f"{', '.join(info['members'][:8])}{'...' if info['size'] > 8 else ''}")

    # 2D PCA-ish view via the two highest-variance topics for a quick scatter
    var_topics = share.var().sort_values(ascending=False).head(2).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for c, members in clusters.items():
        xs = share.loc[members, var_topics[0]]
        ys = share.loc[members, var_topics[1]]
        ax.scatter(xs, ys, color=cmap(c), label=f"Küme {c} (n={len(members)})", s=60, alpha=0.8)
        for il in members:
            if il in METRO or share.loc[il].sum() > 0:
                ax.annotate(il, (share.loc[il, var_topics[0]], share.loc[il, var_topics[1]]),
                            fontsize=6, alpha=0.6)
    ax.set_xlabel(f"Konu T{var_topics[0]} payı"); ax.set_ylabel(f"Konu T{var_topics[1]} payı")
    ax.set_title("İl ilgi profilleri — K-Means++ kümeleri")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG / "rq2_s3_clusters.png"); plt.close()
    return assign


def sub4_correlation(spark, attention_pdf: pd.DataFrame, metrics: dict):
    print("\n=== S4: attention vs population & representation ===")
    pop = population_by_province()
    mpc = mp_counts_by_province()
    att = dict(zip(attention_pdf["il"], attention_pdf["mentions"]))

    rows = []
    for prov in PROVINCES_81:
        rows.append({
            "il": prov,
            "attention": att.get(prov, 0),
            "population": pop.get(prov, np.nan),
            "mp_count": mpc.get(prov, 0),
        })
    cdf = pd.DataFrame(rows).dropna(subset=["population"])
    cdf["att_per_100k"] = cdf["attention"] / (cdf["population"] / 100_000)

    # Spark MLlib Correlation on the [attention, population, mp_count] vectors
    sdf = spark.createDataFrame(cdf[["attention", "population", "mp_count"]])
    assembler = VectorAssembler(
        inputCols=["attention", "population", "mp_count"], outputCol="vec")
    vec_df = assembler.transform(sdf)
    pearson = Correlation.corr(vec_df, "vec", "pearson").head()[0].toArray()
    cols = ["attention", "population", "mp_count"]
    metrics["correlation_pearson"] = {
        f"{cols[0]}~{cols[1]}": round(float(pearson[0, 1]), 4),
        f"{cols[0]}~{cols[2]}": round(float(pearson[0, 2]), 4),
        f"{cols[1]}~{cols[2]}": round(float(pearson[1, 2]), 4),
    }
    # Spearman (rank) via scipy — robust to the heavy right skew of population
    sp_pop = spearmanr(cdf["attention"], cdf["population"])
    sp_mp = spearmanr(cdf["attention"], cdf["mp_count"])
    metrics["correlation_spearman"] = {
        "attention~population": round(float(sp_pop.statistic), 4),
        "attention~mp_count": round(float(sp_mp.statistic), 4),
    }
    print("Pearson:", metrics["correlation_pearson"])
    print("Spearman:", metrics["correlation_spearman"])

    # over/under-attended provinces (residual vs population trend)
    cdf = cdf.sort_values("att_per_100k", ascending=False)
    metrics["most_over_attended"] = cdf.head(8)[["il", "attention", "att_per_100k"]].round(1).to_dict("records")
    metrics["most_under_attended"] = (
        cdf[cdf.attention > 50].tail(8)[["il", "attention", "att_per_100k"]].round(1).to_dict("records"))

    write_delta(spark.createDataFrame(cdf), GOLD / "province_correlates")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].scatter(cdf["population"], cdf["attention"], s=40, color="#2a7f62")
    for _, r in cdf.iterrows():
        if r["attention"] > cdf["attention"].quantile(0.9) or r["population"] > 2e6:
            ax[0].annotate(r["il"], (r["population"], r["attention"]), fontsize=7)
    ax[0].set_xlabel("Nüfus (2022)"); ax[0].set_ylabel("Anılma sayısı")
    ax[0].set_title(f"İlgi ~ Nüfus (Pearson r={metrics['correlation_pearson']['attention~population']})")
    top_per = cdf.head(15).iloc[::-1]
    ax[1].barh(top_per["il"], top_per["att_per_100k"], color="#8e44ad")
    ax[1].set_xlabel("100k kişi başına anılma"); ax[1].set_title("En 'aşırı ilgilenilen' iller (nüfusa göre)")
    plt.tight_layout(); plt.savefig(FIG / "rq2_s4_correlation.png"); plt.close()


# ==========================================================================
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    spark = get_spark("tbmm-rq2", memory="6g")
    attach_sources(spark)
    df = spark.read.format("delta").load(str(SILVER / "yazili_soru_clean")) \
        .where(F.col("ozet").isNotNull()).cache()
    print(f"Loaded silver: {df.count()} questions")

    metrics: dict = {}
    attention_pdf = sub1_attention(df, metrics)
    pt_pd, topic_words = build_province_topic(df, spark)
    share = sub2_topics_by_province(pt_pd, topic_words, metrics)
    sub3_kmeans(spark, share, metrics)
    sub4_correlation(spark, attention_pdf, metrics)

    (DELIV / "rq2_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved metrics -> {DELIV / 'rq2_metrics.json'}")
    print(f"Figures -> {FIG}")
    spark.stop()


if __name__ == "__main__":
    main()
