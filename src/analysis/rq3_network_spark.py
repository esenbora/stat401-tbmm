"""RQ3 — MP co-signing network (Spark-first re-implementation).

Re-does the 28th-term co-signing network analysis with the same medallion /
Spark stack as RQ1 and RQ2, and corrects the interpretation issues found in the
original `notebooks/rq3_network/analyze_network.py` review:

  * Graph construction, PageRank and community detection (LPA) run in **pure
    Spark** (iterative DataFrame message-passing) — no GraphFrames jar needed.
  * Betweenness centrality and graph embeddings have **no Spark/MLlib
    equivalent**, so they stay in networkx / gensim on the (tiny, 276-node)
    graph collected to the driver. This is stated explicitly, not hidden.
  * Silhouette scores are reported honestly (negative = weak separation, not
    "distinct structure").
  * The network is framed as an **opposition coordination** network (the
    governing AK Parti barely files written questions), and the DEM "dominance"
    is attributed partly to the clique inflation of mass campaigns.
  * All randomised steps are seeded for reproducibility.

Spark used: DataFrame-based weighted PageRank + Label Propagation (LPA).
Sub-questions S3 (embeddings) / S4 (bridges): networkx + gensim, by necessity.

Run:  .venv/bin/python src/analysis/rq3_network_spark.py
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
from pyspark.sql import Window

from spark_utils import get_spark, write_delta, attach_sources, SILVER, GOLD

FIG = ROOT / "deliverables" / "figures"
DELIV = ROOT / "deliverables"
DAMPING = 0.85
PR_ITERS = 25
LPA_ITERS = 15
SEED = 42
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

PARTY_COLORS = {
    "CHP": "#e41a1c", "DEM Parti": "#984ea3", "İYİ Parti": "#377eb8",
    "MHP": "#ff7f00", "AK Parti": "#f7d000", "Yeniden Refah": "#4daf4a",
    "TİP": "#a65628", "HÜDA PAR": "#999999", "Bilinmiyor": "#cccccc",
}


# ==========================================================================
def build_graph(spark):
    """Vertices (MPs + party/province) and undirected weighted co-signing edges.

    Edge (a,b,w): a and b each filed >=w questions sharing an identical summary
    (a proxy for co-signing, since Turkish written questions are single-author).
    """
    df = spark.read.format("delta").load(str(SILVER / "yazili_soru_clean"))
    mp = spark.read.format("delta").load(str(SILVER / "mp_party"))

    camp = (df.groupBy("ozet").agg(F.collect_set("mv").alias("mvs"))
            .withColumn("k", F.size("mvs")).where(F.col("k") > 1))
    pairs = (camp.select(F.explode("mvs").alias("a"), F.col("mvs"))
             .select("a", F.explode("mvs").alias("b")).where(F.col("a") < F.col("b")))
    edges = pairs.groupBy("a", "b").agg(F.count("*").alias("w")).cache()

    verts = (edges.select(F.col("a").alias("id")).union(edges.select(F.col("b").alias("id")))
             .distinct().join(mp, F.col("id") == mp.mv, "left")
             .select("id", "party", "mv_province"))
    return verts.cache(), edges


def directed(edges):
    """Undirected -> both directed orientations (for message-passing)."""
    return (edges.select(F.col("a").alias("src"), F.col("b").alias("dst"), "w")
            .union(edges.select(F.col("b").alias("src"), F.col("a").alias("dst"), "w")))


# ==========================================================================
def spark_pagerank(spark, verts, ed):
    """Weighted PageRank via DataFrame power iteration (pure Spark)."""
    print("\n=== S1: weighted PageRank (Spark DataFrame power iteration) ===")
    N = verts.count()
    out_w = ed.groupBy("src").agg(F.sum("w").alias("out_w"))
    pr = verts.select(F.col("id").alias("node")).withColumn("pr", F.lit(1.0 / N))

    for i in range(PR_ITERS):
        contrib = (ed.join(pr, ed.src == pr.node).join(out_w, "src")
                   .select(F.col("dst").alias("node"),
                           (F.col("pr") * F.col("w") / F.col("out_w")).alias("c")))
        incoming = contrib.groupBy("node").agg(F.sum("c").alias("inc"))
        pr = (verts.select(F.col("id").alias("node"))
              .join(incoming, "node", "left").fillna({"inc": 0.0})
              .select("node", ((1 - DAMPING) / N + DAMPING * F.col("inc")).alias("pr")))
    pr = pr.cache()

    top = (pr.join(verts, pr.node == verts.id)
           .select("node", "party", "mv_province", "pr")
           .orderBy(F.desc("pr")))
    write_delta(top, GOLD / "mp_pagerank")
    top_pd = top.limit(10).toPandas()
    print(top_pd[["node", "party", "pr"]].to_string(index=False))
    return pr, top_pd


def spark_lpa(spark, verts, ed):
    """Weighted Label Propagation community detection (pure Spark)."""
    print("\n=== S2: community detection — Label Propagation (Spark) ===")
    labels = verts.select(F.col("id").alias("node"), F.col("id").alias("label"))
    w = Window.partitionBy("node").orderBy(F.desc("sw"), F.asc("nlabel"))
    for i in range(LPA_ITERS):
        msg = (ed.join(labels, ed.dst == labels.node)
               .select(ed.src.alias("node"), F.col("label").alias("nlabel"), "w"))
        agg = msg.groupBy("node", "nlabel").agg(F.sum("w").alias("sw"))
        labels = (agg.withColumn("rn", F.row_number().over(w)).where("rn = 1")
                  .select("node", F.col("nlabel").alias("label")))
    labels = labels.cache()

    res = labels.join(verts, labels.node == verts.id).select("node", "party", "label")
    write_delta(res, GOLD / "mp_community")
    n_comm = res.select("label").distinct().count()
    print(f"LPA communities: {n_comm}")

    pdf = res.toPandas()
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    valid = pdf[pdf.party != "Bilinmiyor"]
    ari = adjusted_rand_score(valid.party, valid.label)
    nmi = normalized_mutual_info_score(valid.party, valid.label)
    # biggest communities party composition
    sizes = pdf.label.value_counts()
    comp = []
    for lab in sizes.head(4).index:
        sub = pdf[pdf.label == lab]
        dist = sub.party.value_counts(normalize=True)
        comp.append({"size": int(len(sub)),
                     "top_party": dist.index[0], "share": round(float(dist.iloc[0]), 3)})
    print(f"ARI vs party: {ari:.4f} | NMI: {nmi:.4f}")
    for c in comp:
        print(f"  community size {c['size']}: {c['top_party']} {c['share']*100:.1f}%")
    return pdf, {"n_communities": int(n_comm), "ari": round(float(ari), 4),
                 "nmi": round(float(nmi), 4), "top_communities": comp}


# ==========================================================================
def networkx_bridges_embeddings(edges_pd, verts_pd, pr_pd, metrics):
    """Betweenness bridges + DeepWalk embeddings — NO Spark/MLlib equivalent.

    These run on the driver because Spark MLlib has neither exact betweenness
    centrality nor node embeddings; the graph is only 276 nodes so this is cheap.
    """
    import networkx as nx
    import random
    print("\n=== S3/S4: bridges (betweenness) + embeddings — networkx/gensim ===")
    random.seed(SEED); np.random.seed(SEED)

    G = nx.Graph()
    party = dict(zip(verts_pd.id, verts_pd.party))
    prov = dict(zip(verts_pd.id, verts_pd.mv_province))
    for _, r in edges_pd.iterrows():
        G.add_edge(r["a"], r["b"], weight=int(r["w"]))

    # ---- S2b: Louvain (no Spark equivalent; Spark LPA is degenerate here) ----
    # On this clique-heavy graph, Spark Label Propagation collapses to ~1 giant
    # community. Louvain (modularity optimisation) recovers the real structure;
    # report it alongside LPA as the substantive community result.
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    communities = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    node_comm = {n: i for i, com in enumerate(communities) for n in com}
    valid_nodes = [n for n in G.nodes() if party.get(n, "Bilinmiyor") != "Bilinmiyor"]
    lv_ari = adjusted_rand_score([party[n] for n in valid_nodes],
                                 [node_comm[n] for n in valid_nodes])
    lv_nmi = normalized_mutual_info_score([party[n] for n in valid_nodes],
                                          [node_comm[n] for n in valid_nodes])
    comp = []
    for com in sorted(communities, key=len, reverse=True)[:4]:
        dist = pd.Series([party.get(n, "Bilinmiyor") for n in com]).value_counts(normalize=True)
        comp.append({"size": len(com), "top_party": dist.index[0],
                     "share": round(float(dist.iloc[0]), 3)})
    metrics["louvain"] = {"n_communities": len(communities),
                          "ari": round(float(lv_ari), 4), "nmi": round(float(lv_nmi), 4),
                          "top_communities": comp}
    print(f"Louvain: {len(communities)} communities, ARI={lv_ari:.4f} NMI={lv_nmi:.4f}")
    for c in comp:
        print(f"  size {c['size']}: {c['top_party']} {c['share']*100:.1f}%")

    # ---- S4: bridges (high betweenness, low intra-party closeness) ----------
    nx.set_edge_attributes(G, {(u, v): 1.0 / d["weight"] for u, v, d in G.edges(data=True)}, "dist")
    btw = nx.betweenness_centrality(G, weight="dist", seed=SEED)
    apsp = dict(nx.all_pairs_dijkstra_path_length(G, weight="dist"))
    intra = {}
    for u in G.nodes():
        same = [v for v in G.nodes() if party.get(v) == party.get(u) and v != u]
        if not same:
            intra[u] = 0.0; continue
        dsum = sum(apsp[u].get(v, 999.0) for v in same)
        intra[u] = len(same) / dsum if dsum else 0.0
    nodes = list(G.nodes())
    from sklearn.preprocessing import minmax_scale
    nb = dict(zip(nodes, minmax_scale([btw[n] for n in nodes])))
    nc = dict(zip(nodes, minmax_scale([intra[n] for n in nodes])))
    bridge = {n: nb[n] * (1 - nc[n]) for n in nodes}
    top_bridges = sorted(bridge.items(), key=lambda x: x[1], reverse=True)[:10]
    metrics["top_bridges"] = [
        {"mp": n, "party": party.get(n), "province": prov.get(n),
         "bridge_score": round(float(s), 4), "betweenness": round(float(btw[n]), 4)}
        for n, s in top_bridges]
    print("Top bridges:")
    for n, s in top_bridges[:5]:
        print(f"  {n} ({party.get(n)}): bridge={s:.4f} btw={btw[n]:.4f}")

    # ---- S3: weighted DeepWalk embeddings (NOT Node2Vec: no p/q 2nd-order) ---
    from gensim.models import Word2Vec
    import umap
    from sklearn.metrics import silhouette_score

    def walks(graph, num=20, length=40):
        out = []
        nn = list(graph.nodes())
        for _ in range(num):
            random.shuffle(nn)
            for node in nn:
                walk = [node]
                while len(walk) < length:
                    nbrs = list(graph.neighbors(walk[-1]))
                    if not nbrs:
                        break
                    ws = np.array([graph[walk[-1]][x]["weight"] for x in nbrs], dtype=float)
                    walk.append(nbrs[int(np.random.choice(len(nbrs), p=ws / ws.sum()))])
                out.append([str(x) for x in walk])
        return out

    w2v = Word2Vec(walks(G), vector_size=64, window=10, min_count=1, sg=1,
                   workers=1, epochs=15, seed=SEED)
    order = [n for n in G.nodes() if str(n) in w2v.wv]
    emb = np.array([w2v.wv[str(n)] for n in order])
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(emb)
    parties = [party.get(n, "Bilinmiyor") for n in order]
    sil_party = float(silhouette_score(coords, parties))
    metrics["silhouette_party"] = round(sil_party, 4)
    print(f"UMAP silhouette by party: {sil_party:.4f} "
          f"({'weak/negative — labels do NOT form clean clusters' if sil_party < 0.25 else 'moderate'})")

    # embedding figure (honest caption)
    fig, ax = plt.subplots(figsize=(10, 8))
    for p in sorted(set(parties)):
        idx = [i for i, q in enumerate(parties) if q == p]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=55, alpha=0.8,
                   color=PARTY_COLORS.get(p, "#777"), label=p)
    ax.set_title(f"DeepWalk + UMAP — MP gömme (parti)\n"
                 f"silhouette={sil_party:.3f} → ayrışma zayıf, parti göreceli baskın")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "rq3_embedding_party.png"); plt.close()
    return top_bridges


def figures(pr_pd, comm_metrics, metrics):
    # PageRank top-10 bar coloured by party
    fig, ax = plt.subplots(figsize=(9, 6))
    d = pr_pd.iloc[::-1]
    ax.barh(d["node"], d["pr"], color=[PARTY_COLORS.get(p, "#777") for p in d["party"]])
    ax.set_title("PageRank Top-10 (Spark) — DEM klik baskınlığı")
    ax.set_xlabel("PageRank")
    plt.tight_layout(); plt.savefig(FIG / "rq3_pagerank_top.png"); plt.close()


# ==========================================================================
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    spark = get_spark("tbmm-rq3", memory="6g")
    attach_sources(spark)

    verts, edges = build_graph(spark)
    ed = directed(edges).cache()
    n_v, n_e = verts.count(), edges.count()
    print(f"Graph: {n_v} vertices, {n_e} undirected edges")

    metrics = {"n_vertices": int(n_v), "n_edges": int(n_e)}
    pr, pr_pd = spark_pagerank(spark, verts, ed)
    comm_pdf, comm_metrics = spark_lpa(spark, verts, ed)
    metrics["community"] = comm_metrics

    verts_pd = verts.toPandas()
    edges_pd = edges.toPandas()
    pr_full = pr.toPandas()
    networkx_bridges_embeddings(edges_pd, verts_pd, pr_full, metrics)
    figures(pr_pd, comm_metrics, metrics)

    (DELIV / "rq3_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved metrics -> {DELIV / 'rq3_metrics.json'}")
    spark.stop()


if __name__ == "__main__":
    main()
