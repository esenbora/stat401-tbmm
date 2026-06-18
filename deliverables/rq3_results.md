# RQ3 — MP Co-attention Network (Results, Spark-first)

What is the structure of coordination among TBMM 28th-term MPs, and do PageRank
influence and community detection align with party boundaries? Run on the same
Spark/Delta stack as RQ1/RQ2.

## Methodology — why co-attention, not co-signing

Written questions are single-authored, so there is no literal co-signing. The
earlier version linked MPs who filed an *identical summary* — but that measures
the **same text-duplication signal as RQ1's MinHash near-duplicates**, so the two
research questions overlapped. We therefore redefine an edge as:

> two MPs are linked if they filed questions targeting the **same ministry**,
> about the **same province**, in the **same month** — coordination of
> *attention*, independent of wording.

Edge weight = number of shared (ministry, province, month) buckets. Built in
Spark from the Silver table (provinces from full-text mentions); **304 MPs,
14,908 weighted edges** (denser than the old co-signing graph because attention
overlaps more widely than exact wording). The network is opposition-dominated
(AK Parti barely files written questions).

| Step | Engine |
|---|---|
| Graph build, PageRank, Label Propagation | **Spark** (DataFrame message-passing) |
| Louvain, betweenness bridges, DeepWalk+UMAP | networkx / gensim (no Spark equivalent) |

## S1 — Central hubs (Spark PageRank)

Ömer Faruk Gergerlioğlu (DEM) remains the top hub (PR ≈ 0.045) — he attends to
the widest range of ministry/province/month issues — followed by other DEM MPs
and Mustafa Bilici (CHP). Centrality is no longer driven by mass-campaign clique
inflation (that was the co-signing artifact); here it reflects breadth of
attention.

## S2 — Communities vs party (key finding)

- **Spark Label Propagation** is degenerate on this dense graph (collapses to one
  community, ARI = 0) — the known LPA weakness; reported honestly.
- **Louvain** finds 5 communities with **weak party alignment: ARI = 0.134,
  NMI = 0.228** — *much lower than the co-signing graph's 0.46*. The largest
  communities are party-mixed (CHP 59%, CHP 67%, CHP 57%), with **one cohesive
  DEM community (58 MPs, 83% DEM)**.

**Interpretation:** when coordination is defined by *what MPs pay attention to*
(ministry × province × month) rather than *how they word it*, party boundaries
blur sharply — MPs across parties attend to the same issues at the same time. The
only group that stays internally cohesive on attention is **DEM Parti**. So
communities mirror party lines only weakly, except for an isolated DEM bloc.

## S3 — Embeddings (DeepWalk + UMAP)

Weighted DeepWalk + UMAP; party silhouette = **−0.26 (negative)** — party labels
do not form clean clusters in attention space, consistent with the low Louvain
ARI. (Weighted DeepWalk, not Node2Vec — there is no 2nd-order p/q bias.)

## S4 — Cross-party bridges

By high betweenness × low intra-party closeness, the top brokers are
**Mustafa Bilici (CHP)**, Mehmet Emin Ekmen (CHP), Cem Avşar (CHP) and Aykut Kaya
(İYİ Parti) — MPs whose attention spans the party-mixed attention communities.
(Unlike the co-signing graph, where Sezgin Tanrıkulu dominated the wording
bridge, the attention bridges are a broader set.)

---

## What changed vs the original analysis

1. **Edges redefined** as ministry×province×month co-attention → removes the
   RQ1/RQ3 overlap (wording vs attention are now distinct signals).
2. Party–community alignment drops 0.46 → **0.13**: attention coordination is far
   more cross-party than wording coordination.
3. PageRank no longer inflated by mass-campaign cliques.
4. Honest reporting kept: LPA degeneracy, negative embedding silhouette,
   opposition-only network, seeds fixed (seed = 42).
5. "Node2Vec" correctly labelled weighted DeepWalk; HITS dropped (degenerate on
   undirected graphs).
