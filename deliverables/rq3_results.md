# RQ3 — MP Co-attention Network (Results, Spark-first)

Do influence and community structure among TBMM 28th-term MPs align with party
boundaries? Run on the same Spark/Delta stack as RQ1/RQ2.

## Methodology — co-attention, not co-signing

Written questions are single-authored, so there is no literal co-signing. The
earlier "identical-summary" definition measured the **same text-duplication
signal as RQ1's MinHash**, so the two RQs overlapped. We redefine an edge as:

> two MPs are linked if they filed questions targeting the **same ministry**,
> about the **same province**, in the **same month** — coordination of
> *attention*, independent of wording.

Edge weight = number of shared (ministry, province, month) buckets. Built in
Spark from the Silver table (provinces from the cleaned `govde` mentions, with
each MP's own province removed). **265 MPs, 6,973 weighted edges.** The network
is opposition-dominated (AK Parti barely files written questions).

| Step | Engine |
|---|---|
| Graph build, PageRank, Label Propagation | **Spark** (DataFrame message-passing) |
| Louvain, betweenness bridges, DeepWalk+UMAP | networkx / gensim (no Spark equivalent) |

## S1 — Central hubs (Spark PageRank)

Ömer Faruk Gergerlioğlu (DEM) is the top hub — he attends to the widest range of
ministry/province/month issues. **Clique caveat (honest):** the bucket edges are
still a *union of cliques* — a popular ministry + busy month is a large clique,
so PageRank still rewards MPs sitting in many/large buckets. The artifact is
**reduced** (it is no longer driven by one templated-text mega-clique as in the
co-signing graph), **not eliminated**.

## S2 — Communities vs party

- **Spark Label Propagation** is degenerate on this dense graph (≈1 community) —
  the known LPA weakness; reported honestly.
- **Louvain:** ARI = **0.325**, NMI = **0.339** — *moderate* party alignment, and
  lower than the old identical-summary graph (0.46). Communities: a 191-MP
  party-mixed bloc (CHP 68%), a cohesive **DEM community (65 MPs, 83%)**, plus
  small pockets.

**Interpretation:** coordination defined by *what MPs attend to* (ministry ×
province × month) aligns with party somewhat less than coordination defined by
*how they word it* (0.33 vs 0.46) — attention crosses party lines more than
wording does, but not dramatically. The DEM bloc stays the most internally
cohesive group on both definitions.

## S3 — Embeddings (DeepWalk + UMAP)

Weighted DeepWalk + UMAP; party silhouette = **−0.26 (negative)** — party labels
do not form clean clusters in attention space, consistent with the moderate
Louvain ARI. (Weighted DeepWalk, not Node2Vec — no 2nd-order p/q bias.)

## S4 — Cross-party bridges

Top brokers (high betweenness × low intra-party closeness): **Mustafa Bilici
(CHP)**, Cem Avşar (CHP), **Mustafa Sezgin Tanrıkulu (CHP)**, Mehmet Emin Ekmen
(CHP) — MPs whose attention spans the party-mixed communities. Tanrıkulu (the
dominant *wording*-bridge in the co-signing graph) remains a top *attention*
bridge too.

---

## What changed vs the original analysis
1. Edges redefined as ministry×province×month **co-attention** → removes the
   RQ1/RQ3 overlap.
2. Built on the cleaned mentions (govde + self-province removed) → 265 nodes /
   6,973 edges; Louvain ARI **0.33** (vs 0.46 wording).
3. PageRank clique inflation **reduced, not eliminated** (stated honestly).
4. LPA degeneracy, negative embedding silhouette, opposition-only scope all kept;
   seeds fixed (42); "Node2Vec"→DeepWalk; HITS dropped.
