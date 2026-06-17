# RQ3 — MP Co-signing Network (Results, corrected & Spark-first)

What is the structure of the co-signing network among TBMM 28th-term MPs, and do
PageRank influence and community detection align with party boundaries? This is
a corrected re-run of the original network analysis (`notebooks/rq3_network/`),
moved onto the same Spark/Delta stack as RQ1/RQ2 and with the review's
interpretation errors fixed.

## Methodology & honest tool boundary

Written questions in the TBMM are filed by a **single** MP — there is no literal
co-signing. We therefore use a **proxy**: two MPs are linked if they filed
questions with an *identical summary*; edge weight = number of shared summaries.
Each shared-summary group is a clique, so a mass campaign of *k* MPs adds
C(*k*,2) edges.

| Step | Engine | Why |
|---|---|---|
| Graph build (vertices/edges) | **Spark** | 2,060 campaigns → 7,720 weighted edges, 276 MPs |
| PageRank | **Spark** (DataFrame power iteration) | demonstrable big-data graph algo |
| Community — Label Propagation | **Spark** (DataFrame message-passing) | the Spark-native community method |
| Community — Louvain | networkx | **no Spark/MLlib equivalent**; needed because LPA is degenerate here |
| Betweenness / bridges | networkx | exact betweenness has no Spark equivalent |
| Embeddings (DeepWalk) + UMAP | gensim / umap | node embeddings have no Spark MLlib equivalent |

> **Framing (was missing before):** 96% of written questions come from CHP, DEM
> and İYİ; the governing AK Parti barely uses the instrument, so only **276 of
> ~600** MPs appear in the graph. This is an **opposition coordination network**,
> not the whole parliament — every claim about "party boundaries" excludes the
> governing party by construction.

## S1 — Central hubs (PageRank)

| Rank | MP | Party | PageRank |
|---|---|---|---|
| 1 | Ömer Faruk Gergerlioğlu | DEM Parti | 0.0199 |
| 2 | Beritan Güneş Altın | DEM Parti | 0.0140 |
| 3 | Çiçek Otlu | DEM Parti | 0.0140 |
| … | (8) Mustafa Sezgin Tanrıkulu | CHP | 0.0119 |

DEM MPs dominate the top of PageRank, and Gergerlioğlu (TBMM's most active
human-rights advocate) is the top hub — consistent with reality.

> **Corrected interpretation:** this dominance is **partly a construction
> artifact**, not pure influence. DEM runs mass campaigns where almost the whole
> faction (≈58 MPs) files identical motions; each such campaign injects a ~58-node
> clique (~1,600 edges) that mechanically inflates every member's centrality.
> PageRank here largely measures *"how large were the campaigns you joined."*
> HITS from the original is dropped: on an **undirected** graph hub = authority,
> so it added nothing over PageRank/eigenvector centrality.

## S2 — Communities vs party boundaries

**Spark Label Propagation is degenerate on this graph:** the dense campaign
cliques cause it to collapse into **2** communities (one 272-node blob + the
4-node AK Parti pocket), ARI = 0.03. This is a known LPA failure mode on
clique-heavy graphs and is reported as such.

**Louvain (modularity)** recovers the real structure:

| Community | Size | Dominant party |
|---|---|---|
| Opposition bloc | 213 | CHP 70.4% (+ İYİ, MHP) |
| DEM clique | 59 | **DEM Parti 98.3%** |
| Government pocket | 4 | AK Parti 100% |

- ARI vs party = **0.4613**, NMI = **0.5509** — moderate alignment.
- DEM forms an isolated, near-pure clique; CHP / İYİ / MHP **merge into one
  bloc** rather than separating — they cross-sign on shared municipal/national
  topics. So communities mirror party lines only partially: the real fault line
  is **DEM vs the rest of the opposition**, not party-by-party.

## S3 — Embeddings (DeepWalk + UMAP)

64-dim embeddings from weighted random walks, projected with UMAP.

> **Corrected interpretation:** the original called this "Node2Vec" — it is not.
> There is no 2nd-order p/q bias, so it is **weighted DeepWalk**. And the party
> silhouette is **−0.22 (negative)** — meaning party labels do **not** form clean
> clusters in embedding space; the original claim of "distinct structural
> separation / moderate structure" was wrong. The honest reading: party is the
> *relatively* strongest organising signal (more than geography), but separation
> is weak. Figure: `rq3_embedding_party.png`.

## S4 — Cross-party bridges

Bridge score = high betweenness × low intra-party closeness (broker globally,
peripheral within own party):

| Rank | MP | Party | Province | Bridge |
|---|---|---|---|---|
| 1 | **Mustafa Sezgin Tanrıkulu** | CHP | Diyarbakır | **0.453** |
| 2 | Selçuk Özdağ | CHP/Yeni Yol | Muğla | 0.146 |
| 3 | Ömer Fethi Gürer | CHP | Niğde | 0.134 |

**Tanrıkulu** is the dominant broker (≈3× the next MP) — a CHP MP from
Diyarbakır whose human-rights / Kurdish-rights agenda overlaps heavily with DEM,
placing him between the CHP bloc and the DEM clique. This is the same broker
RQ1's cross-party near-duplicate analysis flagged (DEM↔CHP is the strongest
cross-party coordination).

---

## What changed vs the original analysis

1. **Silhouette:** −0.19/−0.71 now read as *weak/negative* (poor clustering), not
   "distinct/moderate structure."
2. **Opposition-only framing** added (AK Parti absent — 276/600 MPs).
3. **DEM PageRank dominance** reframed as partly a clique-construction artifact.
4. **"Node2Vec" → "weighted DeepWalk"** (no p/q parameters).
5. **HITS dropped** (degenerate on undirected graphs).
6. **Reproducibility:** all randomised steps seeded (seed = 42).
7. **Spark-native** PageRank + LPA; LPA's degeneracy documented and Louvain kept
   for the substantive community result.
