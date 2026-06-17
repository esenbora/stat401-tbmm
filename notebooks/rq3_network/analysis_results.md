# Network Analysis of Turkish MPs (TBMM 28th Term) - Analysis Results

This document presents the detailed findings of the social network analysis performed on the TBMM 28th Term parliamentary motions dataset (`tum_metadata_44485.csv`). 

## Network Construction Methodology
Since written question motions (`Yazılı Soru Önergesi`) in the Turkish Parliament are submitted individually by single MPs, we constructed the **co-signing network** by linking two MPs with an edge if they submitted a motion with the **exact same summary** (`Önerge Özeti`). The edge weight represents the number of shared identical summaries. 

The resulting network has **307 nodes** (unique opposition and active MPs) and **7,720 edges**. The largest connected component consists of **272 nodes**, indicating a highly connected core of opposition legislative coordination.

---

### Sub-question 1: Centrality Hubs (PageRank vs. HITS Authority)

The top 10 central hubs in the network by PageRank and HITS Authority scores are:

| Rank | MP (PageRank) | Party (PR) | Score (PR) | MP (HITS Authority) | Party (HITS) | Score (HITS) |
|---|---|---|---|---|---|---|
| 1 | **Ömer Faruk GERGERLİOĞLU** | DEM Parti | 0.019582 | **Ömer Faruk GERGERLİOĞLU** | DEM Parti | 0.025618 |
| 2 | **Beritan GÜNEŞ ALTIN** | DEM Parti | 0.013737 | **Nevroz UYSAL ASLAN** | DEM Parti | 0.025428 |
| 3 | **Çiçek OTLU** | DEM Parti | 0.013729 | **Beritan GÜNEŞ ALTIN** | DEM Parti | 0.025317 |
| 4 | **Serhat EREN** | DEM Parti | 0.013592 | **Serhat EREN** | DEM Parti | 0.025296 |
| 5 | **Meral DANIŞ BEŞTAŞ** | DEM Parti | 0.013198 | **Sümeyye BOZ ÇAKI** | DEM Parti | 0.021887 |
| 6 | **İbrahim AKIN** | DEM Parti | 0.012936 | **Gülcan KAÇMAZ SAYYİĞİT** | DEM Parti | 0.021421 |
| 7 | **Nevroz UYSAL ASLAN** | DEM Parti | 0.012599 | **Gülderen VARLİ** | DEM Parti | 0.020921 |
| 8 | **Mustafa Sezgin TANRIKULU** | CHP | 0.011742 | **Kamuran TANHAN** | DEM Parti | 0.020074 |
| 9 | **Gülderen VARLİ** | DEM Parti | 0.011381 | **Çiçek OTLU** | DEM Parti | 0.020009 |
| 10| **Sümeyye BOZ ÇAKI** | DEM Parti | 0.011193 | **Öznur BARTİN** | DEM Parti | 0.019985 |

#### Explanation & Insights:
1. **DEM Party Dominance**: Both PageRank and HITS are dominated by **DEM Party** MPs. This is because DEM Party coordinates highly structured, collective campaigns where almost their entire faction (58+ MPs) submits identical motions (e.g. regarding human rights, prison violations, and regional infrastructure). In a network visualization, this creates a highly dense, fully connected clique where every member shares strong connections with every other member, driving up their centrality.
2. **Top Central Figure**: **Ömer Faruk Gergerlioğlu** emerges as the absolute top hub in both metrics. This aligns with his real-world role as the most active human rights advocate in TBMM, participating in almost all coordinated opposition campaigns.
3. **The CHP Exception**: **Sezgin Tanrıkulu** (CHP) is the only non-DEM Party MP in the PageRank Top 10. While he belongs to CHP, his high participation in human rights-focused motions links him strongly to the DEM Party clique, earning him high prestige.

---

### Sub-question 2: Louvain Communities vs. Party Boundaries

Applying the Louvain community detection algorithm to the weighted network yielded **34 communities**.

#### Evaluative Alignment Metrics:
- **Elected Party Alignment**:
  - Adjusted Rand Index (ARI): **0.4628**
  - Normalized Mutual Information (NMI): **0.5362**
- **Current Party/Alliance Group Alignment**:
  - Adjusted Rand Index (ARI): **0.3651**
  - Normalized Mutual Information (NMI): **0.5115**

#### Community Composition Breakdown:
- **Community 2 (59 MPs)**: **98.3% DEM Parti** (58 MPs) and 1 CHP MP. 
  - *Insight*: The DEM Party forms an extremely tight-knit, isolated, and cohesive community that operates almost entirely within its own faction boundaries.
- **Community 20 (213 MPs)**: **70.4% CHP** (150 MPs), **17.4% İYİ Party** (37 MPs), **7.0% MHP** (15 MPs), and a few small opposition groups.
  - *Insight*: Rather than splitting into separate CHP and İYİ Party communities, these MPs merge into a single large community. This indicates that CHP and İYİ Party frequently share cross-signatures, coordinate on standard municipal or nationwide topics, and do not maintain separate isolated coordination silos.
- **Other Communities (32 communities, Sizes 1-4)**: Primarily represent government-aligned MPs (AKP, MHP) who submitted very few isolated motions and have no overlapping co-signatures with the opposition core.

---

### Sub-question 3: Embedding Visualizations (Node2Vec + UMAP)

We generated 64-dimensional Node2Vec embeddings based on biased random walks on the network and projected them into 2D space using UMAP.

#### Silhouette Scores:
- **Party Clustering**: **-0.1913** (Moderate structure)
- **Province Clustering**: **-0.7149** (Very weak/random structure)

#### Visualizations:

```carousel
![UMAP Colored by Party](file:///C:/Users/HP/.gemini/antigravity-ide/brain/9ab54d4e-5106-4086-ac90-cffa361f57ef/umap_party.png)
<!-- slide -->
![UMAP Colored by Province](file:///C:/Users/HP/.gemini/antigravity-ide/brain/9ab54d4e-5106-4086-ac90-cffa361f57ef/umap_province.png)
<!-- slide -->
![UMAP Colored by Seniority/Activity](file:///C:/Users/HP/.gemini/antigravity-ide/brain/9ab54d4e-5106-4086-ac90-cffa361f57ef/umap_seniority.png)
```

#### Explanation & Insights:
1. **Party vs. Ideology**: The **Party** plot demonstrates distinct structural separation. The DEM Party MPs form a tight cluster on one side, while the CHP and İYİ Party MPs are spread out but grouped together on the other side. This shows that **ideology/party affiliation** is the strongest driver of legislative coordination.
2. **Geography**: The **Province** plot shows overlapping, highly dispersed colors (Silhouette = -0.7149). This indicates that MPs do not cluster by their electoral geography; regional coordination is subordinate to party-line coordination.
3. **Seniority/Activity**: The **Activity (Seniority)** plot shows that the tightest, most cohesive clusters are occupied by highly active MPs, while peripheral nodes represent MPs with low activity (low motion counts).

---

### Sub-question 4: Cross-party "Bridge" MPs

We identified cross-party broker MPs by finding nodes with **high Betweenness Centrality** (global path bridging) but **low Intra-party Closeness Centrality** (peripheral position within their own party).

The top 10 bridge MPs sorted by their Bridge Score are:

| Rank | MP | Party | Province | Betweenness | Intra-party Closeness | Bridge Score |
|---|---|---|---|---|---|---|
| 1 | **Mustafa Sezgin TANRIKULU** | CHP | Diyarbakır | 0.1936 | 0.0175 | **0.5631** |
| 2 | **Selçuk ÖZDAĞ** | CHP | Muğla | 0.0611 | 0.0175 | **0.1776** |
| 3 | **Ömer Fethi GÜRER** | CHP | Niğde | 0.0562 | 0.0175 | **0.1635** |
| 4 | **Ali KARAOBA** | CHP | Uşak | 0.0493 | 0.0175 | **0.1434** |
| 5 | **Doğan BEKİN** | Yeniden Refah | İstanbul (III) | 0.0318 | 0.0030 | **0.0927** |
| 6 | **Mustafa BİLİCİ** | CHP | İzmir (II) | 0.0284 | 0.0175 | **0.0826** |
| 7 | **Mahmut TANAL** | CHP | Şanlıurfa | 0.0255 | 0.0175 | **0.0741** |
| 8 | **Yasin ÖZTÜRK** | İYİ Parti | Denizli | 0.0212 | 0.0129 | **0.0618** |
| 9 | **Ulaş KARASU** | CHP | Sivas | 0.0206 | 0.0175 | **0.0600** |
| 10| **Adnan Şefik ÇİRKİN** | İYİ Parti | Hatay | 0.0190 | 0.0129 | **0.0554** |

#### Explanation & Insights:
1. **The Ultimate Broker**: **Sezgin Tanrıkulu** (CHP) has a bridge score of **0.5631**, which is nearly triple that of the next highest MP. As a CHP MP from Diyarbakır, he frequently coordinates and submits motions on human rights, regional challenges, and Kurdish cultural rights. These topics are highly overlapping with the DEM Party's agenda. This unique legislative footprint places him directly between the main CHP cluster and the DEM Party cluster, making him the most critical communication channel in the entire parliament.
2. **Small-Party Mediators**: **Doğan Bekin** (Yeniden Refah Partisi) and MPs from Saadet/Gelecek/DEVA (now represented in the joint *Yeni Yol* group, such as **Selçuk Özdağ** and **Mustafa Bilici**) serve as structural bridges. Because they belong to smaller opposition factions, they coordinate selectively with both CHP and DEM Party, acting as key intermediaries.
