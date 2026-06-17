# Network Analysis Walkthrough: Turkish MPs (TBMM 28th Term)

This document summarizes the steps taken to perform the social network analysis on the Turkish Grand National Assembly (TBMM) 28th Term dataset.

## Work Accomplished

1. **Data Discovery & Inspection**:
   - Analyzed `tum_metadata_44485.csv` and determined it consists of **44,485 written question motions** (`7/` prefix) from the 28th legislative term.
   - Identified that each motion is registered under a single MP. Therefore, a "co-signing" relation is modeled by identifying MPs who submitted motions with the *exact same summary* (`Önerge Özeti`).

2. **Metadata Scrape and Alignment**:
   - Downloaded and parsed the wikitext of `TBMM 28. dönem milletvekilleri listesi` from Wikipedia using Python.
   - Implemented a robust 2D table parser to handle complex rowspans and colspans.
   - Extracted official electoral provinces, elected parties, and current party/group affiliations for all 600 MPs.
   - Aligned the 307 unique MPs in the CSV file with the parsed Wikipedia data (achieving 100% matching using name normalization and manual mapping overrides for spelling variations).

3. **Network Construction & Graph Analysis**:
   - Created the co-signing network in `networkx`, where nodes represent MPs and edges represent shared identical motion summaries (edge weight = count of overlaps).
   - Extracted the largest connected component of **272 nodes**.
   - Computed centrality scores: **PageRank** and **HITS Authority**.
   - Performed **Louvain community detection** and calculated clustering similarity metrics (Adjusted Rand Index and Normalized Mutual Information) to evaluate alignment with official parties.
   - Detected cross-party **bridge MPs** using a custom score balancing high Betweenness Centrality (global bridging) and low Intra-party Closeness Centrality (peripheral in own party).

4. **Biased Random Walks & Graph Embeddings**:
   - Generated Node2Vec embeddings using weighted random walks and trained a Word2Vec model in `gensim`.
   - Used UMAP to project embeddings into 2D and saved three plots: `umap_party.png`, `umap_province.png`, and `umap_seniority.png`.

## Execution and Output Verification

The pipeline was executed successfully in the workspace directory. All generated files have been saved:
- Python analysis script: [analyze_network.py](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/analyze_network.py)
- Main analysis report: [analysis_results.md](file:///C:/Users/HP/.gemini/antigravity-ide/brain/9ab54d4e-5106-4086-ac90-cffa361f57ef/analysis_results.png) (contains embedding plots and details)
- JSON data export: `network_analysis_results.json`
- Plots:
  - [umap_party.png](file:///C:/Users/HP/.gemini/antigravity-ide/brain/9ab54d4e-5106-4086-ac90-cffa361f57ef/umap_party.png)
  - [umap_province.png](file:///C:/Users/HP/.gemini/antigravity-ide/brain/9ab54d4e-5106-4086-ac90-cffa361f57ef/umap_province.png)
  - [umap_seniority.png](file:///C:/Users/HP/.gemini/antigravity-ide/brain/9ab54d4e-5106-4086-ac90-cffa361f57ef/umap_seniority.png)

## Main Outcomes

- **Hubs**: The DEM Party forms an extremely dense clique due to highly structured collective campaign motions, dominating the PageRank and HITS metrics.
- **Communities**: Louvain community detection successfully identifies the DEM Party as an isolated cluster, while CHP and İYİ Party MPs merge into a single large community, showing close coordination.
- **Bridges**: **Sezgin Tanrıkulu** (CHP, Diyarbakır) emerges as the ultimate cross-party broker between CHP/İYİ and DEM Party, representing a crucial communication channel.
