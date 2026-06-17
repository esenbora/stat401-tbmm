import re
import csv
import json
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import minmax_scale
import umap
from gensim.models import Word2Vec
import random

# Ensure output encoding is utf-8 for Windows console
if sys.platform.startswith('win'):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

# ----------------------------------------------------
# 1. Parsing Wikipedia Data (from tbmm_wiki.txt)
# ----------------------------------------------------
def clean_wiki_link(val):
    val = val.strip()
    if val.startswith('[[') and val.endswith(']]'):
        val = val[2:-2]
    if '|' in val:
        val = val.split('|')[1]
    return val.strip()

def normalize_name(name):
    if not name:
        return ""
    name = name.lower()
    replacements = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'â': 'a', 'î': 'i', 'û': 'u'
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    name = re.sub(r'[^a-z]', '', name)
    return name

print("Step 1: Parsing Wikipedia wikitext...")
with open('tbmm_wiki.txt', encoding='utf-8') as f:
    text = f.read()

tables = re.findall(r'\{\|.*?(?=\|\})', text, re.DOTALL)
table_text = tables[4]
raw_rows = table_text.split('|-')

grid = [[None for _ in range(6)] for _ in range(650)]
r_idx = 0
for raw_row in raw_rows:
    raw_row = raw_row.strip()
    if not raw_row or raw_row.startswith('{|') or '! Seçi' in raw_row or '! Millet' in raw_row or 'colspan="6"' in raw_row:
        continue
        
    cells = []
    lines = raw_row.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('|'):
            line_content = line[1:].strip()
            parts = [p.strip() for p in line_content.split('||')]
            cells.extend(parts)
            
    if not cells:
        continue
        
    c_idx = 0
    for cell in cells:
        while c_idx < 6 and grid[r_idx][c_idx] is not None:
            c_idx += 1
        if c_idx >= 6:
            break
            
        rowspan = 1
        colspan = 1
        cell_content = cell
        
        if '|' in cell and not ('[[' in cell and ']]' in cell and cell.index('|') > cell.index('[[')):
            parts = cell.split('|', 1)
            attrs = parts[0]
            if '[[' in attrs or ']]' in attrs:
                cell_content = cell
            else:
                cell_content = parts[1]
                m_rs = re.search(r'rowspan="?(\d+)"?', attrs)
                if m_rs:
                    rowspan = int(m_rs.group(1))
                m_cs = re.search(r'colspan="?(\d+)"?', attrs)
                if m_cs:
                    colspan = int(m_cs.group(1))
                    
        for dr in range(rowspan):
            for dc in range(colspan):
                if r_idx + dr < len(grid) and c_idx + dc < 6:
                    grid[r_idx + dr][c_idx + dc] = cell_content
        c_idx += colspan
    r_idx += 1

parsed_mps = []
for r in range(r_idx):
    row = grid[r]
    mp_cell = row[1]
    if not mp_cell:
        continue
    mp_name = clean_wiki_link(mp_cell)
    if not mp_name or mp_name.startswith('Milletvekili') or mp_name == 'Seçim bölgesi':
        continue
        
    province_cell = row[0]
    province = ""
    if province_cell:
        m_prov = re.search(r'\[\[(?:.*?milletvekilleri listesi\|)?(.*?)\]\]', province_cell)
        province = m_prov.group(1).strip() if m_prov else re.sub(r'[\'\'\"\"\{\}]', '', clean_wiki_link(province_cell)).strip()
            
    party_cell = row[3]
    party = clean_wiki_link(party_cell) if party_cell else None
    
    change_party_cell = row[5]
    change_party = clean_wiki_link(change_party_cell) if change_party_cell else None
    
    # Clean party names
    def clean_party_name(p):
        if not p: return "Bağımsız"
        p = p.replace("Bağımsız siyasetçi|", "")
        if "Halkların Eşitlik ve Demokrasi Partisi" in p or "Yeşiller ve Sol Gelecek Partisi" in p:
            return "DEM Parti"
        if "Demokrasi ve Atılım Partisi" in p:
            return "DEVA Partisi"
        if "Türkiye İşçi Partisi" in p:
            return "TİP"
        if "Hür Dava Partisi" in p:
            return "HÜDA PAR"
        return p

    parsed_mps.append({
        'name': mp_name,
        'province': province,
        'elected_party': clean_party_name(party),
        'current_party': clean_party_name(change_party if change_party else party)
    })

print(f"Parsed {len(parsed_mps)} MPs from Wikipedia.")

# ----------------------------------------------------
# 2. Loading CSV and Aligning MPs
# ----------------------------------------------------
print("Step 2: Loading CSV and aligning MPs...")
df = pd.read_csv('tum_metadata_44485.csv')
print(f"Loaded {len(df)} motion rows.")

csv_mps = sorted(df['Milletvekili'].dropna().unique())
print(f"Found {len(csv_mps)} unique MPs in CSV.")

wiki_mp_by_norm = {normalize_name(mp['name']): mp for mp in parsed_mps}

# Manual overrides for names that differ
MANUAL_MAPPING = {
    "Hakkı Saruhan OLUÇ": "Hakkı Saruhan Oruç",
    "Mehmet Selim ENSARİOĞLU": "Mehmet Salim Ensarioğlu",
    "Selcan TAŞCI": "Selcan Hamşıoğlu",
    "Şahzade DEMİR": "Şehzade Demir"
}

matches = {}
for csv_name in csv_mps:
    if csv_name in MANUAL_MAPPING:
        matches[csv_name] = wiki_mp_by_norm[normalize_name(MANUAL_MAPPING[csv_name])]
        continue
        
    norm = normalize_name(csv_name)
    if norm in wiki_mp_by_norm:
        matches[csv_name] = wiki_mp_by_norm[norm]
    else:
        # Partial match fallback
        found = False
        for wiki_norm, mp in wiki_mp_by_norm.items():
            if norm in wiki_norm or wiki_norm in norm:
                matches[csv_name] = mp
                found = True
                break
        if not found:
            print(f"WARNING: Unmatched MP in CSV: {csv_name}")

print(f"Successfully matched {len(matches)} / {len(csv_mps)} CSV MPs.")

# Add party/province details to DataFrame
df['party_elected'] = df['Milletvekili'].map(lambda x: matches[x]['elected_party'] if x in matches else "Bilinmiyor")
df['party_current'] = df['Milletvekili'].map(lambda x: matches[x]['current_party'] if x in matches else "Bilinmiyor")
df['province'] = df['Milletvekili'].map(lambda x: matches[x]['province'] if x in matches else "Bilinmiyor")

# ----------------------------------------------------
# 3. Constructing the Co-signing Network
# ----------------------------------------------------
print("Step 3: Constructing the co-signing network...")
# Group by Önerge Özeti to find identical motions
summary_groups = df.groupby('Önerge Özeti')['Milletvekili'].apply(list)

# We only consider summaries with > 1 MP (this defines co-signing)
co_signed_groups = [mps for mps in summary_groups if len(set(mps)) > 1]
print(f"Found {len(co_signed_groups)} campaigns (identical summaries submitted by multiple MPs).")

# Build co-signing edge weights
edge_weights = {}
for mps in co_signed_groups:
    unique_mps = sorted(list(set(mps)))
    for i in range(len(unique_mps)):
        for j in range(i + 1, len(unique_mps)):
            mp_a, mp_b = unique_mps[i], unique_mps[j]
            edge = (mp_a, mp_b)
            edge_weights[edge] = edge_weights.get(edge, 0) + 1

# Create networkx graph
G = nx.Graph()
# Add nodes with metadata
mp_motion_counts = df['Milletvekili'].value_counts()
for mp in csv_mps:
    if mp in matches:
        G.add_node(mp, 
                   party_elected=matches[mp]['elected_party'],
                   party_current=matches[mp]['current_party'],
                   province=matches[mp]['province'],
                   motion_count=int(mp_motion_counts.get(mp, 0)))

# Add weighted edges
for (u, v), w in edge_weights.items():
    G.add_edge(u, v, weight=w)

print(f"Co-signing graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# Keep only the largest connected component for centrality analysis
largest_cc = max(nx.connected_components(G), key=len)
print(f"Largest connected component size: {len(largest_cc)} nodes")

# ----------------------------------------------------
# 4. Centrality Analysis (Sub-question 1)
# ----------------------------------------------------
print("Step 4: Computing PageRank and HITS...")
pagerank = nx.pagerank(G, weight='weight')
hits_hubs, hits_authorities = nx.hits(G, max_iter=500)

top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
top_hits = sorted(hits_authorities.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n--- TOP 10 MPs by PageRank Centrality ---")
for idx, (mp, pr) in enumerate(top_pr):
    party = G.nodes[mp]['party_elected']
    province = G.nodes[mp]['province']
    print(f"{idx+1}. {mp} ({party} - {province}): {pr:.6f}")

print("\n--- TOP 10 MPs by HITS Authority Score ---")
for idx, (mp, auth) in enumerate(top_hits):
    party = G.nodes[mp]['party_elected']
    province = G.nodes[mp]['province']
    print(f"{idx+1}. {mp} ({party} - {province}): {auth:.6f}")

# ----------------------------------------------------
# 5. Louvain Community Detection (Sub-question 2)
# ----------------------------------------------------
print("\nStep 5: Performing Louvain community detection...")
# The louvain library expects partition on the largest component or whole graph
partition = community_louvain.best_partition(G, weight='weight')
nx.set_node_attributes(G, partition, 'louvain_community')

# Number of communities
num_communities = len(set(partition.values()))
print(f"Louvain detected {num_communities} communities.")

# Evaluate alignment with formal parties
nodes_in_partition = list(partition.keys())
true_parties_elected = [G.nodes[n]['party_elected'] for n in nodes_in_partition]
true_parties_current = [G.nodes[n]['party_current'] for n in nodes_in_partition]
predicted_communities = [partition[n] for n in nodes_in_partition]

ari_elected = adjusted_rand_score(true_parties_elected, predicted_communities)
nmi_elected = normalized_mutual_info_score(true_parties_elected, predicted_communities)

ari_current = adjusted_rand_score(true_parties_current, predicted_communities)
nmi_current = normalized_mutual_info_score(true_parties_current, predicted_communities)

print(f"Community alignment with Elected Party:")
print(f"  Adjusted Rand Index (ARI): {ari_elected:.4f}")
print(f"  Normalized Mutual Information (NMI): {nmi_elected:.4f}")

print(f"Community alignment with Current Party/Alliance Group:")
print(f"  Adjusted Rand Index (ARI): {ari_current:.4f}")
print(f"  Normalized Mutual Information (NMI): {nmi_current:.4f}")

# Inspect party composition of each community
print("\n--- Louvain Community Party Composition ---")
for comm_id in sorted(set(partition.values())):
    comm_nodes = [n for n in nodes_in_partition if partition[n] == comm_id]
    party_dist = pd.Series([G.nodes[n]['party_elected'] for n in comm_nodes]).value_counts()
    print(f"Community {comm_id} (Size {len(comm_nodes)}):")
    for party, count in party_dist.items():
        print(f"  - {party}: {count} ({count/len(comm_nodes)*100:.1f}%)")

# ----------------------------------------------------
# 6. Node2Vec + UMAP Embeddings (Sub-question 3)
# ----------------------------------------------------
print("\nStep 6: Generating Node2Vec + UMAP Embeddings...")

# Implement a simple, fast Node2Vec random walk generator
def generate_walks(graph, num_walks=10, walk_length=40):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                neighbors = list(graph.neighbors(curr))
                if not neighbors:
                    break
                # Biased walks: here we do simple weighted walks
                # Since graph has edge weights, we weight neighbor selection by edge weight
                weights = [graph[curr][nbr].get('weight', 1.0) for nbr in neighbors]
                probs = np.array(weights) / sum(weights)
                nxt = np.random.choice(neighbors, p=probs)
                walk.append(nxt)
            walks.append([str(w) for w in walk])
    return walks

print("  Generating biased random walks...")
walks = generate_walks(G, num_walks=20, walk_length=60)

print("  Training Word2Vec model...")
# Learn node embeddings
w2v = Word2Vec(sentences=walks, vector_size=64, window=10, min_count=1, sg=1, workers=4, epochs=15)
nodes_ordered = [n for n in G.nodes() if str(n) in w2v.wv]
embeddings = np.array([w2v.wv[str(n)] for n in nodes_ordered])

print(f"  Reducing embeddings using UMAP...")
# Fit UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_coords = reducer.fit_transform(embeddings)

# Prepare DataFrame for visualization
viz_df = pd.DataFrame({
    'MP': nodes_ordered,
    'x': umap_coords[:, 0],
    'y': umap_coords[:, 1],
    'Party': [G.nodes[n]['party_elected'] for n in nodes_ordered],
    'Province': [G.nodes[n]['province'] for n in nodes_ordered],
    'Motions': [G.nodes[n]['motion_count'] for n in nodes_ordered]
})
viz_df['Log_Motions'] = np.log1p(viz_df['Motions'])

# Generate Plots
print("  Generating UMAP plots...")
plt.style.use('dark_background')

# Plot 1: Party
plt.figure(figsize=(10, 8))
sns.scatterplot(data=viz_df, x='x', y='y', hue='Party', palette='Set1', s=80, alpha=0.8)
plt.title('UMAP Clustering of MPs Colored by Elected Party', fontsize=14, fontweight='bold')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('umap_party.png', dpi=150)
plt.close()

# Plot 2: Province (Show top 8 provinces, label others as Other)
top_provinces = viz_df['Province'].value_counts().head(8).index.tolist()
viz_df['Province_Group'] = viz_df['Province'].apply(lambda x: x if x in top_provinces else 'Diğer (Other)')
plt.figure(figsize=(10, 8))
sns.scatterplot(data=viz_df, x='x', y='y', hue='Province_Group', palette='tab20', s=80, alpha=0.8)
plt.title('UMAP Clustering of MPs Colored by Geography (Province)', fontsize=14, fontweight='bold')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('umap_province.png', dpi=150)
plt.close()

# Plot 3: Seniority (Proxy: Log of total motions submitted)
plt.figure(figsize=(10, 8))
sc = plt.scatter(viz_df['x'], viz_df['y'], c=viz_df['Log_Motions'], cmap='viridis', s=80, alpha=0.8)
plt.colorbar(sc, label='Log(Total Motions + 1)')
plt.title('UMAP Clustering of MPs Colored by Activity Level (Seniority Proxy)', fontsize=14, fontweight='bold')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.savefig('umap_seniority.png', dpi=150)
plt.close()

print("UMAP plots saved: umap_party.png, umap_province.png, umap_seniority.png")

# Calculate clustering silhouette scores or statistics
# Let's see if we cluster more by party or province
from sklearn.metrics import silhouette_score
party_labels = viz_df['Party']
province_labels = viz_df['Province']
print(f"\nSilhouette Scores on UMAP coordinates:")
print(f"  Party Clustering: {silhouette_score(umap_coords, party_labels):.4f}")
print(f"  Province Clustering: {silhouette_score(umap_coords, province_labels):.4f}")

# ----------------------------------------------------
# 7. Bridge MP Detection (Sub-question 4)
# ----------------------------------------------------
print("\nStep 7: Finding cross-party bridge MPs...")
# Betweenness centrality (using path length as reciprocal of edge weight)
# High edge weight means high closeness, so distance is 1 / weight
distance_dict = {(u, v): 1.0 / d['weight'] for u, v, d in G.edges(data=True)}
nx.set_edge_attributes(G, distance_dict, 'distance')

betweenness = nx.betweenness_centrality(G, weight='distance')

# Compute intra-party closeness centrality
# Closeness centrality to members of the SAME party.
# For each node u, we compute the average path distance to all other nodes v of the same party.
# Intra-party Closeness(u) = (|P_u| - 1) / Sum_{v in P_u, v != u} d(u, v)
intra_party_closeness = {}
parties = nx.get_node_attributes(G, 'party_elected')

# Compute all pair shortest path lengths using weighted distance
all_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='distance'))

for u in G.nodes():
    party_u = parties[u]
    same_party_nodes = [v for v in G.nodes() if parties[v] == party_u and v != u]
    
    if not same_party_nodes:
        intra_party_closeness[u] = 0.0
        continue
        
    distances = []
    for v in same_party_nodes:
        if v in all_path_lengths[u]:
            distances.append(all_path_lengths[u][v])
        else:
            # unreachable
            distances.append(999.0) # Penalty value
            
    sum_dist = sum(distances)
    if sum_dist > 0:
        intra_party_closeness[u] = len(same_party_nodes) / sum_dist
    else:
        intra_party_closeness[u] = 0.0

# Normalize centralities to 0-1 for plotting/scoring
mps_list = list(G.nodes())
bet_vals = np.array([betweenness[m] for m in mps_list])
clo_vals = np.array([intra_party_closeness[m] for m in mps_list])

norm_betweenness = minmax_scale(bet_vals)
norm_closeness = minmax_scale(clo_vals)

# Bridge score: high normalized betweenness, low normalized intra-party closeness
# Bridge_Score = Norm_Betweenness * (1 - Norm_Closeness)
bridge_scores = {}
for i, mp in enumerate(mps_list):
    bridge_scores[mp] = norm_betweenness[i] * (1.0 - norm_closeness[i])

top_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n--- TOP 10 CROSS-PARTY BRIDGE MPs ---")
print("High Betweenness Centrality (global broker) & Low Intra-party Closeness (peripheral to own party)")
for idx, (mp, score) in enumerate(top_bridges):
    party = G.nodes[mp]['party_elected']
    province = G.nodes[mp]['province']
    bet = betweenness[mp]
    clo = intra_party_closeness[mp]
    print(f"{idx+1}. {mp} ({party} - {province}): Bridge Score = {score:.4f} (Betweenness = {bet:.4f}, Intra-party Closeness = {clo:.4f})")

# Write results to output file
print("\nStep 8: Exporting results...")
output_data = {
    'top_pagerank': [{'mp': k, 'party': G.nodes[k]['party_elected'], 'province': G.nodes[k]['province'], 'score': float(v)} for k, v in top_pr],
    'top_hits': [{'mp': k, 'party': G.nodes[k]['party_elected'], 'province': G.nodes[k]['province'], 'score': float(v)} for k, v in top_hits],
    'top_bridges': [
        {
            'mp': k, 
            'party': G.nodes[k]['party_elected'], 
            'province': G.nodes[k]['province'], 
            'score': float(bridge_scores[k]),
            'betweenness': float(betweenness[k]),
            'intra_party_closeness': float(intra_party_closeness[k])
        } for k, v in top_bridges
    ],
    'metrics': {
        'num_communities': int(num_communities),
        'ari_elected': float(ari_elected),
        'nmi_elected': float(nmi_elected),
        'ari_current': float(ari_current),
        'nmi_current': float(nmi_current),
        'silhouette_party': float(silhouette_score(umap_coords, party_labels)),
        'silhouette_province': float(silhouette_score(umap_coords, province_labels))
    }
}

with open('network_analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("Analysis successfully completed and saved!")
