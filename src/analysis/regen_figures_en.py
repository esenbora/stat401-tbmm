"""Regenerate the 3 LDA/K-Means figures from the saved (Jun-19) dashboard parquet
snapshots with English labels — no Spark, fully deterministic. The other 7 figures are
deterministic counts and are left untouched.

Reads:  src/dashboard/data/{rq1_ministry_topic, rq2_province_topic, rq2_province_cluster}.parquet
Writes: deliverables/figures/{rq1_s2_ministry_topic, rq2_s2_metro_rural, rq2_s3_clusters}.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "dashboard" / "data"
FIG = ROOT / "deliverables" / "figures"

RQ1_TOPICS = 15
RQ2_TOPICS = 12
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

METRO = {"İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya",
         "Gaziantep", "Şanlıurfa", "Kocaeli", "Mersin", "Diyarbakır", "Kayseri", "Samsun"}


def short_ministry(name: str) -> str:
    return (name or "?").replace(" Bakanlığı", "").replace(" Ve ", "/").strip()


# ---- rq1_s2: Ministry x LDA topic heatmap ------------------------------------
mt = pd.read_parquet(DATA / "rq1_ministry_topic.parquet")[["bakanlik", "dom_topic", "n"]]
piv = mt.pivot(index="bakanlik", columns="dom_topic", values="n").fillna(0)
piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).head(12).index]
piv.index = [short_ministry(i) for i in piv.index]
norm = piv.div(piv.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(norm.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(RQ1_TOPICS)); ax.set_xticklabels([f"T{i}" for i in range(RQ1_TOPICS)])
ax.set_yticks(range(len(norm.index))); ax.set_yticklabels(norm.index)
ax.set_title("Ministry × LDA topic distribution (row-normalised)")
plt.colorbar(im, ax=ax, label="share")
plt.tight_layout(); plt.savefig(FIG / "rq1_s2_ministry_topic.png"); plt.close()

# ---- shared RQ2 province x topic shares --------------------------------------
pt = pd.read_parquet(DATA / "rq2_province_topic.parquet")[["il", "dom_topic", "n"]]
piv2 = pt.pivot(index="il", columns="dom_topic", values="n").fillna(0)
piv2 = piv2.reindex(columns=range(RQ2_TOPICS), fill_value=0)
share = piv2.div(piv2.sum(axis=1), axis=0)

# ---- rq2_s2: metro vs rural topic profile ------------------------------------
metro_prof = share.loc[[i for i in share.index if i in METRO]].mean()
rural_prof = share.loc[[i for i in share.index if i not in METRO]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(RQ2_TOPICS); w = 0.4
ax.bar(x - w / 2, metro_prof.values, w, label="Metropolitan", color="#c0392b")
ax.bar(x + w / 2, rural_prof.values, w, label="Rural/other", color="#27ae60")
ax.set_xticks(x); ax.set_xticklabels([f"T{i}" for i in range(RQ2_TOPICS)])
ax.set_xlabel("LDA topic"); ax.set_ylabel("Mean share")
ax.set_title("Metropolitan vs rural provinces — topic profile"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "rq2_s2_metro_rural.png"); plt.close()

# ---- rq2_s3: K-Means cluster scatter -----------------------------------------
clus = pd.read_parquet(DATA / "rq2_province_cluster.parquet")
clusters: dict[int, list[str]] = {}
for il, c in zip(clus["il"], clus["cluster"]):
    if il in share.index:
        clusters.setdefault(int(c), []).append(il)
var_topics = share.var().sort_values(ascending=False).head(2).index.tolist()
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap("tab10")
for c, members in sorted(clusters.items()):
    ax.scatter(share.loc[members, var_topics[0]], share.loc[members, var_topics[1]],
               color=cmap(c), label=f"Cluster {c} (n={len(members)})", s=60, alpha=0.8)
    for il in members:
        if il in METRO or share.loc[il].sum() > 0:
            ax.annotate(il, (share.loc[il, var_topics[0]], share.loc[il, var_topics[1]]),
                        fontsize=6, alpha=0.6)
ax.set_xlabel(f"Topic T{var_topics[0]} share"); ax.set_ylabel(f"Topic T{var_topics[1]} share")
ax.set_title("Provincial attention profiles — K-Means++ clusters"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "rq2_s3_clusters.png"); plt.close()

print("Regenerated 3 figures from Jun-19 parquet with English labels:")
print("  - rq1_s2_ministry_topic.png")
print("  - rq2_s2_metro_rural.png")
print("  - rq2_s3_clusters.png")
