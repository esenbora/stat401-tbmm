"""STAT 401 — TBMM 28th-term Parliamentary Analytics dashboard.

Three tabs (RQ1 / RQ2 / RQ3) over the Spark-MLlib results. Reads the small
parquet snapshots written by ``src/analysis/export_dashboard.py`` and the
metrics JSON written by the RQ scripts — no Spark needed at view time, so the
app starts instantly.

UI text is English; data values (ministry / party / province names) stay in
Turkish because they are proper nouns.

Run:
    .venv/bin/python -m streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
DELIV = ROOT / "deliverables"
FIG = DELIV / "figures"
GEOJSON = ROOT / "data" / "reference" / "tr_provinces.geojson"

PARTY_COLORS = {
    "CHP": "#e41a1c", "DEM Parti": "#984ea3", "İYİ Parti": "#377eb8",
    "MHP": "#ff7f00", "AK Parti": "#f7d000", "Yeniden Refah": "#4daf4a",
    "TİP": "#a65628", "HÜDA PAR": "#999999", "Bilinmiyor": "#cccccc",
}
# Fixed colour per ministry so the trend lines keep the same colour regardless
# of the year-range slider (built once from the full ministry list below).
_PAL = px.colors.qualitative.Dark24

st.set_page_config(page_title="TBMM Parliamentary Analytics", page_icon="🇹🇷",
                   layout="wide", initial_sidebar_state="expanded")


# ----- loaders (cached) --------------------------------------------------
@st.cache_data
def pq(name: str) -> pd.DataFrame:
    f = DATA / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


@st.cache_data
def metrics(rq: str) -> dict:
    f = DELIV / f"{rq}_metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


@st.cache_data
def geojson() -> dict | None:
    return json.loads(GEOJSON.read_text(encoding="utf-8")) if GEOJSON.exists() else None


def short_ministry(s: str) -> str:
    return (s or "?").replace(" Bakanlığı", "").replace(" Ve ", "/")


# ----- sidebar -----------------------------------------------------------
with st.sidebar:
    st.title("🇹🇷 TBMM Analytics")
    st.caption("STAT 401 — 28th Term Written Questions")
    st.metric("Total questions", "44,484")
    st.metric("MPs who filed questions", "307 / 592")
    st.caption("592 MPs total · 307 filed written questions · 265 in the RQ3 network")
    st.metric("Ministries / provinces", "20 / 81")
    st.caption("Full-text OCR coverage: 100% (108.6M characters)")
    st.divider()
    st.caption("Apache Spark · Delta Lake · Spark MLlib · Streamlit")
    if not (DATA / "rq1_ministry_year.parquet").exists():
        st.error("No data — run `export_dashboard.py` first.")

st.title("TBMM 28th Term — Parliamentary Big-Data Analysis")
tab1, tab2, tab3 = st.tabs([
    "📊 Ministry × Topic × Party (RQ1)",
    "🗺️ Provincial Attention Map (RQ2)",
    "🕸️ MP Network (RQ3)",
])

# =====================================================================
# RQ1
# =====================================================================
with tab1:
    st.header("Ministry × Topic × Party × Time")
    m = metrics("rq1")
    my = pq("rq1_ministry_year")
    pm = pq("rq1_party_ministry")

    if my.empty:
        st.warning("No RQ1 data.")
    else:
        my["b"] = my["bakanlik"].map(short_ministry)
        ministry_colors = {b: _PAL[i % len(_PAL)]
                           for i, b in enumerate(sorted(my["b"].unique()))}
        years = sorted(int(y) for y in my["yil"].dropna().unique())
        yr = st.select_slider("Year range", options=years, value=(years[0], years[-1]), key="rq1yr")
        myf = my[my["yil"].between(yr[0], yr[1])]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Question volume by ministry")
            agg = myf.groupby("b")["n"].sum().sort_values(ascending=False).head(12)
            st.plotly_chart(px.bar(agg, orientation="h", labels={"value": "questions", "b": ""})
                            .update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False),
                            width="stretch")
        with c2:
            st.subheader("Trend by year (Top 6)")
            top6 = myf.groupby("b")["n"].sum().sort_values(ascending=False).head(6).index
            tl = myf[myf["b"].isin(top6)].sort_values(["b", "yil"])
            st.plotly_chart(px.line(tl, x="yil", y="n", color="b", markers=True,
                            color_discrete_map=ministry_colors,
                            labels={"yil": "year", "n": "questions", "b": "ministry"}),
                            width="stretch")

        if not pm.empty:
            st.subheader("Party × Ministry focus (row-normalised)")
            mains = ["CHP", "DEM Parti", "İYİ Parti", "MHP", "AK Parti"]
            sub = pm[pm["party"].isin(mains)].copy()
            sub["b"] = sub["bakanlik"].map(short_ministry)
            piv = sub.pivot_table(index="party", columns="b", values="n", aggfunc="sum").fillna(0)
            piv = piv[piv.sum().sort_values(ascending=False).head(12).index]
            norm = piv.div(piv.sum(axis=1), axis=0)
            st.plotly_chart(px.imshow(norm, aspect="auto", color_continuous_scale="Magma",
                            labels={"color": "share"}), width="stretch")

        cc1, cc2 = st.columns(2)
        with cc1:
            if m.get("party_clf"):
                clf = m["party_clf"]
                st.subheader("Predicting party from text")
                st.metric("Accuracy (LogReg, TF-IDF)", f"{clf['accuracy']*100:.1f}%",
                          f"baseline {clf['baseline_majority']*100:.1f}%")
                st.caption(f"Classes: {', '.join(clf['classes'])} · F1={clf['f1']:.3f}")
                st.image(str(FIG / "rq1_s3_confusion.png"))
        with cc2:
            if m.get("duplicates"):
                d = m["duplicates"]
                st.subheader("Duplicate/coordinated questions (MinHash ≥0.8)")
                st.metric("Near-duplicate pairs", f"{d['near_dup_pairs']:,}")
                st.caption(f"{d['cross_mp_pairs']:,} different MPs · {d['cross_party_pairs']:,} different parties")
                cp = pq("rq1_dup_cross_party")
                if not cp.empty:
                    cp = cp.sort_values("pairs", ascending=False).head(8)
                    cp["pair"] = cp["party_a"] + " ↔ " + cp["party_b"]
                    st.plotly_chart(px.bar(cp, x="pairs", y="pair", orientation="h")
                                    .update_layout(yaxis={"categoryorder": "total ascending"}),
                                    width="stretch")

        if m.get("lda_topics"):
            with st.expander(f"LDA topics ({len(m['lda_topics'])}) — most frequent words"):
                st.dataframe(pd.DataFrame([
                    {"Topic": f"T{t['topic']}", "Words": ", ".join(t["words"])}
                    for t in m["lda_topics"]]), width="stretch", hide_index=True)

# =====================================================================
# RQ2
# =====================================================================
with tab2:
    st.header("Provincial Parliamentary Attention")
    m = metrics("rq2")
    men = pq("rq2_province_mentions")
    corr = pq("rq2_province_correlates")
    clus = pq("rq2_province_cluster")
    gj = geojson()

    if men.empty:
        st.warning("No RQ2 data.")
    else:
        # Ankara raw count is a capital/ministry-address artifact (ministries
        # sit in Ankara), not subject attention — exclude it everywhere on this tab.
        men_disp = men[men["il"] != "Ankara"]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Provincial attention map (mention count)")
            st.caption("Ankara excluded (ministry addresses inflate the body text — not the subject province).")
            if gj is not None:
                fig = px.choropleth(men_disp, geojson=gj, locations="il", featureidkey="properties.name",
                                    color="mentions", color_continuous_scale="YlGnBu",
                                    labels={"mentions": "mentions"})
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=420)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No geojson for the map — see the bar chart.")
        with c2:
            st.subheader("Most-mentioned provinces (Ankara excluded)")
            top = men_disp.sort_values("mentions", ascending=False).head(15)
            st.plotly_chart(px.bar(top, x="mentions", y="il", orientation="h")
                            .update_layout(yaxis={"categoryorder": "total ascending"}, height=420),
                            width="stretch")

        if not corr.empty:
            st.subheader("Attention vs population — targeting")
            # Ankara is the capital/ministry artifact (excluded from the map/bar
            # above too); drop it here so the correlation and per-capita ranking
            # match the report. r is computed from the Ankara-excluded frame, not
            # read from the JSON (which still carries the with-Ankara 0.40).
            corr_disp = corr[corr["il"] != "Ankara"]
            r_full = corr_disp["population"].corr(corr_disp["attention"])
            r = round(r_full, 2)
            cp = m.get("correlation_pearson", {})
            r_with = cp.get("attention~population")
            cc1, cc2 = st.columns([3, 2])
            with cc1:
                fig = px.scatter(corr_disp, x="population", y="attention", hover_name="il",
                                 size="att_per_100k", labels={"population": "population", "attention": "mentions"})
                st.plotly_chart(fig, width="stretch")
                cap = (f"Pearson r(attention, population) = {r} (Ankara excluded)  ·  "
                       f"shared variance ~{r_full**2*100:.0f}%")
                if r_with is not None:
                    cap += f"  ·  with Ankara included r = {r_with} (~{r_with**2*100:.0f}%)"
                st.caption(cap)
            with cc2:
                st.markdown("**Most attention per 100k (Ankara excluded)**")
                over = corr_disp.sort_values("att_per_100k", ascending=False).head(8)
                st.dataframe(over[["il", "attention", "att_per_100k"]].round(1),
                             width="stretch", hide_index=True)

        if not clus.empty:
            st.subheader("K-Means++ attention profiles")
            sil = m.get("kmeans_silhouette")
            clusters = m.get("kmeans_clusters", {})

            # interactive scatter: province × topic shares coloured by cluster.
            pt = pq("rq2_province_topic")
            if not pt.empty:
                wide = pt.pivot_table(index="il", columns="dom_topic", values="n", aggfunc="sum").fillna(0)
                share = wide.div(wide.sum(axis=1), axis=0)
                share.columns = [f"T{c}" for c in share.columns]
                # two highest-variance topics span the clusters best
                tx, ty = share.var().sort_values(ascending=False).head(2).index
                sc = share[[tx, ty]].reset_index().merge(clus, on="il")
                sc["cluster"] = sc["cluster"].astype(str)
                st.plotly_chart(
                    px.scatter(sc, x=tx, y=ty, color="cluster", hover_name="il",
                               labels={tx: f"{tx} share", ty: f"{ty} share"}),
                    width="stretch")

            cols = st.columns(len(clusters) if clusters else 1)
            for i, (cid, info) in enumerate(sorted(clusters.items())):
                with cols[i % len(cols)]:
                    st.metric(f"Cluster {cid}", f"{info['size']} provinces")
                    st.caption("Topics: " + ", ".join(f"T{t}" for t in info["top_topics"]))
                    st.caption(", ".join(info["members"][:6]) + ("…" if info["size"] > 6 else ""))
            if sil is not None:
                st.caption(f"Silhouette = {sil} (weak but positive structure)")

# =====================================================================
# RQ3
# =====================================================================
with tab3:
    st.header("MP Attention (Co-attention) Network")
    m = metrics("rq3")
    pr = pq("rq3_pagerank")
    comm = pq("rq3_community")

    st.caption("Edge = MPs targeting the same **ministry + province + month** "
               "(attention coordination, independent of RQ1's wording-duplication "
               f"signal). The network is opposition-dominated: {m.get('n_vertices', '?')} "
               f"nodes, {m.get('n_edges', '?')} edges.")

    if pr.empty:
        st.warning("No RQ3 data.")
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("PageRank — most central MPs (Spark)")
            top = pr.sort_values("pr", ascending=False).head(15)
            st.plotly_chart(px.bar(top, x="pr", y="node", orientation="h", color="party",
                            color_discrete_map=PARTY_COLORS, labels={"pr": "PageRank", "node": ""})
                            .update_layout(yaxis={"categoryorder": "total ascending"}, height=480),
                            width="stretch")
        with c2:
            st.subheader("DeepWalk + UMAP embedding (party)")
            st.image(str(FIG / "rq3_embedding_party.png"))
            st.caption(f"Silhouette(party) = {m.get('silhouette_party')} → weak separation; "
                       "party is relatively more dominant than geography but there are no clean clusters.")

        cc1, cc2 = st.columns(2)
        with cc1:
            lv = m.get("louvain", {})
            st.subheader("Communities — Louvain")
            st.metric("ARI (agreement with party)", lv.get("ari"), f"NMI {lv.get('nmi')}")
            for c in lv.get("top_communities", []):
                st.caption(f"• {c['size']} MPs — {c['top_party']} {c['share']*100:.0f}%")
            st.caption(f"ARI {lv.get('ari')} (old identical-summary network 0.46): attention "
                       "coordination is somewhat less party-aligned than wording. "
                       "Spark LPA degenerates on this dense graph → Louvain used.")
        with cc2:
            st.subheader("Cross-party bridge MPs")
            if m.get("top_bridges"):
                bdf = pd.DataFrame(m["top_bridges"]).head(8)
                st.dataframe(bdf[["mp", "party", "province", "bridge_score"]],
                             width="stretch", hide_index=True)
                top3 = "; ".join(f"{r.mp} ({r.party})" for r in bdf.head(3).itertuples())
                st.caption(f"In the attention network the bridges span the party-mixed communities: {top3}.")

st.divider()
st.caption("Apache Spark · Delta Lake · Spark MLlib · Streamlit — STAT 401 Final Project")
