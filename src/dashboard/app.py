"""STAT 401 — TBMM 28th-term Parliamentary Analytics dashboard.

Three tabs (RQ1 / RQ2 / RQ3) over the Spark-MLlib results. Reads the small
parquet snapshots written by ``src/analysis/export_dashboard.py`` and the
metrics JSON written by the RQ scripts — no Spark needed at view time, so the
app starts instantly.

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


def topic_label(words_by_topic: dict[int, list[str]], t: int) -> str:
    w = words_by_topic.get(t, [])
    return f"T{t}: {', '.join(w[:3])}" if w else f"T{t}"


# ----- sidebar -----------------------------------------------------------
with st.sidebar:
    st.title("🇹🇷 TBMM Analytics")
    st.caption("STAT 401 — 28. Dönem Yazılı Soru Önergeleri")
    st.metric("Toplam önerge", "44,484")
    st.metric("Soru veren milletvekili", "307")
    st.metric("Bakanlık / il", "20 / 81")
    st.divider()
    st.caption("Apache Spark · Delta Lake · Spark MLlib · Streamlit")
    if not (DATA / "rq1_ministry_year.parquet").exists():
        st.error("Veri yok — önce `export_dashboard.py` çalıştır.")

st.title("TBMM 28. Dönem — Parlamento Big-Data Analizi")
tab1, tab2, tab3 = st.tabs([
    "📊 Bakanlık × Konu × Parti (RQ1)",
    "🗺️ İl İlgi Haritası (RQ2)",
    "🕸️ Milletvekili Ağı (RQ3)",
])

# =====================================================================
# RQ1
# =====================================================================
with tab1:
    st.header("Bakanlık × Konu × Parti × Zaman")
    m = metrics("rq1")
    my = pq("rq1_ministry_year")
    pm = pq("rq1_party_ministry")

    if my.empty:
        st.warning("RQ1 verisi yok.")
    else:
        my["b"] = my["bakanlik"].map(short_ministry)
        years = sorted(int(y) for y in my["yil"].dropna().unique())
        yr = st.select_slider("Yıl aralığı", options=years, value=(years[0], years[-1]), key="rq1yr")
        myf = my[my["yil"].between(yr[0], yr[1])]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Bakanlığa göre soru hacmi")
            agg = myf.groupby("b")["n"].sum().sort_values(ascending=False).head(12)
            st.plotly_chart(px.bar(agg, orientation="h", labels={"value": "soru", "b": ""})
                            .update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False),
                            width="stretch")
        with c2:
            st.subheader("Yıllara göre seyir (Top 6)")
            top6 = myf.groupby("b")["n"].sum().sort_values(ascending=False).head(6).index
            tl = myf[myf["b"].isin(top6)]
            st.plotly_chart(px.line(tl, x="yil", y="n", color="b", markers=True,
                            labels={"yil": "yıl", "n": "soru", "b": "bakanlık"}),
                            width="stretch")

        if not pm.empty:
            st.subheader("Parti × Bakanlık odağı (satır-normalize)")
            mains = ["CHP", "DEM Parti", "İYİ Parti", "MHP", "AK Parti"]
            sub = pm[pm["party"].isin(mains)].copy()
            sub["b"] = sub["bakanlik"].map(short_ministry)
            piv = sub.pivot_table(index="party", columns="b", values="n", aggfunc="sum").fillna(0)
            piv = piv[piv.sum().sort_values(ascending=False).head(12).index]
            norm = piv.div(piv.sum(axis=1), axis=0)
            st.plotly_chart(px.imshow(norm, aspect="auto", color_continuous_scale="Magma",
                            labels={"color": "pay"}), width="stretch")

        cc1, cc2 = st.columns(2)
        with cc1:
            if m.get("party_clf"):
                clf = m["party_clf"]
                st.subheader("Partiyi metinden tahmin")
                st.metric("Doğruluk (LogReg, TF-IDF)", f"{clf['accuracy']*100:.1f}%",
                          f"baseline {clf['baseline_majority']*100:.1f}%")
                st.caption(f"Sınıflar: {', '.join(clf['classes'])} · F1={clf['f1']:.3f}")
                st.image(str(FIG / "rq1_s3_confusion.png"))
        with cc2:
            if m.get("duplicates"):
                d = m["duplicates"]
                st.subheader("Kopya/koordineli önergeler (MinHash ≥0.8)")
                st.metric("Yakın-kopya çift", f"{d['near_dup_pairs']:,}")
                st.caption(f"{d['cross_mp_pairs']:,} farklı mv · {d['cross_party_pairs']:,} farklı parti")
                cp = pq("rq1_dup_cross_party")
                if not cp.empty:
                    cp = cp.sort_values("pairs", ascending=False).head(8)
                    cp["çift"] = cp["party_a"] + " ↔ " + cp["party_b"]
                    st.plotly_chart(px.bar(cp, x="pairs", y="çift", orientation="h")
                                    .update_layout(yaxis={"categoryorder": "total ascending"}),
                                    width="stretch")

        if m.get("lda_topics"):
            with st.expander("LDA konuları (12) — en sık kelimeler"):
                st.dataframe(pd.DataFrame([
                    {"Konu": f"T{t['topic']}", "Kelimeler": ", ".join(t["words"])}
                    for t in m["lda_topics"]]), width="stretch", hide_index=True)

# =====================================================================
# RQ2
# =====================================================================
with tab2:
    st.header("İl Bazlı Parlamento İlgisi")
    m = metrics("rq2")
    men = pq("rq2_province_mentions")
    corr = pq("rq2_province_correlates")
    clus = pq("rq2_province_cluster")
    gj = geojson()

    if men.empty:
        st.warning("RQ2 verisi yok.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("İl ilgi haritası (anılma sayısı)")
            if gj is not None:
                fig = px.choropleth(men, geojson=gj, locations="il", featureidkey="properties.name",
                                    color="mentions", color_continuous_scale="YlGnBu",
                                    labels={"mentions": "anılma"})
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=420)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Harita için geojson yok — bar grafiğe bakın.")
        with c2:
            st.subheader("En çok anılan iller")
            top = men.sort_values("mentions", ascending=False).head(15)
            st.plotly_chart(px.bar(top, x="mentions", y="il", orientation="h")
                            .update_layout(yaxis={"categoryorder": "total ascending"}, height=420),
                            width="stretch")

        if not corr.empty:
            st.subheader("İlgi nüfusa göre — hedefleme")
            cp = m.get("correlation_pearson", {})
            r = cp.get("attention~population", 0)
            cc1, cc2 = st.columns([3, 2])
            with cc1:
                fig = px.scatter(corr, x="population", y="attention", hover_name="il",
                                 size="att_per_100k", labels={"population": "nüfus", "attention": "anılma"})
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Pearson r(ilgi, nüfus) = {r}  ·  nüfus varyansın ~%{r**2*100:.0f}'ini açıklıyor")
            with cc2:
                st.markdown("**100k kişi başına en çok ilgi**")
                over = corr.sort_values("att_per_100k", ascending=False).head(8)
                st.dataframe(over[["il", "attention", "att_per_100k"]].round(1),
                             width="stretch", hide_index=True)

        if not clus.empty:
            st.subheader("K-Means++ ilgi profilleri")
            sil = m.get("kmeans_silhouette")
            clusters = m.get("kmeans_clusters", {})
            cols = st.columns(len(clusters) if clusters else 1)
            for i, (cid, info) in enumerate(sorted(clusters.items())):
                with cols[i % len(cols)]:
                    st.metric(f"Küme {cid}", f"{info['size']} il")
                    st.caption("Konular: " + ", ".join(f"T{t}" for t in info["top_topics"]))
                    st.caption(", ".join(info["members"][:6]) + ("…" if info["size"] > 6 else ""))
            if sil is not None:
                st.caption(f"Silhouette = {sil} (zayıf ama pozitif yapı)")

# =====================================================================
# RQ3
# =====================================================================
with tab3:
    st.header("Milletvekili Ortak-İmza Ağı")
    m = metrics("rq3")
    pr = pq("rq3_pagerank")
    comm = pq("rq3_community")

    st.caption("⚠️ Yazılı sorular tek imzalı → 'ortak imza' = aynı özet proxy'si. "
               "Ağ muhalefet ağırlıklı (AK Parti yazılı soru vermiyor): "
               f"{m.get('n_vertices', '?')} düğüm, {m.get('n_edges', '?')} kenar.")

    if pr.empty:
        st.warning("RQ3 verisi yok.")
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("PageRank — en merkezi mv'ler (Spark)")
            top = pr.sort_values("pr", ascending=False).head(15)
            st.plotly_chart(px.bar(top, x="pr", y="node", orientation="h", color="party",
                            color_discrete_map=PARTY_COLORS, labels={"pr": "PageRank", "node": ""})
                            .update_layout(yaxis={"categoryorder": "total ascending"}, height=480),
                            width="stretch")
        with c2:
            st.subheader("DeepWalk + UMAP gömme (parti)")
            st.image(str(FIG / "rq3_embedding_party.png"))
            st.caption(f"Silhouette(parti) = {m.get('silhouette_party')} → ayrışma zayıf; "
                       "parti, coğrafyadan göreceli daha baskın ama temiz küme yok.")

        cc1, cc2 = st.columns(2)
        with cc1:
            lv = m.get("louvain", {})
            st.subheader("Topluluklar — Louvain")
            st.metric("ARI (parti ile uyum)", lv.get("ari"), f"NMI {lv.get('nmi')}")
            for c in lv.get("top_communities", []):
                st.caption(f"• {c['size']} mv — {c['top_party']} %{c['share']*100:.0f}")
            st.caption("Not: Spark LPA bu klik-yoğun grafta dejenere (her şeyi tek topluluğa katlıyor) "
                       "→ anlamlı yapı için Louvain.")
        with cc2:
            st.subheader("Cross-parti köprü mv'ler")
            if m.get("top_bridges"):
                bdf = pd.DataFrame(m["top_bridges"]).head(8)
                st.dataframe(bdf[["mp", "party", "province", "bridge_score"]],
                             width="stretch", hide_index=True)
                st.caption("Sezgin Tanrıkulu (CHP/Diyarbakır) baskın köprü — CHP↔DEM kanalı.")

st.divider()
st.caption("Apache Spark · Delta Lake · Spark MLlib · Streamlit — STAT 401 Final Project")
