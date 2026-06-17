"""Export the Gold Delta tables to small single-file parquet for the dashboard.

The Streamlit app must start fast and must not spin up Spark. Reading Delta
directories with pandas directly is unsafe (overwrite leaves stale part files
until vacuum), so we read each table *through Spark* (correct current snapshot),
shrink the heavy ones to the aggregates the dashboard needs, and write clean
parquet to ``src/dashboard/data/``.

Run after the three RQ scripts:
    .venv/bin/python src/analysis/export_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pyspark.sql import functions as F
from spark_utils import get_spark, GOLD

OUT = ROOT / "src" / "dashboard" / "data"


def dump(df, name: str) -> None:
    pdf = df.toPandas()
    pdf.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"  {name}.parquet  ({len(pdf)} rows)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spark = get_spark("tbmm-export")
    g = lambda t: spark.read.format("delta").load(str(GOLD / t))  # noqa: E731
    print("Exporting gold -> src/dashboard/data/ ...")

    # RQ1
    dump(g("ministry_year"), "rq1_ministry_year")
    dump(g("party_ministry"), "rq1_party_ministry")
    dump(g("ministry_topic"), "rq1_ministry_topic")
    # duplicate_pairs is 100k+ rows: keep only the cross-party aggregate
    dup = g("duplicate_pairs")
    cross = (dup.where(F.col("party_a") != F.col("party_b"))
             .groupBy("party_a", "party_b").agg(F.count("*").alias("pairs")))
    dump(cross, "rq1_dup_cross_party")

    # RQ2
    dump(g("province_mentions"), "rq2_province_mentions")
    dump(g("province_topic"), "rq2_province_topic")
    dump(g("province_cluster"), "rq2_province_cluster")
    dump(g("province_correlates"), "rq2_province_correlates")

    # RQ3
    dump(g("mp_pagerank"), "rq3_pagerank")
    dump(g("mp_community"), "rq3_community")

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
