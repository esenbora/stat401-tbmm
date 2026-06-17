"""Silver layer: enrich the Bronze written-question metadata for RQ1/RQ2.

Bronze (``data/bronze/yazili_soru_meta.parquet``) holds the scraped fields with
no party/topic/geography enrichment. This step, run on Spark, produces two
Delta tables under ``data/silver``:

* ``silver_yazili_soru_clean`` — one row per question, with ASCII column names,
  the submitting MP's party + electoral province attached, the calendar year
  parsed out of ``geliş_tarihi``, and the list of provinces *mentioned* in the
  summary (RQ2 attention signal).
* ``silver_mp_party`` — the MP dimension table (name, party, province,
  question count).

Run:  ``.venv/bin/python -m src.analysis.prep``   (from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pyspark.sql import functions as F
from pyspark.sql import types as T

from spark_utils import get_spark, write_delta, attach_sources, SILVER, BRONZE
from analysis._common import (
    build_mp_lookup,
    build_province_pattern,
    extract_province_mentions,
)


def build_silver() -> None:
    spark = get_spark("tbmm-prep")
    attach_sources(spark)
    print("Reading bronze parquet ...")
    bronze = spark.read.parquet(str(BRONZE / "yazili_soru_meta.parquet"))

    # ---- normalise columns to ASCII ------------------------------------
    df = (
        bronze.select(
            F.col("guid"),
            F.col("yasama_yili").cast("int").alias("yasama_yili"),
            F.col("esas_no"),
            F.col("geliş_tarihi").alias("tarih_raw"),
            F.col("özet").alias("ozet"),
            F.col("sahip_il").alias("sahip_il"),
            F.col("sahip_isim").alias("mv"),
            F.col("muhatap_bakanlık").alias("bakanlik"),
            F.col("muhatap_bakan").alias("bakan"),
            F.col("durum"),
        )
        .where(F.col("mv").isNotNull() & (F.trim(F.col("mv")) != ""))
    )

    # ---- year from DD.MM.YYYY ------------------------------------------
    df = df.withColumn(
        "tarih", F.to_date(F.col("tarih_raw"), "dd.MM.yyyy")
    ).withColumn("yil", F.year("tarih"))

    # ---- attach MP party + electoral province (small broadcast join) ---
    lookup = build_mp_lookup()
    mp_rows = [
        (name, rec["party"], rec["party_current"], rec["province"])
        for name, rec in lookup.items()
    ]
    mp_schema = T.StructType([
        T.StructField("mv", T.StringType()),
        T.StructField("party", T.StringType()),
        T.StructField("party_current", T.StringType()),
        T.StructField("mv_province", T.StringType()),
    ])
    mp_df = spark.createDataFrame(mp_rows, mp_schema)
    df = df.join(F.broadcast(mp_df), on="mv", how="left").fillna(
        {"party": "Bilinmiyor", "party_current": "Bilinmiyor", "mv_province": "Bilinmiyor"}
    )

    # ---- province mentions from the summary (RQ2) ----------------------
    pattern = build_province_pattern()

    @F.udf(returnType=T.ArrayType(T.StringType()))
    def mentions_udf(text):  # noqa: ANN001 - Spark UDF
        return extract_province_mentions(text, pattern)

    df = df.withColumn("mentioned_provinces", mentions_udf(F.col("ozet")))
    df = df.withColumn("n_mentions", F.size("mentioned_provinces"))

    df = df.cache()
    total = df.count()
    print(f"Silver rows: {total}")

    write_delta(df, SILVER / "yazili_soru_clean", partition_by=["yil"])
    print(f"Wrote silver_yazili_soru_clean (partitioned by yil)")

    # ---- MP dimension table --------------------------------------------
    mp_dim = (
        df.groupBy("mv", "party", "party_current", "mv_province")
        .agg(F.count("*").alias("soru_sayisi"))
        .orderBy(F.desc("soru_sayisi"))
    )
    write_delta(mp_dim, SILVER / "mp_party")
    print(f"Wrote silver_mp_party ({mp_dim.count()} MPs)")

    # quick sanity prints
    print("\nParty distribution (questions):")
    df.groupBy("party").agg(F.count("*").alias("n")).orderBy(F.desc("n")).show(truncate=False)
    print("Year distribution:")
    df.groupBy("yil").agg(F.count("*").alias("n")).orderBy("yil").show()
    print(f"Questions mentioning >=1 province: {df.where(F.col('n_mentions') > 0).count()}")

    spark.stop()


if __name__ == "__main__":
    build_silver()
