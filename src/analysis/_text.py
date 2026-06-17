"""Shared Spark text-processing helpers for RQ1 / RQ2.

Kept separate from ``_common`` (pure Python) because these touch PySpark. The
UDF is built lazily inside :func:`tokenize` so importing this module never
requires an active Spark session.
"""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

from analysis._common import TURKISH_STOPWORDS


def tr_lower(text):  # noqa: ANN001
    """Turkish-aware lower-casing. Java's toLowerCase mangles İ -> 'i̇' (adds a
    combining dot) which the tokenizer then splits ('İzmir' -> 'zmir'); map the
    dotted/undotted I in Python first."""
    if text is None:
        return None
    return text.replace("İ", "i").replace("I", "ı").lower()


def tokenize(df, in_col: str = "ozet", out_col: str = "toks"):
    """Add a filtered token array column (Turkish lower + tokenizer + stopwords)."""
    lower_udf = F.udf(tr_lower, T.StringType())
    df = df.withColumn("_lc", lower_udf(F.col(in_col)))
    tok = RegexTokenizer(inputCol="_lc", outputCol="_toks_raw",
                         pattern=r"[^a-zçğıöşü]+", minTokenLength=3, toLowercase=False)
    sw = StopWordsRemover(inputCol="_toks_raw", outputCol=out_col,
                          stopWords=TURKISH_STOPWORDS)
    return sw.transform(tok.transform(df)).drop("_lc", "_toks_raw")
