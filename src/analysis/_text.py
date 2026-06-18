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


_TR_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def tr_lower(text):  # noqa: ANN001
    """Turkish lower-case + ASCII fold.

    35% of the corpus is OCR'd and frequently drops diacritics ('çocuk' ->
    'cocuk', 'kadın' -> 'kadin') while the text-layer 65% keeps them. Folding
    BOTH to ASCII merges the variants into one token so topic models and
    duplicate detection don't split on OCR noise. (Province *mention* detection
    uses a separate, diacritic-preserving path in _common, so geography stays
    accurate.)"""
    if text is None:
        return None
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return text.translate(_TR_FOLD)


# Stop words folded the same way so they match the folded tokens.
_STOP_FOLDED = sorted({
    (s.replace("İ", "i").replace("I", "ı").lower()).translate(_TR_FOLD)
    for s in TURKISH_STOPWORDS
})


def tokenize(df, in_col: str = "ozet", out_col: str = "toks"):
    """Add a filtered token array column (Turkish lower + ASCII fold + stopwords)."""
    lower_udf = F.udf(tr_lower, T.StringType())
    df = df.withColumn("_lc", lower_udf(F.col(in_col)))
    tok = RegexTokenizer(inputCol="_lc", outputCol="_toks_raw",
                         pattern=r"[^a-z]+", minTokenLength=3, toLowercase=False)
    sw = StopWordsRemover(inputCol="_toks_raw", outputCol=out_col,
                          stopWords=_STOP_FOLDED)
    return sw.transform(tok.transform(df)).drop("_lc", "_toks_raw")
