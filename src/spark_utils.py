"""PySpark + Delta Lake bootstrap for Bronze/Silver/Gold pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def _pin_python() -> None:
    """Force Spark workers + driver onto *this* interpreter (the venv).

    Without this, Spark picks the first ``python3`` on PATH for UDF workers,
    which on this machine is system Python 3.14 while the driver is the venv's
    3.11 — Spark refuses to run with mismatched minor versions.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def _ensure_java_home() -> None:
    """Point JAVA_HOME at the Homebrew OpenJDK 17 if the env var is unset.

    PySpark 4.x needs a JVM (Java 17/21). On this Mac, Java is installed via
    `brew install openjdk@17` but not symlinked into the system, so we resolve
    it explicitly instead of relying on a shell profile.
    """
    if os.environ.get("JAVA_HOME"):
        return
    for candidate in (
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
        "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
    ):
        if Path(candidate).exists():
            os.environ["JAVA_HOME"] = candidate
            return


def get_spark(app_name: str = "tbmm-stat401", memory: str = "4g") -> SparkSession:
    """Create a local Spark session with Delta Lake enabled.

    Delta jars are resolved via ``configure_spark_with_delta_pip`` so the Maven
    coordinate always matches the installed ``delta-spark`` wheel (Scala 2.13 /
    Delta 4.x for PySpark 4.x) instead of being hard-coded.
    """
    _ensure_java_home()
    _pin_python()
    from delta import configure_spark_with_delta_pip  # needs delta on path

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def attach_sources(spark: SparkSession) -> None:
    """Ship ``src/`` to Spark workers so UDFs can ``import analysis._common``.

    In local mode each Python worker is a separate process that does not inherit
    the driver's ``sys.path``; without this, any UDF referencing project modules
    fails with ``ModuleNotFoundError``. We zip the ``src`` tree once and register
    it via ``addPyFile`` (idempotent within a session).
    """
    import zipfile
    import tempfile

    src_dir = Path(__file__).resolve().parent  # .../src
    zip_path = Path(tempfile.gettempdir()) / "tbmm_src.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for py in src_dir.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            zf.write(py, py.relative_to(src_dir))
    spark.sparkContext.addPyFile(str(zip_path))


def write_delta(df, path: str | Path, mode: str = "overwrite", partition_by: list[str] | None = None):
    w = df.write.format("delta").mode(mode)
    if partition_by:
        w = w.partitionBy(*partition_by)
    w.save(str(path))


def read_delta(spark: SparkSession, path: str | Path):
    return spark.read.format("delta").load(str(path))


# Schema layer conventions
BRONZE = Path("data/bronze")
SILVER = Path("data/silver")
GOLD = Path("data/gold")

TABLES = {
    "bronze_yazili_soru_meta": BRONZE / "yazili_soru_meta",
    "bronze_yazili_soru_pdf": BRONZE / "yazili_soru_pdf",     # binary
    "bronze_mp_profile": BRONZE / "mp_profile",
    "bronze_tutanak": BRONZE / "tutanak",
    "silver_ocr_text": SILVER / "ocr_text",                    # (guid, page, text, conf)
    "silver_yazili_soru_clean": SILVER / "yazili_soru_clean",
    "silver_mp_party": SILVER / "mp_party",
    "silver_cosign_edges": SILVER / "cosign_edges",
    "gold_ministry_topic_party_year": GOLD / "ministry_topic_party_year",
    "gold_province_topic": GOLD / "province_topic",
    "gold_mp_network": GOLD / "mp_network",
}


if __name__ == "__main__":
    s = get_spark()
    s.sql("SELECT 1 AS ok").show()
    print("Spark + Delta ready")
    s.stop()
