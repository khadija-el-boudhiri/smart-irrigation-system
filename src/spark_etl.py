"""
Batch ETL with Apache Spark (local mode by default).

Use this when you add more raw CSV files or grow the dataset; training still reads
one CSV via TRAIN_DATA_PATH (see README).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

try:
    from src.schema import MODEL_FEATURES, TARGET_COLUMN
except ModuleNotFoundError:
    from schema import MODEL_FEATURES, TARGET_COLUMN


def build_spark() -> SparkSession:
    master = os.environ.get("SPARK_MASTER", "local[*]")
    try:
        return (
            SparkSession.builder.appName("smart_irrigation_etl")
            .master(master)
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
    except Exception as e:
        raise RuntimeError(
            "Spark needs a JDK (Java 11+) and JAVA_HOME set. "
            "Install Temurin JDK, restart the terminal, then retry."
        ) from e


def run_etl(input_path: str, output_csv: str) -> int:
    required = MODEL_FEATURES + [TARGET_COLUMN]
    spark = build_spark()
    try:
        df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Found: {df.columns}")

        for col in MODEL_FEATURES:
            df = df.withColumn(col, F.col(col).cast("double"))
        df = df.withColumn(TARGET_COLUMN, F.col(TARGET_COLUMN).cast("int"))

        df = df.select(*required).dropna()
        count = df.count()

        out_dir = tempfile.mkdtemp(prefix="spark_etl_", dir=os.path.dirname(output_csv) or ".")
        try:
            df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_dir)
            parts = sorted(glob.glob(os.path.join(out_dir, "part-*.csv")))
            if not parts:
                raise RuntimeError("Spark produced no part-*.csv file.")
            shutil.copyfile(parts[0], output_csv)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

        return int(count)
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark ETL for irrigation training data")
    parser.add_argument(
        "--input",
        default="data/processed/features_ready.csv",
        help="CSV path or glob, e.g. data/raw/*.csv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/features_spark.csv",
        help="Single CSV path for downstream pandas training.",
    )
    args = parser.parse_args()
    n = run_etl(args.input, args.output)
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
