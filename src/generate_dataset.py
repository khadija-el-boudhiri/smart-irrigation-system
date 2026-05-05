"""
Synthetic generation for `features_ready.csv` (see `data/data_card.md`).
Uses only NumPy and Pandas — no sklearn, Spark, or MLflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.schema import MODEL_FEATURES, TARGET_COLUMN
except ModuleNotFoundError:
    from schema import MODEL_FEATURES, TARGET_COLUMN

DEFAULT_N = 20_000
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("data/processed/features_ready.csv")
LABEL_NOISE_FRACTION = 0.04
LOGIT_SCALE = 3.2


def generate_dataset(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Build the synthetic sensor dataset, round feature floats to 4 decimals,
    write CSV with columns in schema order, and return the DataFrame.
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    soil_pct = rng.uniform(10.0, 90.0, size=n)
    temperature = rng.uniform(10.0, 42.0, size=n)
    pressure = rng.uniform(9780.0, 10120.0, size=n)
    altitude = rng.uniform(0.0, 500.0, size=n)

    soil_score = (40.0 - soil_pct) / 30.0
    temp_score = (temperature - 22.0) / 15.0
    pres_score = (9960.0 - pressure) / 120.0
    alt_score = (altitude - 250.0) / 300.0

    logit = (
        0.55 * soil_score
        + 0.35 * temp_score
        + 0.07 * pres_score
        + 0.03 * alt_score
    )
    logit = logit + rng.normal(0.0, 0.22, size=n)

    prob = 1.0 / (1.0 + np.exp(-logit * LOGIT_SCALE))
    status = (prob > 0.5).astype(np.int64)

    n_flip = int(round(LABEL_NOISE_FRACTION * n))
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    status[flip_idx] = 1 - status[flip_idx]

    df = pd.DataFrame(
        {
            "soil_pct": soil_pct,
            "temperature": temperature,
            "pressure": pressure,
            "altitude": altitude,
            TARGET_COLUMN: status,
        }
    )

    for col in MODEL_FEATURES:
        df[col] = df[col].round(4)

    column_order = MODEL_FEATURES + [TARGET_COLUMN]
    df = df[column_order]

    df.to_csv(output_path, index=False)
    return df


def _print_summary(df: pd.DataFrame, output_path: Path) -> None:
    n = len(df)
    c0 = int((df[TARGET_COLUMN] == 0).sum())
    c1 = int((df[TARGET_COLUMN] == 1).sum())
    p0 = 100.0 * c0 / n if n else 0.0
    p1 = 100.0 * c1 / n if n else 0.0

    print(f"Rows: {n}")
    print(f"Class 0: {c0} ({p0:.2f}%)")
    print(f"Class 1: {c1} ({p1:.2f}%)")
    print(f"Saved dataset to: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate synthetic features_ready.csv (see data/data_card.md)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    df = generate_dataset(n=DEFAULT_N, seed=DEFAULT_SEED, output_path=args.output)
    _print_summary(df, Path(args.output))


if __name__ == "__main__":
    main()
