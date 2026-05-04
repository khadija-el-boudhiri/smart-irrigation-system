# Data card — Smart Irrigation Sensor Dataset

## Dataset name

Smart Irrigation Sensor Dataset

## Version

1.0

## Created by

DataOps person

## Format

CSV

## Rows

20,000

## Columns

`soil_pct`, `temperature`, `pressure`, `altitude`, `status`

## Feature descriptions

| Column | Description |
|--------|-------------|
| `soil_pct` | Soil moisture as a percentage of field capacity (0–100%). |
| `temperature` | Ambient air temperature in degrees Celsius. |
| `pressure` | Atmospheric pressure in hectopascals (hPa). |
| `altitude` | Field elevation above sea level in meters. |
| `status` | Binary irrigation decision label derived from the scenario (see Target column). |

## Value ranges

These are the **valid physical ranges** enforced during preprocessing (`validate_ranges` in `src/preprocess.py`); rows outside these bounds are dropped before training.

| Feature | Valid range | Unit |
|---------|-------------|------|
| `soil_pct` | 0–100 | % |
| `temperature` | 10–42 | °C |
| `pressure` | 9,780–10,120 | hPa |
| `altitude` | 0–500 | m |
| `status` | 0 or 1 | — (binary) |

## Target column

**`status`** — `0` means no irrigation needed; `1` means irrigation needed.

## Class balance

Approximately **55%** class 0 and **45%** class 1 (exact counts may shift slightly after dropping invalid or missing rows).

## Generation method

Synthetically generated using a logistic scoring function that combines all four input features, with **4% label noise** to simulate real sensor uncertainty.

## Known limitations

Synthetic data may not capture all real-world sensor drift, calibration errors, or rare edge cases seen in production deployments.

## How to update the dataset

1. Regenerate or refresh the processed CSV by running:

   ```bash
   python src/spark_etl.py
   ```

2. Re-track the artifact with DVC:

   ```bash
   dvc add data/processed/features_ready.csv
   ```

3. Push the updated data (and `.dvc` metadata) to remote storage:

   ```bash
   dvc push
   ```

## Pipeline dependency

This dataset (typically `data/processed/features_ready.csv`) is the **input** to the ZenML training pipeline: it is loaded in **`load_data_step`** and consumed downstream for model training and promotion.
