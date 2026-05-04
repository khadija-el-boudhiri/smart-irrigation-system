# Smart Irrigation System

College MLOps project for predicting if irrigation is needed based on sensor values.

## Quick file map

- `src/schema.py`: shared names used across the project (features, target, experiment/model names).
- `src/preprocess.py`: checks dataset columns and splits train/test (raw features).
- `src/train_models.py`: wraps each model in a scaler + classifier pipeline, trains, evaluates, logs to MLflow (so the API gets the same preprocessing as training).
- `src/evaluate.py`: helper metrics (accuracy, f1, confusion matrix, report).
- `src/promote_model.py`: picks best MLflow run and assigns alias (`staging` or `production`).
- `src/spark_etl.py`: optional Spark batch job (read CSV/glob, validate schema, drop bad rows, write one CSV for training).
- `api/app.py`: Flask API for prediction.

## Who handles what

- DataOps: dataset quality + schema checks (`src/preprocess.py`, optional `src/spark_etl.py`)
- MLOps: training, evaluation, promotion (`src/train_models.py`, `src/evaluate.py`, `src/promote_model.py`)
- DevOps: API runtime + deployment (`api/app.py`)

## Run order

From the project root:

1. `pip install -r requirements.txt`
2. (Optional Spark ETL, e.g. more or new CSV files) Install a **JDK 11+** (e.g. Eclipse Temurin), set **JAVA_HOME**, then:  
   `python src/spark_etl.py --input "data/raw/*.csv" --output data/processed/features_spark.csv`  
   Then train on that file:  
   `set TRAIN_DATA_PATH=data/processed/features_spark.csv` (Windows) or `export TRAIN_DATA_PATH=...` (Linux/macOS)  
   before training (step 4).
3. `python src/preprocess.py` (sanity check on whatever `TRAIN_DATA_PATH` points to, default is `features_ready.csv`)
4. `python src/train_models.py`
5. `python src/promote_model.py --target production`
6. `python api/app.py`

Test request:

`curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"soil_pct\":35.2,\"temperature\":28.0,\"pressure\":9984.5,\"altitude\":12.1}"`

## Data note

- Default training file: `data/processed/features_ready.csv` (override with `TRAIN_DATA_PATH`).
- Spark output example: `data/processed/features_spark.csv` (gitignored; regenerate locally).
- DVC: pipeline in `dvc.yaml` (`etl` writes `data/processed/features_ready.csv`, then `validate` runs `python src/preprocess.py`). Run `dvc repro --dry` or `dvc repro` after JDK 11+ is installed for Spark ETL.
