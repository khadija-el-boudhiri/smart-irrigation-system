# Smart Irrigation System

College MLOps project for predicting if irrigation is needed based on sensor values.

## Quick file map

- `src/schema.py`: shared names used across the project (features, target, experiment/model names).
- `src/preprocess.py`: checks dataset columns and splits train/test (raw features).
- `src/train_models.py`: wraps each model in a scaler + classifier pipeline, trains, evaluates, logs to MLflow (so the API gets the same preprocessing as training).
- `src/evaluate.py`: helper metrics (accuracy, f1, confusion matrix, report).
- `src/promote_model.py`: picks best MLflow run and assigns alias (`staging` or `production`).
- `api/app.py`: Flask API for prediction.

## Who handles what

- DataOps: dataset quality + schema checks (`src/preprocess.py`)
- MLOps: training, evaluation, promotion (`src/train_models.py`, `src/evaluate.py`, `src/promote_model.py`)
- DevOps: API runtime + deployment (`api/app.py`)

## Run order

From the project root:

1. `pip install -r requirements.txt`
2. `python src/preprocess.py`
3. `python src/train_models.py`
4. `python src/promote_model.py --target production`
5. `python api/app.py`

Test request:

`curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"soil_pct\":35.2,\"temperature\":28.0,\"pressure\":9984.5,\"altitude\":12.1}"`

## Data note

- Active dataset: `data/processed/features_ready.csv`
- DVC pointer used: `data/processed/features_ready.csv.dvc`
