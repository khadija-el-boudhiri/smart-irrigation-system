# Smart Irrigation MLOps
An end-to-end Machine Learning pipeline and API for smart irrigation predictions.

## Project Overview
This project provides a robust MLOps pipeline for a smart irrigation system, utilizing machine learning to predict whether irrigation is needed based on environmental factors. It integrates data versioning, model tracking, and containerized deployment to ensure reproducible and reliable predictions. A comprehensive monitoring stack oversees API performance and model health in production.

## Architecture

```text
    [Data] ──> [DVC] ──> [Training] ──> [MLflow] ──> [API] ──> [Client]
                               |                       |
                               v                       v
                         [Prometheus] <──────── [Grafana]
```

## Quick Start
To set up the project locally, open a Windows CMD prompt and run the following commands:
```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
docker-compose up -d
```

## API Reference

The API runs on port 5000 via Flask.

| Method | Endpoint | Description | Example Request | Example Response |
|--------|----------|-------------|-----------------|------------------|
| GET | `/` | API root | N/A | `{"message": "Smart Irrigation API is running"}` |
| GET | `/health` | Health check | N/A | `{"status": "ok"}` |
| POST | `/predict` | Predict irrigation need | `{"soil_pct": 35.2, "temperature": 28.0, "pressure": 9984.5, "altitude": 12.1}` | `{"needs_irrigation": true}` |

**Input Validation Requirements:**
- `soil_pct`: 0 - 100
- `temperature`: 10 - 42
- `pressure`: 9780 - 10120
- `altitude`: 0 - 500

## ML Pipeline
Three models were evaluated during training: Logistic Regression (best, CV F1=0.848), Random Forest, and XGBoost. Data is tracked with DVC (`data/processed/features_ready.csv`) and model tracking is handled via MLflow (`sqlite:///mlflow.db`). The best model is registered as `PlantWaterModel` with the alias `production`.

To retrain and promote a new model, use the following commands:
```cmd
dvc pull
python src/train.py
```
*(After training, the best model can be promoted to the "production" alias in the MLflow UI or via your deployment scripts).*

## Monitoring
The infrastructure includes Prometheus and Grafana for system and model monitoring (including node-exporter).
- **Grafana**: Accessible at `http://localhost:3000`. Features the "Smart Irrigation MLOps Overview" dashboard.
- **Prometheus**: Accessible at `http://localhost:9090`. Configured with alert rules for `ModelLatencyHigh`, `APIErrorRate`, and `MLflowDown`.

## CI/CD
The project uses a Jenkins pipeline with 8 stages:
1. **Checkout**: Pulls the latest code from the repository.
2. **Lint & test**: Runs code quality checks and unit tests.
3. **DVC pull**: Retrieves the latest tracked data.
4. **Train**: Trains the models and logs metrics to MLflow.
5. **Evaluate**: Evaluates the model performance.
6. **Build Docker image**: Containerizes the application.
7. **Deploy**: Deploys the services.
8. **Post**: Performs post-deployment steps.

**Pipeline Parameters:**
- `DEPLOY_ENV` (staging/production)
- `SKIP_TRAIN` (boolean)

**Required Credentials:**
- `DVC_ACCESS_KEY`
- `DOCKER_REGISTRY_CREDENTIALS`
- `SLACK_WEBHOOK`

## Project Structure
```text
smart-irrigation-system/
├── api/
│   └── app.py
├── data/
│   └── processed/
├── docker-compose.yml
├── Jenkinsfile
├── pytest.ini
├── README.md
├── requirements.txt
├── src/
│   └── train.py
└── tests/
    ├── conftest.py
    └── test_api.py
```

## License
MIT
