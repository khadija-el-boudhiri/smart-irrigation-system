# Irrigation Intelligente — Système de prédiction d'arrosage

## Description

Système intelligent qui prédit si une plante a besoin d'être arrosée
en fonction de 4 mesures capteurs : humidité du sol, température,
pression atmosphérique et altitude.

Le modèle de machine learning (Régression Logistique, F1=0.848) est
entraîné sur des données réelles et exposé via une API web avec une
interface utilisateur .

---

## Équipe projet

| Rôle | Nom |
|------|-----|
| MLOps | Salma Nhira |
| DataOps | Nouaman Biba |
| DevOps | Khadija El Boudhiri |

### Responsabilités

**MLOps — Salma Nhira**
- Entraînement des modèles de machine learning (Régression Logistique, Random Forest, XGBoost)
- Suivi des expériences avec MLflow
- Évaluation et sélection du meilleur modèle (F1 = 0.848)
- Enregistrement du modèle dans le registre MLflow (PlantWaterModel@production)
- Promotion automatique du meilleur modèle via src/promote_model.py

**DataOps — Nouaman Biba**
- Collecte et validation des données capteurs
- Pipeline de prétraitement des données (src/preprocess.py)
- Versionnement des données avec DVC (data/processed/features_ready.csv)
- Définition du schéma et des règles de validation (src/schema.py)
- Tests de qualité des données (26 tests automatisés)

**DevOps — Khadija El Boudhiri**
- Développement de l'API Flask (api/app.py)
- Interface utilisateur web  (api/index.html)
- Containerisation Docker (Dockerfile.api, docker-compose.yml)
- Pipeline CI/CD Jenkins (Jenkinsfile — 8 étapes)
- Configuration du monitoring Prometheus + Grafana
- Gestion Git, branches et déploiement

---

## Démarrage rapide (Windows)

### Option A — Double-clic (le plus simple)

1. Décompressez le ZIP dans un dossier de votre choix
2. Ouvrez un terminal CMD dans ce dossier et créez l'environnement :

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_api.txt
```

3. Double-cliquez sur le fichier `start.bat`
4. Ouvrez votre navigateur sur : **http://127.0.0.1:5000/ui**

C'est tout. L'interface s'affiche.

### Option B — Terminal manuel

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_api.txt
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
set MLFLOW_MODEL_URI=runs:/05dbc64a0c2e4236bf2e2f8c82f61f04/model
python api\app.py
```

Puis ouvrez : http://127.0.0.1:5000/ui

---

## Prérequis

- Python 3.11 installé — https://www.python.org/downloads/
- Connexion internet (pour installer les dépendances)
- Windows 10 ou supérieur

---

## Architecture
Données capteurs (CSV)
│
▼
[DVC] Versionnement
│
▼
[src/preprocess.py] Nettoyage & validation
│
▼
[src/train_models.py] Entraînement
Régression Logistique │ Random Forest │ XGBoost
│
▼
[MLflow] Suivi des expériences + Registre modèle
PlantWaterModel@production
│
▼
[api/app.py] API Flask — port 5000
│
▼
[api/index.html] Interface web 
│
▼
Utilisateur
── Monitoring ──────────────────────────
Prometheus (port 9090) → Grafana (port 3000)
Alertes : ModelLatencyHigh, APIErrorRate, MLflowDown

---

## Utilisation de l'interface

1. Réglez les 4 curseurs selon les mesures de votre capteur
2. Cliquez sur **"Lancer l'analyse"**
3. Le résultat s'affiche immédiatement :
   - 🟠 **Arrosage nécessaire** — le sol est trop sec
   - 🟢 **Pas d'arrosage nécessaire** — le sol est suffisamment humide
4. L'historique des 6 dernières analyses s'affiche en bas

---

## Tester l'API manuellement

Dans un second terminal (avec .venv activé) :

Vérification santé :
```cmd
curl http://127.0.0.1:5000/health
```
Réponse attendue : `{"status": "ok"}`

Test prédiction :
```cmd
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"soil_pct\":35.2,\"temperature\":28.0,\"pressure\":9984.5,\"altitude\":12.1}"
```
Réponse attendue : `{"needs_irrigation": true}` ou `{"needs_irrigation": false}`

---

## Lancer les tests automatisés

```cmd
pip install pytest
pytest tests/ -v
```
Résultat attendu : **26 passed**

---

## Structure du projet
smart-irrigation-system/
├── api/
│   ├── app.py              ← API Flask principale
│   └── index.html          ← Interface web 
├── src/
│   ├── train_models.py     ← Entraînement des 3 modèles
│   ├── promote_model.py    ← Enregistrement du meilleur modèle
│   ├── preprocess.py       ← Nettoyage des données
│   ├── schema.py           ← Définition des champs
│   ├── model_training.py   ← Pipelines sklearn
│   └── mlflow_config.py    ← Configuration MLflow
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_evaluate.py
│   └── test_preprocess.py
├── data/
│   └── processed/
│       └── features_ready.csv
├── grafana/
│   └── provisioning/
├── Dockerfile.api
├── docker-compose.yml
├── Jenkinsfile
├── prometheus.yml
├── alerts.yml
├── dvc.yaml
├── .env.example
├── requirements.txt        ← Dépendances complètes (ML, tests, dev)
├── requirements_api.txt    ← Dépendances légères (API Flask)
├── pytest.ini
├── start.bat               ← Lancement en un double-clic
└── README.md

---

## Référence API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | / | Statut de l'API |
| GET | /health | Vérification santé |
| GET | /ui | Interface |
| POST | /predict | Prédiction d'arrosage |

Champs acceptés par POST /predict :

| Champ | Type | Min | Max | Unité |
|-------|------|-----|-----|-------|
| soil_pct | float | 0 | 100 | % |
| temperature | float | 10 | 42 | °C |
| pressure | float | 9780 | 10120 | hPa |
| altitude | float | 0 | 500 | m |

---

## Modèle de machine learning

- 3 modèles entraînés : Régression Logistique, Random Forest, XGBoost
- Meilleur modèle : Régression Logistique (F1 = 0.848)
- Enregistré dans MLflow sous le nom PlantWaterModel
- Suivi des expériences via base SQLite locale (mlflow.db)

Pour réentraîner depuis zéro (optionnel) :
```cmd
python src/train_models.py
python src/promote_model.py
```

---

## CI/CD — Jenkins

Le Jenkinsfile définit 8 étapes :

| Étape | Description |
|-------|-------------|
| Checkout | Clonage du dépôt |
| Lint & Test | flake8 + pytest |
| DVC Pull | Récupération des données |
| Train | Entraînement des modèles |
| Evaluate | Promotion du meilleur modèle |
| Build Docker Image | Construction et push de l'image |
| Deploy | Déploiement via docker compose |
| Post | Archivage + notification Slack |

Credentials Jenkins requis : `DVC_ACCESS_KEY`, `DOCKER_REGISTRY_CREDENTIALS`, `SLACK_WEBHOOK`

---

## Monitoring (optionnel — nécessite Docker Desktop)

```cmd
docker compose up -d
```

| Service | URL | Identifiants |
|---------|-----|--------------|
| Interface web | http://localhost:5000/ui | — |
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | — |
| MLflow UI | http://localhost:5001 | — |

---

## Variables d'environnement

Copiez `.env.example` vers `.env` :
```cmd
copy .env.example .env
```

| Variable | Valeur par défaut |
|----------|-------------------|
| MLFLOW_TRACKING_URI | sqlite:///mlflow.db |
| MLFLOW_MODEL_URI | runs:/05dbc64a0c2e4236bf2e2f8c82f61f04/model |
| MODEL_NAME | PlantWaterModel |
| GF_SECURITY_ADMIN_PASSWORD | changeme_strong_password |

---

## Licence

IASD 2026
