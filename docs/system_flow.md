# Smart Irrigation System — System Flow

## 1. Overview
This system automates irrigation decisions using IoT data and machine learning.

---

## 2. Data Flow

### Step 1 — Data Collection (DataOps)
- IoT sensors generate data:
  - soil_moisture
  - temperature
  - humidity

- Data is stored in:
  data/raw/

---

### Step 2 — Data Processing (DataOps)
- Data is cleaned and validated
- Output stored in:
  data/processed/clean_data.csv

- Requirements:
  - No missing values
  - Valid ranges (0–100 for soil moisture)

---

### Step 3 — Model Training (MLOps)
- Processed data is used to train a model
- Model is saved to:
  mlops/models/model.pkl

---

### Step 4 — API Prediction (DevOps)
- API receives input:
  - soil_pct
  - temperature
  - pressure
  - altitude

- API loads model
- API returns:
  - needs_irrigation = true/false

---

### Step 5 — Monitoring (Future)
- Monitor predictions
- Detect model drift
- Trigger retraining if needed