# API Contract — Smart Irrigation System

## Base URL
http://localhost:5000

---

## 1. Health Check

### Endpoint
GET /

### Response
{
  "message": "Smart Irrigation API is running"
}

---

## 2. Prediction Endpoint

### Endpoint
POST /predict

### Request Body (JSON)
{
  "soil_pct": float,
  "temperature": float,
  "pressure": float,
  "altitude": float
}

### Example Request
{
  "soil_pct": 35.2,
  "temperature": 28.0,
  "pressure": 9984.5,
  "altitude": 12.1
}

---

## Response (Success)

{
  "needs_irrigation": true
}

---

## Response (Error)

### Missing fields
{
  "error": "Missing required fields"
}

### Model not available
{
  "error": "Model is not loaded yet"
}

---

## Rules

- All inputs must be numeric
- soil_pct should be between 0 and 100
- API must respond in JSON format