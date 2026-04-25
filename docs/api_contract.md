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
  "soil_moisture": float,
  "temperature": float,
  "humidity": float
}

### Example Request
{
  "soil_moisture": 35,
  "temperature": 28,
  "humidity": 60
}

---

## Response (Success)

{
  "decision": "IRRIGATE"
}

OR

{
  "decision": "DO_NOT_IRRIGATE"
}

---

## Response (Error)

### Missing fields
{
  "error": "Missing required fields"
}

### Model not available
{
  "error": "Model not available"
}

---

## Rules

- All inputs must be numeric
- soil_moisture must be between 0 and 100
- API must respond in JSON format