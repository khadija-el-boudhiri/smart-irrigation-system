# API Contract — Smart Irrigation System

## Base URL

- **Flask implementation**: http://localhost:5000
- **FastAPI implementation**: http://localhost:8000

---

## 1. Home

### Endpoint
GET /

### Response (Flask)
```json
{
  "message": "Smart Irrigation API is running"
}
```

### Response (FastAPI)
```json
{
  "message": "Smart Irrigation FastAPI is running"
}
```

---

## 2. Health Check

### Endpoint
GET /health

### Response
```json
{
  "status": "ok"
}
```

### Note
Available in both Flask and FastAPI implementations.

---

## 3. UI Endpoint

### Endpoint
GET /ui

### Response (Success - if api/index.html exists)
Returns the HTML file from `api/index.html`

### Response (Error - if api/index.html not found)
```json
{
  "message": "UI not available yet"
}
```
Status Code: 503

### Error Response (Flask variant)
```json
{
  "error": "UI not available yet"
}
```
Status Code: 503

---

## 4. Prediction Endpoint

### Endpoint
POST /predict

### Request Body (JSON)
```json
{
  "soil_pct": float,
  "temperature": float,
  "pressure": float,
  "altitude": float
}
```

### Example Request
```json
{
  "soil_pct": 35.2,
  "temperature": 28.0,
  "pressure": 9984.5,
  "altitude": 12.1
}
```

### Response (Success)
```json
{
  "needs_irrigation": true
}
```

### Response (Error)

#### Model not available
```json
{
  "error": "Model is not loaded yet"
}
```
Status Code: 503

#### Field out of valid range
```json
{
  "error": "Field {field} is out of valid range"
}
```
Status Code: 400

---

## Input Validation Rules

### Field Ranges (Required Limits)
All inputs must be numeric and within these ranges:
- `soil_pct`: 0 to 100
- `temperature`: 10 to 42 (°C)
- `pressure`: 9780 to 10120 (hPa)
- `altitude`: 0 to 500 (meters)

### Response Format
- API must respond in JSON format
- Successful predictions return status code 200
- Validation errors return status code 400
- Service unavailability returns status code 503

---

## Additional Endpoints (FastAPI Only)

### Metrics Endpoint
GET /metrics

Returns Prometheus-compatible metrics for monitoring.

### Static Files
GET /static/{path}

Serves static files from the `api/` directory (if mounted successfully).

---

## Implementation Notes

- Both Flask and FastAPI implementations maintain the same API contract
- Field validation is consistent across both implementations
- The /health endpoint is now available in both Flask and FastAPI