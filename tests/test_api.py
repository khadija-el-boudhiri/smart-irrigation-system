def test_home_returns_200(client):
    response = client.get('/')
    assert response.status_code == 200
    assert "message" in response.get_json()

def test_health_returns_ok(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}

def test_predict_valid_payload(client, valid_payload):
    response = client.post('/predict', json=valid_payload)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert "needs_irrigation" in response.get_json()

def test_predict_missing_field(client, missing_field_payload):
    response = client.post('/predict', json=missing_field_payload)
    assert response.status_code in [400, 503]

def test_predict_out_of_range(client, out_of_range_payload):
    response = client.post('/predict', json=out_of_range_payload)
    assert response.status_code in [400, 503]
