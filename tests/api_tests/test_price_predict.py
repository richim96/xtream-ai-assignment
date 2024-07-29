"""Tests for the `price_router` in diamond_app"""

from fastapi.testclient import TestClient
from xtream_service.api.diamond import app

client = TestClient(app)


def test_price_predict():
    """Test the functionality of `price_predict` endpoint."""
    payload: dict = {
        "carat": 1.2,
        "cut": "Premium",
        "color": "E",
        "clarity": "VS1",
        "depth": 64.2,
        "table": 60.1,
        "x": 5.94,
        "y": 5.6,
        "z": 4.22,
    }
    response = client.post("/diamonds/price/", json=payload)

    assert response.status_code == 200
    assert response.json()["response_type"] == "price_predict"
    assert isinstance(response.json()["predicted_price"], int)
