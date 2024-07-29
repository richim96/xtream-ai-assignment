"""Tests for the `sample_router` in diamond_app"""

from fastapi.testclient import TestClient
from xtream_service.api.diamond import app

client = TestClient(app)


def test_price_predict():
    """Test the functionality of `price_predict` endpoint."""
    params: dict = {
        "carat": 1.2,
        "cut": "Premium",
        "color": "E",
        "clarity": "VS1",
        "n_samples": 4,
    }
    response = client.get("/diamonds/samples/", params=params)

    assert response.status_code == 200
    assert response.json()["response_type"] == "sample_get"
    assert isinstance(response.json()["samples"], list)
    assert response.json()["n_samples"] <= params["n_samples"]
