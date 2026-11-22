from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.food11 import serve


class FakeModel:
    def predict(self, batch):
        assert batch.shape == (1, 3, 128, 128)
        assert batch.dtype == np.float32
        return np.log([[0.90] + [0.01] * 10])


def test_api(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/predict", files={"file": b"unused"}).status_code == 503
        monkeypatch.setattr(serve, "model", FakeModel())
        assert client.post("/predict", files={"file": b"invalid"}).status_code == 422
        with Path("tests/fixtures/bread.jpg").open("rb") as image:
            response = client.post(
                "/predict", files={"file": ("bread.jpg", image, "image/jpeg")}
            )
        assert response.status_code == 200
        assert response.json()["category"] == "Bread"
        assert response.json()["confidence"] == pytest.approx(0.90)
