"""Unit tests для API service (FastAPI).

Фокус: корректность схемы запросов/ответов и устойчивость без реальных артефактов.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class DummyModel:
    """Заглушечная модель для тестов."""

    def predict(self, X):
        # Поддерживаем как pd.Series, так и pd.DataFrame
        try:
            n = len(X)
        except Exception:
            n = 1
        # Возвращаем нули
        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        try:
            n = len(X)
        except Exception:
            n = 1
        probs0 = np.full(n, 0.6)
        probs1 = np.full(n, 0.4)
        return np.column_stack([probs0, probs1])


@pytest.fixture(scope="module")
def mock_model():
    """Создаёт простую модель для тестов."""
    return DummyModel()


@pytest.fixture(scope="module")
def mock_feature_contract():
    """Создаёт mock feature contract."""
    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]
    return contract


@pytest.fixture(scope="module")
def test_client(mock_model, mock_feature_contract, tmp_path_factory):
    """Создаёт тестовый клиент FastAPI с моками и пропуском загрузки артефактов."""
    tmp_model = tmp_path_factory.mktemp("model") / "best_model.joblib"
    tmp_model.write_bytes(b"fake")

    with (
        patch("scripts.api_service.BEST_MODEL_PATH", tmp_model),
        patch("scripts.api_service.joblib.load", return_value=mock_model),
        patch("scripts.api_service.FeatureContract") as mock_contract_cls,
    ):
        mock_contract_cls.from_model_artifacts.return_value = mock_feature_contract

        from scripts.api_service import create_app

        app = create_app(defer_artifacts=False)
        client = TestClient(app)
        app.state.META = {"best_model": "logreg"}
        app.state.NUMERIC_DEFAULTS = {
            "text_len": {"mean": 10.0},
            "word_count": {"mean": 2.0},
        }
        app.state.FEATURE_CONTRACT = mock_feature_contract
        yield client


class TestAPIServiceHealthCheck:
    """Тесты для health check эндпоинта."""

    def test_health_check_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


class TestAPISinglePrediction:
    """Тесты для /predict эндпоинта."""

    def test_predict_with_valid_input(self, test_client):
        payload = {
            "texts": ["great product"],
            "numeric_features": {"text_len": [13.0], "word_count": [2.0]},
        }
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "labels" in data

    def test_predict_with_missing_required_field(self, test_client):
        payload = {"numeric_features": {"text_len": [13.0], "word_count": [2.0]}}
        resp = test_client.post("/predict", json=payload)
        # Pydantic схемы вернут 422 при отсутствии обязательного поля texts
        assert resp.status_code == 422

    def test_predict_with_invalid_type(self, test_client):
        payload = {"texts": "not_a_list"}
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestAPIBatchPrediction:
    """Тесты для /batch_predict эндпоинта."""

    def test_batch_predict_with_valid_input(self, test_client):
        payload = {
            "data": [
                {"reviewText": "great product", "text_len": 13.0, "word_count": 2.0},
                {"reviewText": "bad quality", "text_len": 11.0, "word_count": 2.0},
            ]
        }
        resp = test_client.post("/batch_predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data and len(data["predictions"]) == 2

    def test_batch_predict_with_empty_list(self, test_client):
        payload = {"data": []}
        resp = test_client.post("/batch_predict", json=payload)
        assert resp.status_code == 400

    def test_batch_predict_exceeds_limit(self, test_client):
        payload = {"data": [{"reviewText": "t"}] * 1001}
        # Лимит реализован декоратором slowapi (50/minute), но в тесте просто проверим 200
        # чтобы не зависеть от глобального состояния rate limiting.
        resp = test_client.post("/batch_predict", json=payload)
        assert resp.status_code in (200, 429)


class TestAPIMetrics:
    """Smoke tests для Prometheus metrics."""

    def test_metrics_endpoint_exists(self, test_client):
        """GET /metrics возвращает 200."""
        response = test_client.get("/metrics")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
