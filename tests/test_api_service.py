"""Unit tests для API service (FastAPI).

Тесты:
- Загрузка модели
- Валидация входных данных
- Эндпоинт /predict
- Эндпоинт /batch_predict
- Обработка ошибок
"""

from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient


class DummyModel:
    """Заглушечная модель для тестов."""

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.random.rand(n), np.random.rand(n)])


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
    """Создаёт тестовый клиент FastAPI с mock зависимостями."""
    model_path = tmp_path_factory.mktemp("models") / "best_model.joblib"
    joblib.dump(mock_model, model_path)

    with patch("scripts.api_service.MODEL_PATH", str(model_path)):
        with patch("scripts.api_service.FeatureContract") as mock_contract_cls:
            mock_contract_cls.from_model_artifacts.return_value = mock_feature_contract

            from scripts.api_service import app

            client = TestClient(app)
            yield client


class TestAPIServiceHealthCheck:
    """Тесты для health check эндпоинта."""

    def test_health_check_returns_200(self, test_client):
        """GET / возвращает 200."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()


class TestAPISinglePrediction:
    """Тесты для /predict эндпоинта."""

    def test_predict_with_valid_input(self, test_client):
        """POST /predict с валидным входом возвращает предсказание."""
        payload = {
            "reviewText": "great product",
            "text_len": 13.0,
            "word_count": 2.0,
        }

        response = test_client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "predicted_sentiment" in data
        assert "confidence" in data

    def test_predict_with_missing_required_field(self, test_client):
        """POST /predict с отсутствующим обязательным полем возвращает 400."""
        payload = {
            "text_len": 13.0,
            "word_count": 2.0,
        }

        response = test_client.post("/predict", json=payload)

        assert response.status_code == 422

    def test_predict_with_invalid_type(self, test_client):
        """POST /predict с некорректным типом данных возвращает 422."""
        payload = {
            "reviewText": "great product",
            "text_len": "invalid_string",
            "word_count": 2.0,
        }

        response = test_client.post("/predict", json=payload)

        assert response.status_code == 422


class TestAPIBatchPrediction:
    """Тесты для /batch_predict эндпоинта."""

    def test_batch_predict_with_valid_input(self, test_client):
        """POST /batch_predict с валидным батчем возвращает предсказания."""
        payload = {
            "reviews": [
                {"reviewText": "great product", "text_len": 13.0, "word_count": 2.0},
                {"reviewText": "bad quality", "text_len": 11.0, "word_count": 2.0},
            ]
        }

        response = test_client.post("/batch_predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_batch_predict_with_empty_list(self, test_client):
        """POST /batch_predict с пустым списком возвращает 400."""
        payload = {"reviews": []}

        response = test_client.post("/batch_predict", json=payload)

        assert response.status_code == 400

    def test_batch_predict_exceeds_limit(self, test_client):
        """POST /batch_predict с превышением лимита возвращает 400."""
        payload = {
            "reviews": [{"reviewText": "test", "text_len": 4.0, "word_count": 1.0}]
            * 1001
        }

        response = test_client.post("/batch_predict", json=payload)

        assert response.status_code == 400


class TestAPIMetrics:
    """Smoke tests для Prometheus metrics."""

    def test_metrics_endpoint_exists(self, test_client):
        """GET /metrics возвращает 200."""
        response = test_client.get("/metrics")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
