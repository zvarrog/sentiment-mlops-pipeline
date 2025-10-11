"""Smoke тест для API с проверкой feature contract."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# Имитируем модель для тестов
class MockPipeline:
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return [1] * len(X)  # Возвращаем список длиной как количество строк
        else:
            return [1] * len(X)  # Для Series или других итерируемых

    def predict_proba(self, X):
        n = len(X)
        return [[0.1, 0.9]] * n  # Возвращаем одинаковые вероятности для всех


@pytest.fixture
def mock_app_state():
    """Мокаем состояние приложения."""
    with patch("scripts.api_service.app.state") as mock_state:
        # Используем актуальное имя атрибута, которое читает API
        mock_state.MODEL = MockPipeline()
        mock_state.META = {
            "best_model": "test_model",
            "test_metrics": {"f1_macro": 0.85},
        }
        mock_state.NUMERIC_DEFAULTS = {
            "text_len": {"mean": 100.0, "std": 50.0},
            "word_count": {"mean": 20.0, "std": 10.0},
        }
        # Мокаем feature contract
        mock_contract = Mock()
        mock_contract.expected_numeric_columns = ["text_len", "word_count"]
        mock_contract.get_feature_info.return_value = {
            "required_text_columns": ["reviewText"],
            "expected_numeric_columns": ["text_len", "word_count"],
            "total_features": 3,
        }
        mock_contract.validate_input_data.return_value = {}
        mock_state.FEATURE_CONTRACT = mock_contract
        yield mock_state


@pytest.fixture
def client():
    """Тестовый клиент FastAPI."""
    # Мокаем загрузку артефактов
    with patch("scripts.api_service._load_artifacts"):
        from scripts.api_service import app

        return TestClient(app)


def test_api_feature_contract(mock_app_state):
    # Импортируем список числовых фич из тренера
    # Берём конструктор датафрейма из API
    from scripts.api_service import _build_dataframe

    texts = [
        "Loved this Kindle!",
        "Terrible experience",
    ]
    df, _ignored = _build_dataframe(texts)

    # Базовые ожидания
    assert "reviewText" in df.columns

    # Числовые признаки, которые тренер ожидает, должны присутствовать (созданы/заполнены)
    # В актуальной архитектуре список ожидаемых числовых фич задаётся FeatureContract.
    # В этом тесте он мокается как ["text_len", "word_count"].
    expected_numeric = ["text_len", "word_count"]
    missing = [c for c in expected_numeric if c not in df.columns]
    assert not missing, f"API dataframe lacks expected numeric columns: {missing}"

    # Размер совпадает с количеством входных текстов
    assert len(df) == len(texts)


def test_health_endpoint(client, mock_app_state):
    """Тест health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "best_model" in data


def test_metadata_endpoint(client, mock_app_state):
    """Тест metadata endpoint."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()

    assert "model_info" in data
    assert "feature_contract" in data
    assert "health" in data

    # Проверяем структуру ответа
    assert data["model_info"]["best_model"] == "test_model"
    assert "required_text_columns" in data["feature_contract"]
    assert data["health"]["model_loaded"] is True


def test_predict_endpoint_simple(client, mock_app_state):
    """Тест базового predict endpoint."""
    request_data = {"texts": ["Great product!", "Bad quality"]}

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()

    assert "labels" in data
    assert len(data["labels"]) == 2
    assert all(isinstance(label, int) for label in data["labels"])


def test_predict_endpoint_with_features(client, mock_app_state):
    """Тест predict endpoint с числовыми признаками."""
    request_data = {
        "texts": ["Great product!", "Bad quality"],
        "numeric_features": {"text_len": [13.0, 11.0], "word_count": [2.0, 2.0]},
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()

    assert "labels" in data
    assert len(data["labels"]) == 2


def test_batch_predict_endpoint(client, mock_app_state):
    """Тест batch predict endpoint."""
    request_data = {
        "data": [
            {"reviewText": "Great product!", "text_len": 13.0, "word_count": 2.0},
            {"reviewText": "Bad quality", "text_len": 11.0, "word_count": 2.0},
            {"reviewText": "Average item", "text_len": 12.0},
        ]
    }

    response = client.post("/batch_predict", json=request_data)
    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 3

    # Проверяем структуру предсказаний
    for pred in data["predictions"]:
        assert "index" in pred
        assert "prediction" in pred


def test_feature_validation_warnings(client, mock_app_state):
    """Тест валидации признаков с предупреждениями."""
    request_data = {
        "texts": ["Very long text that might be an outlier"],
        "numeric_features": {"text_len": [1000.0]},  # Выброс
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()

    # В откатанной версии API warnings могут быть None, если нет игнорированных фич
    assert "warnings" in data  # Поле должно быть
    # data["warnings"] может быть None — это нормально


def test_empty_texts_error(client, mock_app_state):
    """Тест ошибки при пустом списке текстов."""
    request_data = {"texts": []}

    response = client.post("/predict", json=request_data)
    assert response.status_code == 400
    assert "пуст" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
