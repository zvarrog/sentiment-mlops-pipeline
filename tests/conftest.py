"""Shared fixtures для тестов."""

import os

import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Настраивает переменные окружения для тестов."""
    os.environ["AIRFLOW_HOME"] = "/tmp/airflow_test"
    os.environ["RAW_DATA_DIR"] = "/tmp/data/raw"
    os.environ["PROCESSED_DATA_DIR"] = "/tmp/data/processed"
    os.environ["MODEL_DIR"] = "/tmp/models"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["POSTGRES_METRICS_URI"] = "postgresql://test:test@localhost:5432/test"


@pytest.fixture
def sample_dataframe():
    """Минимальный DataFrame для тестов."""
    return pd.DataFrame(
        {
            "reviewText": ["good product", "bad quality", "excellent"],
            "overall": [5, 1, 5],
            "text_len": [12.0, 11.0, 9.0],
            "word_count": [2.0, 2.0, 1.0],
        }
    )


@pytest.fixture
def sample_text_corpus():
    """Минимальный текстовый корпус."""
    return [
        "great product highly recommend",
        "terrible quality waste of money",
        "average nothing special",
    ]


@pytest.fixture
def temp_artifact_dir(tmp_path):
    """Временная директория для артефактов модели."""
    artifact_dir = tmp_path / "artefacts"
    artifact_dir.mkdir()
    return artifact_dir
