"""Shared fixtures для тестов."""

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
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


@pytest.fixture(scope="session")
def sample_parquet_files_small(tmp_path_factory) -> Iterator[Path]:
    """Создаёт небольшие parquet-файлы для быстрых интеграционных тестов.

    Использует переменную окружения TEST_PER_CLASS (по умолчанию 500)
    для генерации сбалансированного датасета с 5 классами.
    """
    per_class = int(os.getenv("TEST_PER_CLASS", "500"))
    classes = [1, 2, 3, 4, 5]
    n_total = per_class * len(classes)

    # Генерация синтетических данных
    np.random.seed(42)
    data = {
        "reviewText": np.random.choice(
            [
                "This product is excellent and works perfectly",
                "Terrible quality, very disappointed with purchase",
                "Amazing device, highly recommend to everyone",
                "Waste of money, stopped working after week",
                "Great value for the price, satisfied customer",
                "Poor design and cheap materials used throughout",
                "Outstanding performance, exceeded my expectations completely",
                "Mediocre at best, nothing special about it",
            ],
            size=n_total,
        ),
        "overall": np.repeat(classes, per_class),
        "text_len": np.random.randint(10, 300, size=n_total).astype(float),
        "word_count": np.random.randint(1, 80, size=n_total).astype(float),
    }
    df = pd.DataFrame(data)

    # Создаём временную директорию
    processed_dir = tmp_path_factory.mktemp("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем train/val/test splits
    df.to_parquet(processed_dir / "train.parquet", index=False)
    df.iloc[:per_class].to_parquet(processed_dir / "val.parquet", index=False)
    df.iloc[per_class : per_class * 2].to_parquet(
        processed_dir / "test.parquet", index=False
    )

    old_processed_dir = os.environ.get("PROCESSED_DATA_DIR")
    os.environ["PROCESSED_DATA_DIR"] = str(processed_dir)

    yield processed_dir

    if old_processed_dir:
        os.environ["PROCESSED_DATA_DIR"] = old_processed_dir


@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """Подменяет MLflow вызовы на noop для избежания зависимости от внешних сервисов."""

    class DummyRun:
        info = SimpleNamespace(run_id="dummy_run_id")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    # Мокаем основные MLflow функции
    monkeypatch.setattr(mlflow, "start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)

    # Мокаем sklearn.log_model и pyfunc.log_model
    try:
        import importlib

        mlflow_sklearn = importlib.import_module("mlflow.sklearn")
        monkeypatch.setattr(mlflow_sklearn, "log_model", lambda *a, **k: None)
    except (ImportError, AttributeError):
        pass

    try:
        import importlib

        mlflow_pyfunc = importlib.import_module("mlflow.pyfunc")
        monkeypatch.setattr(mlflow_pyfunc, "log_model", lambda *a, **k: None)
    except (ImportError, AttributeError):
        pass

    yield
