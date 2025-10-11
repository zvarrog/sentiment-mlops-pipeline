#!/usr/bin/env python3
"""
Простой тест для проверки MLflow настроек (локально и в контейнере).
"""

import logging
import os
import sys
from pathlib import Path

# Гарантируем, что корень репозитория в sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow

from scripts.logging_config import setup_training_logging
from scripts.train import _configure_mlflow_tracking

logger = logging.getLogger(__name__)


def test_mlflow_config():
    # Настраиваем логирование
    setup_training_logging()

    # Логируем исходные переменные окружения
    logger.info(
        f"MLFLOW_TRACKING_URI (env): {os.environ.get('MLFLOW_TRACKING_URI', 'НЕ ЗАДАНО')}"
    )
    logger.info(f"Текущий рабочий каталог: {os.getcwd()}")
    airflow_mlruns = Path("/opt/airflow/mlruns")
    logger.info(f"Существует /opt/airflow/mlruns: {airflow_mlruns.exists()}")

    # Вызываем нашу функцию настройки
    _configure_mlflow_tracking()

    # Проверяем результат
    uri = mlflow.get_tracking_uri()
    logger.info(f"MLflow tracking URI: {uri}")
    assert uri is not None

    # Попробуем создать тестовый run
    with mlflow.start_run(run_name="config_test") as run:
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Artifact URI: {run.info.artifact_uri}")

        # Логируем тестовый параметр
        mlflow.log_param("test_param", "success")
        logger.info("Тест успешно завершён!")
