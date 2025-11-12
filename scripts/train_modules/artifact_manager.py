"""Управление артефактами обучения модели."""

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from scripts.logging_config import get_logger

log = get_logger("artifact_manager")


def save_model_artifact(model: Any, path: Path) -> None:
    """Сохраняет модель в joblib файл.

    Args:
        model: Обученная модель
        path: Путь для сохранения
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Модель сохранена: %s", path)


def save_json_artifact(data: dict[str, Any], path: Path) -> None:
    """Сохраняет JSON артефакт.

    Args:
        data: Данные для сохранения
        path: Путь для сохранения
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("JSON артефакт сохранен: %s", path)


def save_csv_artifact(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет CSV артефакт.

    Args:
        df: DataFrame для сохранения
        path: Путь для сохранения
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("CSV артефакт сохранен: %s", path)


def save_text_artifact(content: str, path: Path) -> None:
    """Сохраняет текстовый артефакт.

    Args:
        content: Текст для сохранения
        path: Путь для сохранения
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info("Текстовый артефакт сохранен: %s", path)


def log_mlflow_artifact_safe(path: Path, artifact_name: str) -> None:
    """Безопасное логирование артефакта в MLflow.

    Args:
        path: Путь к артефакту
        artifact_name: Имя артефакта для логирования
    """
    try:
        mlflow.log_artifact(str(path))
        log.debug("MLflow артефакт залогирован: %s", artifact_name)
    except Exception as e:
        log.warning("Не удалось залогировать %s в MLflow: %s", artifact_name, e)


def calculate_baseline_statistics(
    x_train: pd.DataFrame, numeric_cols: list[str]
) -> dict[str, dict[str, float]]:
    """Вычисляет baseline статистики для числовых признаков.

    Args:
        x_train: Тренировочные данные
        numeric_cols: Список числовых колонок

    Returns:
        Словарь со статистиками: {column: {mean, std, min, max}}
    """
    baseline_stats = {}
    for col in numeric_cols:
        if col in x_train.columns:
            series = x_train[col]
            baseline_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
    return baseline_stats
