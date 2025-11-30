"""Общая логика работы с моделями и MLflow Registry."""

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from scripts.config import (
    MLFLOW_KEEP_LATEST,
    MODEL_ARTEFACTS_DIR,
    MODEL_PRODUCTION_THRESHOLD,
)
from scripts.logging_config import get_logger
from scripts.models.kinds import ModelKind

log = get_logger(__name__)


class DistilBertWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for DistilBERT model to be saved in MLflow."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: Any
    ) -> Any:
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["reviewText"].tolist()
        else:
            texts = model_input
        return self.model.predict(texts)


def load_old_model_metric() -> float | None:
    """Загружает метрику предыдущей модели из best_model_meta.json."""
    best_meta_path = Path(MODEL_ARTEFACTS_DIR) / "best_model_meta.json"

    if not best_meta_path.exists():
        return None

    try:
        with open(best_meta_path, encoding="utf-8") as f:
            old_meta = json.load(f)
        old_model_metric = old_meta.get("best_val_f1_macro")
        if isinstance(old_model_metric, (int, float)):
            log.info(
                "Найдена предыдущая модель с val_f1_macro=%.4f",
                old_model_metric,
            )
            return float(old_model_metric)
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("Не удалось загрузить метаданные старой модели: %s", e)
        return None


def should_replace_model(
    new_metric: float, old_metric: float | None, model_name: str
) -> bool:
    """Проверяет, нужно ли заменять старую модель новой."""
    if old_metric is None:
        log.info("Предыдущей модели нет — сохраняем новую")
        return True

    if new_metric <= old_metric:
        log.info(
            "Новая модель %s (val_f1_macro=%.4f) НЕ лучше предыдущей (%.4f)",
            model_name,
            new_metric,
            old_metric,
        )
        return False

    log.info(
        "Новая модель %s (val_f1_macro=%.4f) лучше предыдущей (%.4f) — заменяем",
        model_name,
        new_metric,
        old_metric,
    )
    return True


def _log_model_to_mlflow(
    model_path: Path, model_kind: ModelKind, model_name: str
) -> None:
    """Логирует модель в MLflow (pyfunc для DistilBERT, sklearn для остальных)."""
    if model_kind == ModelKind.distilbert:
        artifacts = {"model_path": str(model_path)}
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DistilBertWrapper(),
            artifacts=artifacts,
            registered_model_name=model_name,
        )
    else:
        model_obj = joblib.load(model_path)
        mlflow.sklearn.log_model(
            sk_model=model_obj,
            artifact_path="model",
            registered_model_name=model_name,
        )


def _rotate_production_versions(
    client: MlflowClient, model_name: str, keep_n: int = 3
) -> None:
    """Архивирует старые Production-версии, оставляя только keep_n последних."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [
            v for v in versions if getattr(v, "current_stage", "") == "Production"
        ]
        # Сортируем по номеру версии (от большей к меньшей)
        prod_versions.sort(key=lambda v: int(getattr(v, "version", "0")), reverse=True)

        for v in prod_versions[keep_n:]:
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived",
                archive_existing_versions=False,
            )

        if len(prod_versions) > keep_n:
            log.info(
                "Заархивированы старые Production-версии, оставлено %d последних",
                keep_n,
            )
    except (OSError, ValueError, RuntimeError) as e:
        log.warning("Не удалось выполнить ротацию Production-версий: %s", e)


def register_model_in_mlflow(
    model_path: Path,
    model_kind: ModelKind,
    test_f1_macro: float,
    mlflow_run_active: bool = False,
) -> None:
    """Регистрирует модель в MLflow Registry с переводом в Staging/Production."""
    model_name = "sentiment_kindle_model"

    try:
        if mlflow_run_active:
            _log_model_to_mlflow(model_path, model_kind, model_name)
        else:
            with mlflow.start_run(run_name=f"register_{model_kind.value}"):
                mlflow.log_param("model", model_kind.value)
                mlflow.log_param("test_f1_macro", test_f1_macro)
                _log_model_to_mlflow(model_path, model_kind, model_name)

        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if not latest_versions:
            log.warning("Не удалось получить последнюю версию модели из Registry")
            return

        latest_version = latest_versions[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging",
            archive_existing_versions=False,
        )
        log.info(
            "Модель %s версия %s зарегистрирована в MLflow Registry (stage: Staging)",
            model_name,
            latest_version,
        )

        if test_f1_macro >= MODEL_PRODUCTION_THRESHOLD:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=False,
            )
            log.info(
                "Модель %s версия %s переведена в Production (F1=%.4f >= %.2f)",
                model_name,
                latest_version,
                test_f1_macro,
                MODEL_PRODUCTION_THRESHOLD,
            )

            _rotate_production_versions(client, model_name, MLFLOW_KEEP_LATEST)

        else:
            log.warning(
                "Модель %s версия %s остаётся в Staging (F1=%.4f < %.2f)",
                model_name,
                latest_version,
                test_f1_macro,
                MODEL_PRODUCTION_THRESHOLD,
            )
    except (OSError, ValueError, RuntimeError) as e:
        log.warning("Не удалось зарегистрировать модель в MLflow Registry: %s", e)
