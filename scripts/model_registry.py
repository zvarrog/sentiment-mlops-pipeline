"""Общая логика работы с моделями и MLflow Registry."""

import json
from pathlib import Path

from scripts.config import (
    MODEL_ARTEFACTS_DIR,
    MODEL_PRODUCTION_THRESHOLD,
)
from scripts.logging_config import get_logger
from scripts.models.kinds import ModelKind

log = get_logger(__name__)


def load_old_model_metric() -> float | None:
    """Загружает метрику предыдущей модели из best_model_meta.json."""
    best_meta_path = Path(MODEL_ARTEFACTS_DIR) / "best_model_meta.json"

    if not best_meta_path.exists():
        return None

    try:
        with open(best_meta_path, encoding="utf-8") as f:
            old_meta = json.load(f)
            old_model_metric = old_meta.get("best_val_f1_macro")
            if old_model_metric:
                log.info(
                    "Найдена предыдущая модель с val_f1_macro=%.4f",
                    old_model_metric,
                )
            return old_model_metric
    except Exception as e:
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


def register_model_in_mlflow(
    model_path: Path,
    model_kind: ModelKind,
    test_f1_macro: float,
    mlflow_run_active: bool = False,
) -> None:
    """Регистрирует модель в MLflow Registry с переводом в Staging/Production.

    Args:
        model_path: Путь к сохранённой модели
        model_kind: Тип модели (ModelKind enum)
        test_f1_macro: F1-метрика на test set
        mlflow_run_active: Если True, предполагается активный MLflow run
    """
    import mlflow
    from mlflow.tracking import MlflowClient

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
                archive_existing_versions=True,
            )
            log.info(
                "Модель %s версия %s переведена в Production (F1=%.4f >= %.2f)",
                model_name,
                latest_version,
                test_f1_macro,
                MODEL_PRODUCTION_THRESHOLD,
            )
        else:
            log.warning(
                "Модель %s версия %s остаётся в Staging (F1=%.4f < %.2f)",
                model_name,
                latest_version,
                test_f1_macro,
                MODEL_PRODUCTION_THRESHOLD,
            )
    except Exception as e:
        log.warning("Не удалось зарегистрировать модель в MLflow Registry: %s", e)


def _log_model_to_mlflow(
    model_path: Path, model_kind: ModelKind, model_name: str
) -> None:
    """Логирует модель в MLflow (pyfunc для DistilBERT, sklearn для остальных)."""
    import mlflow

    if model_kind == ModelKind.distilbert:
        import mlflow.pyfunc

        class DistilBertWrapper(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                import joblib

                self.model = joblib.load(context.artifacts["model_path"])

            def predict(self, context, model_input):
                import pandas as pd

                if isinstance(model_input, pd.DataFrame):
                    texts = model_input["reviewText"].tolist()
                else:
                    texts = model_input
                return self.model.predict(texts)

        artifacts = {"model_path": str(model_path)}
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DistilBertWrapper(),
            artifacts=artifacts,
            registered_model_name=model_name,
        )
    else:
        import joblib
        import mlflow.sklearn

        model_obj = joblib.load(model_path)
        mlflow.sklearn.log_model(
            sk_model=model_obj,
            artifact_path="model",
            registered_model_name=model_name,
        )
