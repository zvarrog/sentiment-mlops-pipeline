"""Оркестратор обучения моделей.

Координирует процесс обучения:
- Оптимизацию гиперпараметров (Optuna)
- Финальное обучение лучшей модели
- Сохранение артефактов
- Регистрацию в MLflow
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report

from scripts.artefact_store import artefact_store
from scripts.config import (
    BEST_MODEL_PATH,
    MISCLASSIFIED_SAMPLES_LIMIT,
    MODEL_ARTEFACTS_DIR,
    NUMERIC_COLS,
    TRAIN_DEVICE,
)
from scripts.evaluation_reporter import save_feature_importances_safe
from scripts.logging_config import get_logger
from scripts.model_registry import register_model_in_mlflow
from scripts.models.distilbert import DistilBertClassifier
from scripts.models.kinds import ModelKind
from scripts.train_modules.evaluation import compute_metrics, log_confusion_matrix
from scripts.train_modules.optuna_optimizer import optimize_model
from scripts.train_modules.pipeline_builders import ModelBuilderFactory
from scripts.types import TrialResult
from scripts.utils import get_baseline_stats
from scripts.visualization import plot_roc_pr_curves

log = get_logger(__name__)


@dataclass
class TrainingResult:
    """Результат обучения модели."""

    model_kind: ModelKind
    pipeline: Any
    params: dict[str, Any]
    val_f1_macro: float
    test_metrics: dict[str, float]
    duration_sec: float


class TrainingOrchestrator:
    """Координирует полный цикл обучения модели."""

    def __init__(
        self,
        study_base_name: str,
        optuna_storage: str,
        model_dir: Path = MODEL_ARTEFACTS_DIR,
    ):
        self.study_base_name = study_base_name
        self.optuna_storage = optuna_storage
        self.model_dir = model_dir

    def run_optimization(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        model_kinds: list[ModelKind],
    ) -> dict[ModelKind, TrialResult]:
        """Запускает оптимизацию Optuna для выбранных моделей."""
        results: dict[ModelKind, TrialResult] = {}

        for model_kind in model_kinds:
            with mlflow.start_run(nested=True, run_name=f"model={model_kind.value}"):
                log.info("Оптимизация модели: %s", model_kind.value)

                study = optimize_model(
                    self.study_base_name,
                    model_kind,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    self.optuna_storage,
                )

                if not study.trials or not study.best_trial:
                    log.warning("%s: нет успешных trials — пропуск", model_kind.value)
                    continue

                best_trial = study.best_trial
                results[model_kind] = {
                    "best_value": float(best_trial.value or 0.0),
                    "best_params": best_trial.params,
                    "study_name": study.study_name,
                }

        return results

    def train_final_model(
        self,
        model_kind: ModelKind,
        params: dict[str, Any],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> Any:
        """Обучает финальную модель с заданными параметрами."""
        if model_kind is ModelKind.distilbert:
            pipeline = DistilBertClassifier(
                epochs=params.get("db_epochs", 2),
                lr=params.get("db_lr", 2e-5),
                max_len=params.get("db_max_len", 160),
                device=TRAIN_DEVICE,
            )
            pipeline.fit(x_train["reviewText"], y_train)
        else:
            fixed_trial = optuna.trial.FixedTrial(params)
            available_cols = [c for c in NUMERIC_COLS if c in x_train.columns]
            builder = ModelBuilderFactory.get_builder(
                model_kind, fixed_trial, numeric_cols=available_cols
            )
            pipeline = builder.build()
            pipeline.fit(x_train, y_train)

        return pipeline

    def evaluate_and_save(
        self,
        pipeline: Any,
        model_kind: ModelKind,
        params: dict[str, Any],
        val_f1: float,
        x_train: pd.DataFrame,
        x_val: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        baseline_stats: dict[str, dict[str, float]],
        duration: float,
    ) -> dict[str, float]:
        """Оценивает модель и сохраняет все артефакты."""
        is_distilbert = model_kind is ModelKind.distilbert

        # Предсказания на тесте
        if is_distilbert:
            test_preds = pipeline.predict(x_test["reviewText"])
        else:
            test_preds = pipeline.predict(x_test)

        test_metrics = compute_metrics(y_test, test_preds)

        # Логируем метрики в MLflow
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Сохраняем артефакты
        self._save_all_artifacts(
            pipeline=pipeline,
            model_kind=model_kind,
            params=params,
            val_f1=val_f1,
            test_metrics=test_metrics,
            test_preds=test_preds,
            x_test=x_test,
            y_test=y_test,
            baseline_stats=baseline_stats,
            duration=duration,
            dataset_sizes={"train": len(x_train), "val": len(x_val), "test": len(x_test)},
        )

        return test_metrics

    def _save_all_artifacts(
        self,
        pipeline: Any,
        model_kind: ModelKind,
        params: dict[str, Any],
        val_f1: float,
        test_metrics: dict[str, float],
        test_preds: np.ndarray,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        baseline_stats: dict[str, dict[str, float]],
        duration: float,
        dataset_sizes: dict[str, int],
    ) -> None:
        """Сохраняет все артефакты обучения."""
        out_dir = self.model_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        is_distilbert = model_kind is ModelKind.distilbert

        # Модель
        artefact_store.save_model(BEST_MODEL_PATH, pipeline)
        self._log_artifact_safe(BEST_MODEL_PATH, "best_model")

        # Baseline статистики
        bs_path = out_dir / "baseline_numeric_stats.json"
        artefact_store.save_json(bs_path, baseline_stats)
        self._log_artifact_safe(bs_path, "baseline_stats")

        # Confusion matrix и classification report
        cm_path = out_dir / "confusion_matrix_test.png"
        log_confusion_matrix(y_test, test_preds, cm_path)
        self._log_artifact_safe(cm_path, "confusion_matrix")

        cr_txt = classification_report(y_test, test_preds, output_dict=False)
        cr_path = out_dir / "classification_report_test.txt"
        artefact_store.save_text(cr_path, cr_txt)
        self._log_artifact_safe(cr_path, "classification_report")

        # Важности признаков (только для классических моделей)
        if not is_distilbert:
            fi_path = save_feature_importances_safe(pipeline, out_dir)
            if fi_path:
                self._log_artifact_safe(fi_path, "feature_importances")

        # ROC/PR кривые
        try:
            roc_path, pr_path = plot_roc_pr_curves(pipeline, x_test, y_test, out_dir, is_distilbert)
            if roc_path:
                self._log_artifact_safe(roc_path, "roc_curve")
            if pr_path:
                self._log_artifact_safe(pr_path, "pr_curve")
        except (ValueError, OSError, RuntimeError) as e:
            log.warning("Не удалось построить ROC/PR кривые: %s", e)

        # Ошибки классификации
        self._save_misclassified(x_test, y_test, test_preds, out_dir)

        # Метаданные
        meta = {
            "best_model": model_kind.value,
            "best_params": params,
            "best_val_f1_macro": val_f1,
            "test_metrics": test_metrics,
            "sizes": dataset_sizes,
            "duration_sec": duration,
        }
        meta_path = out_dir / "best_model_meta.json"
        artefact_store.save_json(meta_path, meta)

    def _save_misclassified(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        test_preds: np.ndarray,
        out_dir: Path,
    ) -> None:
        """Сохраняет примеры ошибок классификации."""
        mis_idx = np.where(test_preds != y_test)[0]
        if len(mis_idx) == 0:
            return

        mis_samples = x_test.iloc[mis_idx].copy()
        mis_samples["true"] = (
            y_test.iloc[mis_idx] if isinstance(y_test, pd.Series) else y_test[mis_idx]
        )
        mis_samples["pred"] = test_preds[mis_idx]

        mis_path = out_dir / "misclassified_samples_test.csv"
        artefact_store.save_csv(mis_path, mis_samples.head(MISCLASSIFIED_SAMPLES_LIMIT))
        self._log_artifact_safe(mis_path, "misclassified_samples")

    @staticmethod
    def _log_artifact_safe(path: Path, name: str) -> None:
        """Безопасное логирование артефакта в MLflow."""
        try:
            mlflow.log_artifact(str(path))
        except (OSError, RuntimeError, ValueError) as e:
            log.warning("Не удалось залогировать %s: %s", name, e)
