"""Обучение моделей с Optuna и MLflow.

Точка входа для запуска обучения. Использует TrainingOrchestrator
для координации процесса.
"""

import argparse
import logging
import time
import warnings

import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn

from scripts.config import (
    BEST_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_ARTEFACTS_DIR,
    MODEL_DIR,
    N_FOLDS,
    NUMERIC_COLS,
    OPTUNA_STORAGE,
    SEED,
    SELECTED_MODEL_KINDS,
    STUDY_BASE_NAME,
)
from scripts.logging_config import get_logger
from scripts.model_registry import (
    load_old_model_metric,
    register_model_in_mlflow,
    should_replace_model,
)
from scripts.models.kinds import ModelKind
from scripts.shutdown import register_shutdown_handlers
from scripts.train_modules.data_loading import load_splits
from scripts.train_modules.orchestrator import TrainingOrchestrator
from scripts.utils import get_baseline_stats

log = get_logger(__name__)


def _configure_logging_levels() -> None:
    """Настройка уровней логирования для внешних библиотек."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("optuna").setLevel(logging.ERROR)
    logging.getLogger("git").setLevel(logging.ERROR)


def _setup_environment() -> None:
    """Настройка окружения: сигналы, директории, MLflow."""
    if not OPTUNA_STORAGE:
        raise EnvironmentError(
            "OPTUNA_STORAGE не задан. Укажите строку подключения к PostgreSQL в .env"
        )

    _configure_logging_levels()
    register_shutdown_handlers(exit_on_signal=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _log_initial_params(x_train: pd.DataFrame) -> None:
    """Логирование начальных параметров в MLflow."""
    mlflow.log_params(
        {
            "seed": SEED,
            "numeric_cols": ",".join([c for c in NUMERIC_COLS if c in x_train.columns]),
            "text_clean_stage": "spark_process",
            "cv_n_folds": N_FOLDS,
            "version_sklearn": sklearn.__version__,
            "version_optuna": optuna.__version__,
            "version_mlflow": mlflow.__version__,
            "version_pandas": pd.__version__,
        }
    )


def run(
    force: bool = False,
    selected_models: list[ModelKind] | None = None,
) -> None:
    """Основная функция запуска обучения.

    Args:
        force: Принудительное переобучение даже если модель существует.
        selected_models: Список моделей для обучения (None = все из конфига).
    """
    _setup_environment()

    log.info("force=%s, наличие best_model=%s", force, BEST_MODEL_PATH.exists())

    old_model_metric = load_old_model_metric()

    if BEST_MODEL_PATH.exists() and not force:
        log.info("Модель уже существует и force=False — пропуск")
        return

    # Загрузка данных
    x_train, x_val, x_test, y_train, y_val, y_test = load_splits()
    log.info("Размеры: train=%d, val=%d, test=%d", len(x_train), len(x_val), len(x_test))

    start_time = time.time()
    orchestrator = TrainingOrchestrator(STUDY_BASE_NAME, OPTUNA_STORAGE, MODEL_ARTEFACTS_DIR)

    with mlflow.start_run(run_name="classical_pipeline"):
        _log_initial_params(x_train)
        baseline_stats = get_baseline_stats(x_train)

        # Оптимизация
        models_to_train = selected_models or SELECTED_MODEL_KINDS
        per_model_results = orchestrator.run_optimization(
            x_train, y_train, x_val, y_val, models_to_train
        )

        if not per_model_results:
            raise RuntimeError("Оптимизация не дала успешных результатов")

        # Выбор лучшей модели
        best_model, best_info = max(per_model_results.items(), key=lambda x: x[1]["best_value"])
        new_metric = best_info["best_value"]

        if not force and not should_replace_model(new_metric, old_model_metric, best_model.value):
            return

        log.info("Лучшая модель: %s (val_f1_macro=%.4f)", best_model.value, new_metric)

        # Логируем в MLflow
        mlflow.log_param("best_model", best_model.value)
        mlflow.log_params({f"best_{k}": v for k, v in best_info["best_params"].items()})
        mlflow.log_metric("best_val_f1_macro", new_metric)

        # Финальное обучение на train + val
        x_full = pd.concat([x_train, x_val], axis=0, ignore_index=True)
        y_full = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)

        final_pipeline = orchestrator.train_final_model(
            best_model, best_info["best_params"], x_full, y_full
        )

        duration = time.time() - start_time

        # Оценка и сохранение артефактов
        test_metrics = orchestrator.evaluate_and_save(
            pipeline=final_pipeline,
            model_kind=best_model,
            params=best_info["best_params"],
            val_f1=new_metric,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_test=y_test,
            baseline_stats=baseline_stats,
            duration=duration,
        )

        # Регистрация в MLflow
        register_model_in_mlflow(
            BEST_MODEL_PATH,
            best_model,
            test_metrics.get("f1_macro", 0.0),
            mlflow_run_active=True,
        )

        mlflow.log_metric("training_duration_sec", duration)


def main() -> None:
    """Точка входа: парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Обучение моделей с Optuna и MLflow")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительное переобучение даже если модель существует",
    )
    args = parser.parse_args()

    run(force=args.force)


if __name__ == "__main__":
    main()
