"""Оптимизация гиперпараметров с помощью Optuna."""

import signal
from collections.abc import Callable

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from scripts.config import (
    DISTILBERT_TIMEOUT_SEC,
    MIN_TRIALS_BEFORE_EARLY_STOP,
    N_FOLDS,
    NUMERIC_COLS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SEC,
    SEED,
)
from scripts.logging_config import get_logger
from scripts.models.kinds import ModelKind
from scripts.train_modules.evaluation import compute_metrics
from scripts.train_modules.pipeline_builders import ModelBuilderFactory

log = get_logger("optuna_optimizer")

_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    log.warning("Получен сигнал %s, прерываем оптимизацию", signum)
    _interrupted = True


def create_objective(
    model_kind: ModelKind,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Callable[[optuna.Trial], float]:
    """Создаёт objective функцию для Optuna.

    Args:
        model_kind: Тип модели
        x_train: Тренировочные данные
        y_train: Тренировочные метки
        x_val: Валидационные данные
        y_val: Валидационные метки

    Returns:
        Objective функция для Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr(
            "numeric_cols", [c for c in NUMERIC_COLS if c in x_train.columns]
        )
        trial.set_user_attr("n_train_samples", len(x_train))
        mlflow.log_param("model", model_kind.value)

        is_distilbert = model_kind is ModelKind.distilbert

        if N_FOLDS > 1:
            return _evaluate_with_cv(trial, model_kind, x_train, y_train, is_distilbert)
        return _evaluate_holdout(
            trial, model_kind, x_train, y_train, x_val, y_val, is_distilbert
        )

    return objective


def _evaluate_with_cv(
    trial: optuna.Trial,
    model_kind: ModelKind,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    is_distilbert: bool,
) -> float:
    """Оценка модели с кросс-валидацией."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    f1_scores: list[float] = []

    builder = ModelBuilderFactory.get_builder(model_kind, trial)

    if is_distilbert:
        texts = x_train["reviewText"].values
        for tr_idx, va_idx in skf.split(texts, y_train):
            x_tr, x_va = texts[tr_idx], texts[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            pipe = builder.build()
            pipe.fit(x_tr, y_tr)
            preds_fold = pipe.predict(x_va)
            f1_scores.append(f1_score(y_va, preds_fold, average="macro"))
    else:
        for tr_idx, va_idx in skf.split(x_train, y_train):
            x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            pipe = builder.build()
            pipe.fit(x_tr, y_tr)
            preds_fold = pipe.predict(x_va)
            f1_scores.append(f1_score(y_va, preds_fold, average="macro"))

    mean_f1 = float(np.mean(f1_scores))
    mlflow.log_metric("cv_f1_macro", mean_f1)
    return mean_f1


def _evaluate_holdout(
    trial: optuna.Trial,
    model_kind: ModelKind,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    is_distilbert: bool,
) -> float:
    """Оценка модели на отложенной выборке."""
    builder = ModelBuilderFactory.get_builder(model_kind, trial)
    pipe = builder.build()

    if is_distilbert:
        pipe.fit(x_train["reviewText"], y_train)
        preds = pipe.predict(x_val["reviewText"])
    else:
        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_val)

    metrics = compute_metrics(y_val, preds)
    mlflow.log_metrics(metrics)
    return metrics["f1_macro"]


def optimize_model(
    study_name: str,
    model_kind: ModelKind,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    storage: str,
) -> optuna.Study:
    """Запускает оптимизацию гиперпараметров.

    Args:
        study_name: Имя Optuna study
        model_kind: Тип модели
        x_train: Тренировочные данные
        y_train: Тренировочные метки
        x_val: Валидационные данные
        y_val: Валидационные метки
        storage: Storage для Optuna

    Returns:
        Оптимизированная Optuna study
    """
    global _interrupted
    _interrupted = False

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    full_study_name = f"{study_name}_{model_kind.value}"
    log.info("Создание/загрузка Optuna study: %s", full_study_name)

    study = optuna.create_study(
        study_name=full_study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=MIN_TRIALS_BEFORE_EARLY_STOP, n_warmup_steps=3
        ),
    )

    objective_fn = create_objective(model_kind, x_train, y_train, x_val, y_val)

    is_distilbert = model_kind is ModelKind.distilbert
    timeout = DISTILBERT_TIMEOUT_SEC if is_distilbert else OPTUNA_TIMEOUT_SEC

    log.info(
        "Оптимизация %s: n_trials=%d, timeout=%ds",
        model_kind.value,
        OPTUNA_N_TRIALS,
        timeout,
    )

    def stop_on_interrupt(study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback для остановки оптимизации при получении сигнала."""
        if _interrupted:
            study.stop()

    try:
        study.optimize(
            objective_fn,
            n_trials=OPTUNA_N_TRIALS,
            timeout=timeout,
            callbacks=[stop_on_interrupt],
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        log.warning("Оптимизация прервана пользователем")

    log.info(
        "Оптимизация %s завершена: всего trials=%d, лучший f1_macro=%.4f, параметры=%s",
        model_kind.value,
        len(study.trials),
        study.best_value,
        {k: v for k, v in study.best_params.items() if not k.startswith("_")},
    )

    return study
