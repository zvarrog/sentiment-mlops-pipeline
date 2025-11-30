"""Общие утилиты для извлечения параметров запуска и флагов."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scripts.config import NUMERIC_COLS


def get_baseline_stats(x_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Вычисляет базовые статистики (mean, std) для числовых колонок."""
    baseline_stats: dict[str, dict[str, float]] = {}
    for col in NUMERIC_COLS:
        if col not in x_train.columns:
            continue
        series = x_train[col]
        baseline_stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std() or 0.0),
        }
    return baseline_stats


def to_bool(x: Any, default: bool = False) -> bool:
    """Преобразует значение в булево."""
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off", ""}:
            return False
        return bool(default)
    if isinstance(x, (int, float)):
        return bool(x)
    return bool(default)


def get_value(context: dict[str, Any], name: str, default: str | None = None) -> str:
    """Извлекает строковое значение из контекста Airflow (params, dag_run.conf, dag.params)."""
    params = context.get("params") or {}
    if name in params:
        return str(params.get(name))
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    if conf is not None and name in conf:
        return str(conf.get(name))
    dag = context.get("dag")
    if dag and name in getattr(dag, "params", {}):
        return str(dag.params.get(name))
    return str(default or "")


def get_flag(context: dict[str, Any], name: str, default: bool = False) -> bool:
    """Извлекает булево значение из контекста Airflow.

    Использует get_value как единый источник извлечения строки,
    затем приводит к булеву через to_bool.
    """
    raw = get_value(context, name, str(default).lower())
    return to_bool(raw, default)


def get_model_input(pipeline, x_data: pd.DataFrame) -> pd.DataFrame | pd.Series:
    """Определяет правильный input для модели.

    Пайплайны с препроцессором ('pre') принимают полный DataFrame,
    остальные (text-only) — только колонку reviewText.

    Args:
        pipeline: Обученный sklearn Pipeline.
        x_data: DataFrame с данными.

    Returns:
        DataFrame или Series в зависимости от типа пайплайна.
    """
    if "pre" in getattr(pipeline, "named_steps", {}):
        return x_data
    return x_data["reviewText"]
