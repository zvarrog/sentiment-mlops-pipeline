"""Общие утилиты для извлечения параметров запуска и флагов."""

from __future__ import annotations

import json
import os
from pathlib import Path
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


def get_value(context: dict, name: str, default: str | None = None) -> str:
    """Извлекает строковое значение из контекста Airflow (params, dag_run.conf, dag.params)."""
    params = context.get("params") or {}
    if name in params:
        return str(params.get(name))
    dag_run = context.get("dag_run")
    if getattr(dag_run, "conf", None) and name in dag_run.conf:
        return str(dag_run.conf.get(name))
    dag = context.get("dag")
    if dag and name in getattr(dag, "params", {}):
        return str(dag.params.get(name))
    return str(default or "")


def get_flag(context: dict, name: str, default: bool = False) -> bool:
    """Извлекает булево значение из контекста Airflow."""
    params = context.get("params") or {}
    if name in params:
        return to_bool(params.get(name), default)
    dag_run = context.get("dag_run")
    if getattr(dag_run, "conf", None) and name in dag_run.conf:
        return to_bool(dag_run.conf.get(name), default)
    dag = context.get("dag")
    if dag and name in getattr(dag, "params", {}):
        return to_bool(dag.params.get(name), default)
    return bool(default)


def atomic_write_json(path: str | Path, data: dict, **kwargs: Any) -> None:
    """Атомарная запись JSON-файла через временный файл."""
    path_obj = Path(path)
    temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=kwargs.get("indent", 2))
        os.replace(temp_path, path_obj)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_parquet(path: str | Path, df: pd.DataFrame, **kwargs: Any) -> None:
    """Атомарная запись Parquet-файла через временный файл."""
    path_obj = Path(path)
    temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")

    try:
        df.to_parquet(temp_path, **kwargs)
        os.replace(temp_path, path_obj)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
