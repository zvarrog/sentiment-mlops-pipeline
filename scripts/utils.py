"""Общие утилиты для извлечения параметров запуска и флагов."""

from __future__ import annotations

from typing import Any

import pandas as pd


def get_baseline_stats(x_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    from scripts.config import NUMERIC_COLS

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
    return bool(x)


def get_value(context: dict, name: str, default: str | None = None) -> str:
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


def atomic_write_json(path, data: dict, **kwargs):
    """Атомарная запись JSON-файла через временный файл."""
    import json
    import os
    from pathlib import Path

    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=kwargs.get("indent", 2))
        # Атомарное переименование (на POSIX это атомарно)
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
