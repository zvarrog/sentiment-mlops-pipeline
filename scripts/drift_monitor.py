"""Мониторинг дрейфа числовых признаков через PSI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import (
    DRIFT_ARTEFACTS_DIR,
    DRIFT_IGNORE_COLS,
    MIN_SAMPLES_FOR_PSI,
    MODEL_ARTEFACTS_DIR,
    NUMERIC_COLS,
    PROCESSED_DATA_DIR,
    PSI_EPSILON,
)
from scripts.logging_config import get_logger

log = get_logger(__name__)


def _calculate_bin_cuts(data: np.ndarray, bins: int = 10) -> np.ndarray:
    """Вычисляет границы бинов по квантилям для PSI."""
    clean_data = data[~np.isnan(data)]
    if len(clean_data) < 2:
        return np.array([])
    q = np.linspace(0, 1, bins + 1)
    return np.unique(np.quantile(clean_data, q))


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    cuts: np.ndarray | None = None,
) -> float:
    """Вычисляет Population Stability Index между двумя распределениями.

    Args:
        expected (np.ndarray): Базовое (эталонное) распределение признака.
        actual (np.ndarray): Актуальное распределение для проверки на дрейф.
        bins (int): Количество бинов для гистограммы. По умолчанию 10.
        cuts (np.ndarray | None): Заранее вычисленные границы бинов.

    Returns:
        float: Значение PSI. >0.2 указывает на значимый дрейф.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < MIN_SAMPLES_FOR_PSI or len(actual) < MIN_SAMPLES_FOR_PSI:
        return 0.0

    # Единые бины: по квантилям expected (train)
    if cuts is None:
        cuts = _calculate_bin_cuts(expected, bins=bins)

    if len(cuts) <= 2:
        return 0.0

    exp_counts = np.histogram(expected, bins=cuts)[0].astype(float)
    act_counts = np.histogram(actual, bins=cuts)[0].astype(float)

    # Сглаживание для избежания деления на ноль в логарифме
    exp_pct = (exp_counts + PSI_EPSILON) / (exp_counts.sum() + PSI_EPSILON * len(exp_counts))
    act_pct = (act_counts + PSI_EPSILON) / (act_counts.sum() + PSI_EPSILON * len(act_counts))

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _safe_values(series: pd.Series) -> np.ndarray:
    """Безопасное преобразование Series в numpy array."""
    return series.fillna(0).astype(float).values


def calculate_drift(
    new_df: pd.DataFrame,
    train_df: pd.DataFrame,
    base_stats: dict,
    threshold: float = 0.2,
) -> list[dict]:
    """Вычисляет PSI для всех числовых колонок."""
    report: list[dict] = []
    for col in NUMERIC_COLS:
        if col in DRIFT_IGNORE_COLS:
            continue
        if col not in base_stats:
            continue
        if col not in new_df.columns or col not in train_df.columns:
            continue

        actual = _safe_values(cast(pd.Series, new_df[col]))
        expected = _safe_values(cast(pd.Series, train_df[col]))

        # Используем helper для вычисления cuts, чтобы гарантировать идентичность бинов
        cuts = _calculate_bin_cuts(expected[~np.isnan(expected)], bins=10)
        val = psi(expected, actual, cuts=cuts)

        report.append({"feature": col, "psi": float(val), "drift": bool(val > threshold)})
    return report


def plot_drift_histograms(
    report: list[dict],
    new_df: pd.DataFrame,
    train_df: pd.DataFrame,
    plots_out: Path,
) -> None:
    """Строит и сохраняет гистограммы для признаков с дрейфом."""
    try:
        for item in report:
            col = item.get("feature")
            if not col or col not in new_df.columns or col not in train_df.columns:
                continue

            expected_arr = _safe_values(cast(pd.Series, train_df[col]))
            actual_arr = _safe_values(cast(pd.Series, new_df[col]))

            plt.figure(figsize=(6, 4))
            plt.hist(expected_arr, bins=30, alpha=0.5, label="expected", density=True)
            plt.hist(actual_arr, bins=30, alpha=0.5, label="actual", density=True)
            plt.title(f"PSI={item.get('psi'):.3f} | drift={item.get('drift')} | {col}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_out / f"{col}_hist.png")
            plt.close()
    except (OSError, ValueError) as e:
        log.warning("Не удалось построить гистограммы: %s", e)


def run_drift_monitor(
    new_path: str | Path,
    threshold: float = 0.2,
    save: bool = True,
    out_dir: str | Path | None = None,
) -> list[dict]:
    """Запускает расчёт PSI по числовым признакам.

    Args:
        new_path: Путь к parquet с актуальными данными
        threshold: Порог PSI для флага дрейфа
        save: Сохранить отчёт в drift/drift_report.json
        out_dir: Директория для сохранения (по умолчанию DRIFT_ARTEFACTS_DIR)

    Returns:
        Список словарей: {feature, psi, drift}
    """
    baseline_path = MODEL_ARTEFACTS_DIR / "baseline_numeric_stats.json"
    if not baseline_path.exists():
        log.warning("Дрифт‑монитор: отсутствует %s — мониторинг пропущен", str(baseline_path))
        return []

    try:
        with open(baseline_path, encoding="utf-8") as f:
            base_stats = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.error("Ошибка чтения baseline stats: %s", e)
        return []

    try:
        new_df = pd.read_parquet(new_path)
    except (OSError, ValueError) as e:
        log.error("Ошибка чтения новых данных %s: %s", new_path, e)
        return []

    train_parquet = PROCESSED_DATA_DIR / "train.parquet"
    if not train_parquet.exists():
        log.error("Отсутствует обучающая выборка: %s", train_parquet)
        return []

    try:
        train_df = pd.read_parquet(train_parquet)
    except (OSError, ValueError) as e:
        log.error("Ошибка чтения обучающей выборки %s: %s", train_parquet, e)
        return []

    report = calculate_drift(new_df, train_df, base_stats, threshold)

    if save:
        default_out = DRIFT_ARTEFACTS_DIR
        base_out = Path(out_dir) if out_dir else default_out
        plots_out = base_out / "plots"
        base_out.mkdir(parents=True, exist_ok=True)
        plots_out.mkdir(parents=True, exist_ok=True)

        out_json = base_out / "drift_report.json"
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            log.info(
                "Отчёт дрейфа сохранён: %s (гистограммы: %s)",
                str(out_json),
                str(plots_out),
            )
        except OSError as e:
            log.error("Не удалось сохранить JSON отчет: %s", e)

        try:
            pd.DataFrame(report).to_csv(base_out / "drift_report.csv", index=False)
        except (OSError, ValueError) as e:
            log.warning("Не удалось сохранить CSV: %s", e)

        plot_drift_histograms(report, new_df, train_df, plots_out)

    return report
