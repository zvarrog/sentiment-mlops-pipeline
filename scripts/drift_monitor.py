"""Мониторинг дрейфа числовых признаков через PSI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scripts.config import (
    DRIFT_ARTEFACTS_DIR,
    MIN_SAMPLES_FOR_PSI,
    MODEL_ARTEFACTS_DIR,
    PROCESSED_DATA_DIR,
)
from scripts.constants import NUMERIC_COLS
from scripts.logging_config import get_logger

log = get_logger("drift_monitor")


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    cuts: np.ndarray | None = None,
) -> float:
    """Вычисляет Population Stability Index между двумя распределениями.

    Args:
        expected: Базовое (эталонное) распределение признака
        actual: Актуальное распределение признака для проверки на дрейф
        bins: Количество бинов для гистограммы (по умолчанию 10)
        cuts: Необязательные заранее вычисленные границы бинов. Если заданы, bins игнорируется.

    Returns:
        float: Значение PSI (>0.2 обычно указывает на значимый дрейф)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < MIN_SAMPLES_FOR_PSI or len(actual) < MIN_SAMPLES_FOR_PSI:
        return 0.0
    # Единые бины: по квантилям expected (train)
    if cuts is None:
        quantiles = np.linspace(0, 1, bins + 1)
        cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) <= 2:
        return 0.0
    exp_counts = np.histogram(expected, bins=cuts)[0].astype(float)
    act_counts = np.histogram(actual, bins=cuts)[0].astype(float)
    exp_pct = (exp_counts + 1e-6) / (exp_counts.sum() + 1e-6 * len(exp_counts))
    act_pct = (act_counts + 1e-6) / (act_counts.sum() + 1e-6 * len(act_counts))
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def send_slack_alert(message: str, webhook_url: str | None = None):
    """Отправляет алерт в Slack при обнаружении дрейфа."""
    webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        log.warning("SLACK_WEBHOOK_URL не установлен, алерт не отправлен")
        return

    msg = message.replace("@", "＠").replace("#", "＃")
    payload = {
        "text": f"[DRIFT ALERT]\n{msg}",
        "username": "Drift Monitor",
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        response.raise_for_status()
        log.info("Slack alert sent successfully")
    except requests.RequestException as e:
        log.error("Failed to send Slack alert: %s", e)


def run_drift_monitor(
    new_path: str | Path,
    threshold: float = 0.2,
    save: bool = True,
    out_dir: str | Path | None = None,
    alert_on_drift: bool = True,
) -> list[dict]:
    """Запускает расчёт PSI по числовым признакам.

    Args:
        new_path: Путь к parquet с актуальными данными
        threshold: Порог PSI для флага дрейфа
        save: Сохранить отчёт в drift/drift_report.json
        out_dir: Директория для сохранения (по умолчанию DRIFT_ARTEFACTS_DIR)
        alert_on_drift: Отправить алерт в Slack при дрейфе

    Returns:
        Список словарей: {feature, psi, drift}
    """
    baseline_path = MODEL_ARTEFACTS_DIR / "baseline_numeric_stats.json"
    if not baseline_path.exists():
        log.warning(
            "Дрифт‑монитор: отсутствует %s — мониторинг пропущен", str(baseline_path)
        )
        return []

    with open(baseline_path, encoding="utf-8") as f:
        base_stats = json.load(f)

    new_df = pd.read_parquet(new_path)

    train_parquet = PROCESSED_DATA_DIR / "train.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(
            f"Отсутствует обучающая выборка для мониторинга дрейфа: {train_parquet}"
        )

    train_df = pd.read_parquet(train_parquet)

    # Игнорируемые колонки
    ignore_cols = {
        c.strip() for c in os.getenv("DRIFT_IGNORE_COLS", "").split(",") if c.strip()
    }

    def _safe_values(series: pd.Series) -> np.ndarray:
        return series.fillna(0).astype(float).values

    report: list[dict] = []
    for col in NUMERIC_COLS:
        if col not in new_df.columns or col not in base_stats or col in ignore_cols:
            continue
        actual = _safe_values(new_df[col])

        if col not in train_df.columns:
            raise KeyError(
                f"Колонка '{col}' отсутствует в train.parquet — дрейф-мониторинг невозможен"
            )

        expected = _safe_values(train_df[col])
        q = np.linspace(0, 1, 11)
        cuts = np.unique(np.quantile(expected[~np.isnan(expected)], q))
        val = psi(expected, actual, cuts=cuts)

        report.append(
            {"feature": col, "psi": float(val), "drift": bool(val > threshold)}
        )

    if save:
        default_out = DRIFT_ARTEFACTS_DIR
        base_out = Path(out_dir) if out_dir else default_out
        plots_out = base_out / "plots"
        base_out.mkdir(parents=True, exist_ok=True)
        plots_out.mkdir(parents=True, exist_ok=True)

        out_json = base_out / "drift_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(
            "Отчёт дрейфа сохранён: %s (гистограммы: %s)",
            str(out_json),
            str(plots_out),
        )

        try:
            pd.DataFrame(report).to_csv(base_out / "drift_report.csv", index=False)
        except (OSError, ValueError) as e:
            log.warning("Не удалось сохранить CSV: %s", e)

        try:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("Agg")

            for item in report:
                col = item.get("feature")
                if not col or col not in new_df.columns or col not in train_df.columns:
                    continue

                expected_arr = train_df[col].fillna(0).astype(float).values
                actual = new_df[col].fillna(0).astype(float).values

                plt.figure(figsize=(6, 4))
                plt.hist(
                    expected_arr, bins=30, alpha=0.5, label="expected", density=True
                )
                plt.hist(actual, bins=30, alpha=0.5, label="actual", density=True)
                plt.title(
                    f"PSI={item.get('psi'):.3f} | drift={item.get('drift')} | {col}"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_out / f"{col}_hist.png")
                plt.close()
        except (OSError, ValueError, ImportError) as e:
            log.warning("Не удалось построить гистограммы: %s", e)

    # Отправляем Slack алерт при обнаружении дрейфа
    drifted = [r for r in report if r.get("drift")]
    if drifted and alert_on_drift:
        features = ", ".join(r.get("feature", "?") for r in drifted)
        message = f"Обнаружен дрейф по {len(drifted)} признакам: {features}"
        send_slack_alert(message)

    return report
