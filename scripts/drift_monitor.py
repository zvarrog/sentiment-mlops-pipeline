"""Простой мониторинг дрейфа числовых признаков (PSI)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_modules.feature_space import NUMERIC_COLS


# Берём пути к артефактам динамически из окружения
def _resolve_dirs() -> tuple[Path, Path]:
    """Возвращает (MODEL_ARTEFACTS_DIR, DRIFT_ARTEFACTS_DIR) на основе окружения.

    Приоритет: переменные окружения MODEL_ARTEFACTS_DIR/DRIFT_ARTEFACTS_DIR,
    иначе строим от MODEL_DIR (по умолчанию 'artefacts').
    """
    model_dir = Path(os.getenv("MODEL_DIR", "artefacts"))
    model_arts = Path(
        os.getenv("MODEL_ARTEFACTS_DIR", str(model_dir / "model_artefacts"))
    )
    drift_arts = Path(
        os.getenv("DRIFT_ARTEFACTS_DIR", str(model_dir / "drift_artefacts"))
    )
    return model_arts, drift_arts


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
    # Защита от NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
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


def run_drift_monitor(
    new_path: str | Path,
    threshold: float = 0.2,
    save: bool = True,
    out_dir: str | Path | None = None,
) -> list[dict]:
    """Запускает расчёт PSI по числовым признакам и возвращает отчёт.

    Параметры:
      - new_path: путь к parquet с актуальными данными (например, test.parquet)
      - threshold: порог PSI для флага дрейфа
      - save: если True — сохранить отчёт в drift/drift_report.json

    Возвращает список словарей: {feature, psi, drift}.
    """
    MODEL_ARTEFACTS_DIR, DRIFT_ARTEFACTS_DIR = _resolve_dirs()
    processed_dir = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    baseline_path = MODEL_ARTEFACTS_DIR / "baseline_numeric_stats.json"
    if not baseline_path.exists():
        import logging

        logging.getLogger(__name__).warning(
            "Дрифт‑монитор: отсутствует %s — мониторинг пропущен", str(baseline_path)
        )
        return []

    with open(baseline_path, encoding="utf-8") as f:
        base_stats = json.load(f)

    new_df = pd.read_parquet(new_path)

    # Пытаемся использовать реальное train распределение вместо синтетической нормали
    train_parquet = processed_dir / "train.parquet"
    train_df: pd.DataFrame | None = None
    if train_parquet.exists():
        try:
            train_df = pd.read_parquet(train_parquet)
        except Exception as e:
            raise RuntimeError(
                f"Не удалось прочитать train.parquet для дрейф-мониторинга: {e}"
            )
    else:
        # Требуем наличие обучающей выборки для корректного сравнения
        raise FileNotFoundError(
            f"Отсутствует обучающая выборка для мониторинга дрейфа: {train_parquet}"
        )

    # Игнорируемые колонки
    ignore_cols = set(
        [c.strip() for c in os.getenv("DRIFT_IGNORE_COLS", "").split(",") if c.strip()]
    )

    # Импутация как в пайплайне модели: пропуски -> 0
    def _safe_values(series: pd.Series) -> np.ndarray:
        return series.fillna(0).astype(float).values

    report: list[dict] = []
    for col in NUMERIC_COLS:
        if col not in new_df.columns or col not in base_stats or col in ignore_cols:
            continue
        actual = _safe_values(new_df[col])
        # expected: по возможности используем train.parquet; иначе — синтетическая нормаль на основе baseline
        if col in train_df.columns:
            expected = _safe_values(train_df[col])
            # Единые бины из train, чтобы корректно сравнивать
            q = np.linspace(0, 1, 11)
            cuts = np.unique(np.quantile(expected[~np.isnan(expected)], q))
            val = psi(expected, actual, cuts=cuts)
        else:
            # Если конкретной колонки нет в train — это повод для пересмотра фичконтракта
            raise KeyError(
                f"Колонка '{col}' отсутствует в train.parquet — дрейф-мониторинг невозможен"
            )
        report.append(
            {"feature": col, "psi": float(val), "drift": bool(val > threshold)}
        )

    if save:
        default_out = DRIFT_ARTEFACTS_DIR
        base_out = Path(out_dir) if out_dir else default_out
        plots_out = base_out / "plots"
        base_out.mkdir(parents=True, exist_ok=True)
        plots_out.mkdir(parents=True, exist_ok=True)

        # JSON отчёт
        out_json = base_out / "drift_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        try:
            import logging

            logging.getLogger(__name__).info(
                "Отчёт дрейфа сохранён: %s (гистограммы: %s)",
                str(out_json),
                str(plots_out),
            )
        except Exception:
            pass

        # CSV отчёт
        try:
            import pandas as _pd

            _pd.DataFrame(report).to_csv(base_out / "drift_report.csv", index=False)
        except Exception:
            pass

        # Гистограммы для каждой фичи
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt

            new_df = pd.read_parquet(new_path)
            for item in report:
                col = item.get("feature")
                if not col or col not in new_df.columns:
                    continue
                # Ожидаемое: train.parquet (если доступен), иначе — синтетическая нормаль
                expected_arr: np.ndarray | None = None
                if train_parquet.exists():
                    try:
                        _trdf = pd.read_parquet(train_parquet)
                        if col in _trdf.columns:
                            expected_arr = _trdf[col].fillna(0).astype(float).values
                    except Exception:
                        expected_arr = None
                if expected_arr is None:
                    _bspath = _resolve_dirs()[0] / "baseline_numeric_stats.json"
                    with open(_bspath, encoding="utf-8") as f:
                        base_stats = json.load(f)
                    if col not in base_stats:
                        continue
                    mean = base_stats[col]["mean"]
                    std = base_stats[col]["std"] or 1e-6
                    expected_arr = np.random.normal(
                        mean, std, size=min(5000, len(new_df))
                    )
                actual = new_df[col].fillna(0).astype(float).values

                _plt.figure(figsize=(6, 4))
                _plt.hist(
                    expected_arr, bins=30, alpha=0.5, label="expected", density=True
                )
                _plt.hist(actual, bins=30, alpha=0.5, label="actual", density=True)
                _plt.title(
                    f"PSI={item.get('psi'):.3f} | drift={item.get('drift')} | {col}"
                )
                _plt.legend()
                _plt.tight_layout()
                _plt.savefig(plots_out / f"{col}_hist.png")
                _plt.close()
        except Exception:
            pass
    return report
