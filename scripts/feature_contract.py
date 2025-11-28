"""Контракт признаков: загрузка артефактов и валидация без широких except."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_TEXT_COL = "reviewText"
OUTLIER_SIGMAS = 3  # число сигм для детекции выбросов


@dataclass
class FeatureContract:
    """Контракт признаков для валидации входных данных."""

    # Обязательные текстовые колонки
    required_text_columns: list[str]
    # Ожидаемые числовые колонки (могут быть заполнены дефолтами)
    expected_numeric_columns: list[str]
    # Базовые статистики для валидации числовых признаков
    baseline_stats: dict[str, dict[str, float]] | None = None

    @classmethod
    def from_model_artifacts(
        cls,
        model_artefact_dir: Path,
        baseline_filename: str = "baseline_numeric_stats.json",
        schema_filename: str = "model_schema.json",
    ) -> "FeatureContract":
        """Строит контракт на основе baseline_numeric_stats.json и/или model_schema.json."""
        baseline_path = model_artefact_dir / baseline_filename
        schema_path = model_artefact_dir / schema_filename
        baseline_stats: dict[str, dict[str, float]] | None = None
        expected_numeric: list[str] = []

        if baseline_path.exists():
            try:
                baseline_stats = json.loads(baseline_path.read_text(encoding="utf-8"))
                if isinstance(baseline_stats, dict) and baseline_stats:
                    expected_numeric = [str(k) for k in baseline_stats]
            except (OSError, json.JSONDecodeError):
                baseline_stats = None

        if schema_path.exists():
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                inp = schema.get("input", {}) if isinstance(schema, dict) else {}
                used = inp.get("numeric_features") if isinstance(inp, dict) else None
                if isinstance(used, list) and used:
                    # Если уже загрузили из baseline, объединяем (хотя они должны совпадать)
                    expected_numeric = list(
                        set(expected_numeric) | {str(x) for x in used}
                    )
            except (OSError, json.JSONDecodeError):
                pass

        if not expected_numeric:
            raise RuntimeError(
                "Отсутствует список числовых признаков. Нужен хотя бы один из файлов: "
                f"{baseline_path.name} или {schema_path.name}"
            )

        return cls([REQUIRED_TEXT_COL], sorted(expected_numeric), baseline_stats)

    def validate_input_data(
        self, data: dict[str, Any] | pd.DataFrame
    ) -> dict[str, list[str]]:
        """Возвращает словарь предупреждений по входным данным."""
        if isinstance(data, pd.DataFrame):
            data = {c: data[c].tolist() for c in data.columns}

        issues: dict[str, list[str]] = {}

        missing_text = [c for c in self.required_text_columns if c not in data]
        if missing_text:
            issues["missing_required_columns"] = missing_text

        missing_numeric: list[str] = []
        invalid_types: list[str] = []
        outliers: list[str] = []

        for col in self.expected_numeric_columns:
            if col not in data:
                missing_numeric.append(col)
                continue
            raw = data[col]
            values = raw if isinstance(raw, (list, tuple)) else [raw]
            for i, v in enumerate(values):
                if not isinstance(v, (int, float)):
                    invalid_types.append(f"{col}[{i}] -> {type(v).__name__}")
            if self.baseline_stats and col in self.baseline_stats:
                baseline = self.baseline_stats[col]
                mean = baseline.get("mean", 0.0)
                std = baseline.get("std", 1.0)
                if std > 0:
                    for i, v in enumerate(values):
                        if (
                            isinstance(v, (int, float))
                            and abs(v - mean) > OUTLIER_SIGMAS * std
                        ):
                            outliers.append(
                                f"{col}[{i}]: {v} (≈{mean:.2f}±{OUTLIER_SIGMAS * std:.2f})"
                            )

        if missing_numeric:
            issues["missing_numeric_columns"] = missing_numeric
        if invalid_types:
            issues["invalid_types"] = invalid_types
        if outliers:
            issues["potential_outliers"] = outliers
        return issues

    def get_feature_info(self) -> dict[str, Any]:
        """Возвращает информацию о контракте признаков."""
        info = {
            "required_text_columns": self.required_text_columns,
            "expected_numeric_columns": self.expected_numeric_columns,
            "total_features": len(self.required_text_columns)
            + len(self.expected_numeric_columns),
        }

        if self.baseline_stats:
            info["baseline_stats_available"] = True
            info["baseline_features"] = list(self.baseline_stats.keys())
        else:
            info["baseline_stats_available"] = False

        return info
