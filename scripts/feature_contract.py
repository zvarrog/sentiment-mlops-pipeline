"""Feature contract для валидации входных данных и признаков модели."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_TEXT_COL = "reviewText"


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
    def from_model_artifacts(cls, model_artefact_dir: Path) -> "FeatureContract":
        """Создаёт контракт из baseline_numeric_stats.json или model_schema.json."""
        baseline_path = model_artefact_dir / "baseline_numeric_stats.json"
        schema_path = model_artefact_dir / "model_schema.json"
        baseline_stats = None
        expected_numeric: list[str] = []

        # Загружаем baseline
        if baseline_path.exists():
            try:
                with open(baseline_path, encoding="utf-8") as f:
                    baseline_stats = json.load(f)
            except Exception:
                baseline_stats = None
            # Ориентируемся на ключи baseline (реально использованные признаки)
            if isinstance(baseline_stats, dict) and baseline_stats:
                expected_numeric = [str(k) for k in baseline_stats]

        # Фактически использованные числовые признаки из схемы
        if schema_path.exists():
            try:
                with open(schema_path, encoding="utf-8") as f:
                    schema = json.load(f)
                inp = schema.get("input", {})
                # Классические модели сохраняют список под ключом numeric_features
                used = inp.get("numeric_features")
                if isinstance(used, list) and used:
                    expected_numeric = [str(x) for x in used]
            except Exception:
                pass

        # Валидация: список признаков обязателен
        if not expected_numeric:
            raise RuntimeError(
                f"Не удалось определить список числовых признаков из артефактов в {model_artefact_dir}. "
                f"Проверьте наличие {baseline_path.name} или {schema_path.name}"
            )

        return cls(
            required_text_columns=[REQUIRED_TEXT_COL],
            expected_numeric_columns=expected_numeric,
            baseline_stats=baseline_stats,
        )

    def validate_input_data(self, data: dict[str, Any]) -> dict[str, list[str]]:
        """Валидирует входные данные и возвращает предупреждения."""
        warnings = {}

        # Проверка обязательных текстовых колонок
        missing_text = []
        for col in self.required_text_columns:
            if col not in data:
                missing_text.append(col)
        if missing_text:
            warnings["missing_required_columns"] = missing_text

        # Проверка числовых колонок
        missing_numeric = []
        outlier_warnings = []

        for col in self.expected_numeric_columns:
            if col not in data:
                missing_numeric.append(col)
            elif self.baseline_stats and col in self.baseline_stats:
                # Выбросы (3 сигмы)
                baseline = self.baseline_stats[col]
                mean = baseline.get("mean", 0.0)
                std = baseline.get("std", 1.0)

                if isinstance(data[col], (list, tuple)):
                    values = data[col]
                else:
                    values = [data[col]]

                for i, val in enumerate(values):
                    if (
                        isinstance(val, (int, float))
                        and std > 0
                        and abs(val - mean) > 3 * std
                    ):
                        outlier_warnings.append(
                            f"{col}[{i}]: {val} (expected ~{mean:.2f}±{3 * std:.2f})"
                        )

        if missing_numeric:
            warnings["missing_numeric_columns"] = missing_numeric
        if outlier_warnings:
            warnings["potential_outliers"] = outlier_warnings

        return warnings

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
