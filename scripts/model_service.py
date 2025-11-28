import json
from typing import Any

import joblib
import pandas as pd

from scripts.config import (
    BASELINE_NUMERIC_STATS_PATH,
    BEST_MODEL_META_PATH,
    BEST_MODEL_PATH,
    MODEL_ARTEFACTS_DIR,
)
from scripts.feature_contract import FeatureContract
from scripts.feature_engineering import transform_features
from scripts.logging_config import get_logger

log = get_logger("model_service")


class ModelService:
    def __init__(self):
        self.model = None
        self.meta = {}
        self.numeric_defaults = {}
        self.feature_contract = None
        self.loaded = False

    def load_artifacts(self) -> None:
        """Загружает артефакты модели, метаданные и контракт."""
        log.info("Загрузка артефактов модели...")

        if not BEST_MODEL_PATH.exists():
            log.warning(
                "Модель не найдена: %s — сервис работает в режиме ожидания",
                BEST_MODEL_PATH,
            )
            self.loaded = False
            return

        try:
            self.model = joblib.load(BEST_MODEL_PATH)
            log.info("Модель загружена: %s", BEST_MODEL_PATH)
        except (OSError, ValueError, EOFError) as e:
            log.exception("Ошибка при загрузке модели: %s", e)
            raise

        # Метаданные
        if BEST_MODEL_META_PATH.exists():
            try:
                self.meta = json.loads(BEST_MODEL_META_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                log.warning("Ошибка чтения метаданных: %s", e)
                self.meta = {}
        else:
            log.warning("Метаданные не найдены: %s", BEST_MODEL_META_PATH)

        # Baseline статистики
        if BASELINE_NUMERIC_STATS_PATH.exists():
            try:
                self.numeric_defaults = json.loads(
                    BASELINE_NUMERIC_STATS_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as e:
                log.warning("Ошибка чтения baseline статистики: %s", e)
                self.numeric_defaults = {}
        else:
            log.warning("Baseline статистики не найдены (дрифт-мониторинг ограничен)")

        # Контракт признаков
        try:
            self.feature_contract = FeatureContract.from_model_artifacts(
                MODEL_ARTEFACTS_DIR
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("Не удалось загрузить контракт признаков: %s", e)
            self.feature_contract = None

        self.loaded = True
        log.info("Все артефакты успешно загружены")

    def predict(
        self, texts: list[str], numeric_features: dict[str, list[float]] | None = None
    ) -> tuple[list[int], list[list[float]] | None, dict[str, list[str]] | None]:
        """
        Выполняет предсказание.
        Возвращает: (labels, probs, warnings)
        """
        if not self.loaded:
            # Попытка ленивой загрузки
            self.load_artifacts()
            if not self.loaded:
                raise RuntimeError("Модель не загружена")

        # Валидация входных данных (проверка длины и т.д. должна быть на уровне API,
        # но здесь проверяем согласованность признаков)
        expected_cols = (
            self.feature_contract.expected_numeric_columns
            if self.feature_contract
            else []
        )

        # Генерация признаков
        df, ignored_features = transform_features(
            texts, numeric_features, list(expected_cols)
        )

        # Предсказание
        is_text_only = bool(getattr(self.model, "text_only", False))

        if is_text_only:
            # Если модель работает только с текстом (например, пайплайн с векторизатором)
            X = pd.Series(texts)
        else:
            X = df

        preds = self.model.predict(X)

        probs = None
        if hasattr(self.model, "predict_proba"):
            try:
                probs_arr = self.model.predict_proba(X)
                probs = [row.tolist() for row in probs_arr]
            except (ValueError, TypeError):
                pass

        warnings = {}
        if ignored_features:
            warnings["ignored_features"] = ignored_features

        return [int(x) for x in preds], probs, warnings or None

    def get_metadata(self) -> dict[str, Any]:
        feature_info = (
            self.feature_contract.get_feature_info() if self.feature_contract else {}
        )

        return {
            "model_info": {
                "best_model": self.meta.get("best_model", "unknown"),
                "best_params": self.meta.get("best_params", {}),
                "test_metrics": self.meta.get("test_metrics", {}),
                "training_duration_sec": self.meta.get("duration_sec", None),
                "dataset_sizes": self.meta.get("sizes", {}),
            },
            "feature_contract": feature_info,
            "health": {
                "model_loaded": self.loaded,
                "baseline_stats_loaded": bool(self.numeric_defaults),
                "feature_contract_loaded": self.feature_contract is not None,
            },
        }
