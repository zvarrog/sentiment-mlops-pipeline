"""Сервис для загрузки модели и выполнения предсказаний."""

import json
import threading
from typing import Any, cast

import joblib

from scripts.config import (
    BASELINE_NUMERIC_STATS_PATH,
    BEST_MODEL_META_PATH,
    BEST_MODEL_PATH,
    MODEL_ARTEFACTS_DIR,
)
from scripts.feature_contract import FeatureContract
from scripts.feature_engineering import transform_features
from scripts.logging_config import get_logger
from scripts.types import (
    BaselineStats,
    DatasetSizes,
    FeatureInfo,
    HealthStatus,
    MetadataResponse,
    ModelInfo,
    ModelMeta,
    TestMetrics,
)

log = get_logger(__name__)

# Значения по умолчанию для пустых метаданных
_EMPTY_META: ModelMeta = {
    "best_model": "unknown",
    "best_params": {},
    "best_val_f1_macro": 0.0,
    "test_metrics": {},
    "sizes": {"train": 0, "val": 0, "test": 0},
}


class ModelService:
    """Сервис загрузки артефактов модели и выполнения предсказаний."""

    def __init__(self) -> None:
        self.model: Any = None
        self.meta: ModelMeta = _EMPTY_META.copy()
        self.numeric_defaults: BaselineStats = {}
        self.feature_contract: FeatureContract | None = None
        self.loaded: bool = False
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return str(self.meta.get("best_model", "unknown"))

    def load_artifacts(self) -> None:
        """Загружает артефакты модели, метаданные и контракт (потокобезопасно)."""
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
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
                    raw_meta = json.loads(BEST_MODEL_META_PATH.read_text(encoding="utf-8"))
                    self.meta = cast(ModelMeta, raw_meta)
                except (OSError, json.JSONDecodeError) as e:
                    log.warning("Ошибка чтения метаданных: %s", e)
                    self.meta = _EMPTY_META.copy()
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

            # Контракт признаков (фатальная ошибка если отсутствует список признаков)
            try:
                self.feature_contract = FeatureContract.from_model_artifacts(MODEL_ARTEFACTS_DIR)
            except (OSError, ValueError, KeyError, TypeError) as e:
                log.warning("Не удалось загрузить контракт признаков: %s", e)
                self.feature_contract = None

            self.loaded = True
            log.info("Все артефакты успешно загружены")

    def predict(
        self, texts: list[str], numeric_features: dict[str, list[float]] | None = None
    ) -> tuple[list[int], list[list[float]] | None, dict[str, list[str]] | None]:
        """Выполняет предсказание. Возвращает (labels, probs, warnings)."""
        if not self.model:
            self.load_artifacts()
            if not self.model:
                raise RuntimeError("Модель не загружена")

        # Валидация входных данных (проверка длины и т.д. должна быть на уровне API,
        # но здесь проверяем согласованность признаков)
        expected_cols = (
            self.feature_contract.expected_numeric_columns if self.feature_contract else []
        )

        # Генерация признаков
        df, ignored_features = transform_features(texts, numeric_features, list(expected_cols))

        # Предсказание: определяем тип входных данных по структуре пайплайна
        from scripts.utils import get_model_input

        X = get_model_input(self.model, df)

        preds = self.model.predict(X)

        probs = None
        if hasattr(self.model, "predict_proba"):
            try:
                probs_arr = self.model.predict_proba(X)
                probs = [row.tolist() for row in probs_arr]
            except (ValueError, TypeError) as e:
                log.debug("predict_proba недоступен: %s", e)

        warnings = {}
        if ignored_features:
            warnings["ignored_features"] = ignored_features

        return [int(x) for x in preds], probs, warnings or None

    def get_metadata(self) -> MetadataResponse:
        """Возвращает метаданные модели и статус сервиса."""
        feature_info = self.feature_contract.get_feature_info() if self.feature_contract else {}

        raw_meta = cast(dict[str, Any], self.meta)
        training_duration = raw_meta.get("duration_sec")
        dataset_sizes = self.meta.get("sizes", {"train": 0, "val": 0, "test": 0})

        model_info: ModelInfo = {
            "best_model": self.meta.get("best_model", "unknown"),
            "best_params": dict(self.meta.get("best_params", {})),
            "test_metrics": cast(TestMetrics, dict(self.meta.get("test_metrics", {}))),
            "training_duration_sec": float(training_duration) if training_duration else None,
            "dataset_sizes": cast(DatasetSizes, dict(dataset_sizes)),
        }

        health: HealthStatus = {
            "model_loaded": self.model is not None,
            "baseline_stats_loaded": bool(self.numeric_defaults),
            "feature_contract_loaded": self.feature_contract is not None,
        }

        return {
            "model_info": model_info,
            "feature_contract": cast(FeatureInfo, feature_info),
            "health": health,
        }
