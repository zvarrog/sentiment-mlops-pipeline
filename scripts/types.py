"""Типизированные структуры данных проекта.

Заменяет dict[str, Any] на строго типизированные TypedDict для:
- Метаданных модели
- Baseline статистик
- Метрик оценки
- Результатов дрейф-мониторинга
- Результатов Optuna-оптимизации
"""

from __future__ import annotations

from typing_extensions import TypedDict


class ColumnStats(TypedDict):
    """Статистики для одной числовой колонки."""

    mean: float
    std: float


class BaselineStats(TypedDict, total=False):
    """Baseline статистики для числовых колонок.

    Ключи — имена колонок из NUMERIC_COLS.
    Используется как dict[str, ColumnStats] для динамического доступа.
    """

    text_len: ColumnStats
    word_count: ColumnStats
    kindle_freq: ColumnStats
    sentiment: ColumnStats
    user_avg_len: ColumnStats
    user_review_count: ColumnStats
    item_avg_len: ColumnStats
    item_review_count: ColumnStats
    exclamation_count: ColumnStats
    caps_ratio: ColumnStats
    question_count: ColumnStats
    avg_word_length: ColumnStats


class TestMetrics(TypedDict, total=False):
    """Метрики оценки модели на тестовой выборке."""

    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float


class DatasetSizes(TypedDict):
    """Размеры выборок данных."""

    train: int
    val: int
    test: int


class ModelMeta(TypedDict, total=False):
    """Метаданные лучшей модели.

    Все поля опциональны, так как данные могут быть частично загружены из JSON.
    """

    best_model: str
    best_params: dict[str, float | int | str | bool]
    best_val_f1_macro: float
    test_metrics: TestMetrics
    sizes: DatasetSizes
    duration_sec: float  # опционально — присутствует в сохранённых артефактах


class DriftReportItem(TypedDict):
    """Элемент отчёта о дрейфе признака."""

    feature: str
    psi: float
    drift: bool
    threshold: float


class ModelInfo(TypedDict, total=False):
    """Информация о модели для API metadata."""

    best_model: str
    best_params: dict[str, float | int | str | bool]
    test_metrics: TestMetrics
    training_duration_sec: float | None
    dataset_sizes: DatasetSizes


class HealthStatus(TypedDict):
    """Статус здоровья сервиса."""

    model_loaded: bool
    baseline_stats_loaded: bool
    feature_contract_loaded: bool


class FeatureInfo(TypedDict, total=False):
    """Информация о признаках из контракта."""

    required_text_columns: list[str]
    expected_numeric_columns: list[str]
    baseline_available: bool


class MetadataResponse(TypedDict):
    """Полный ответ /metadata эндпоинта."""

    model_info: ModelInfo
    feature_contract: FeatureInfo
    health: HealthStatus


class TrialResult(TypedDict):
    """Результат одного trial Optuna-оптимизации."""

    best_value: float
    best_params: dict[str, float | int | str | bool]
    study_name: str
