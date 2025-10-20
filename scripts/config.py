"""Централизованная конфигурация проекта через Pydantic Settings.

Единый источник истины для всех настроек приложения.
Поддерживает загрузку из .env файла и переопределение через переменные окружения.
"""

from pathlib import Path
from typing import List

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from scripts.logging_config import get_logger
from scripts.models.kinds import ModelKind


class Settings(BaseSettings):
    """Настройки проекта."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # === Директории ===
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    model_dir: Path = Field(default=Path("artefacts"))

    @computed_field
    @property
    def model_artefacts_dir(self) -> Path:
        """Директория для артефактов модели."""
        return self.model_dir / "model_artefacts"

    @computed_field
    @property
    def drift_artefacts_dir(self) -> Path:
        """Директория для артефактов дрейфа."""
        return self.model_dir / "drift_artefacts"

    # === Датасет ===
    kaggle_dataset: str = Field(default="bharadwaj6/kindle-reviews")
    csv_name: str = Field(default="kindle_reviews.csv")
    json_name: str = Field(default="kindle_reviews.json")

    # === Флаги ===
    force_download: bool = Field(default=False)
    force_process: bool = Field(default=False)
    force_train: bool = Field(default=False)

    # === Флаги дрейфа и валидации ===
    inject_synthetic_drift: bool = Field(default=False)
    run_drift_monitor: bool = Field(default=False)
    run_data_validation: bool = Field(default=True)

    # === Обработка данных ===
    per_class_limit: int = Field(default=35000)
    hashing_tf_features: int = Field(default=6144)
    shuffle_partitions: int = Field(default=32)
    min_df: int = Field(default=3)
    min_tf: int = Field(default=2)
    seed: int = Field(default=42)

    # === Оптимизация памяти ===
    memory_warning_mb: int = Field(default=3072)
    tfidf_max_features_min: int = Field(default=2000)
    tfidf_max_features_max: int = Field(default=6000)
    tfidf_max_features_step: int = Field(default=500)
    force_svd_threshold_mb: int = Field(default=4000)

    # === Обучение ===
    optuna_n_trials: int = Field(default=30)
    optuna_storage: str = Field(default="sqlite:///optuna_study.db")
    study_base_name: str = Field(default="kindle_optuna")
    n_folds: int = Field(default=1)
    train_device: str = Field(default="cpu")
    early_stop_patience: int = Field(default=8)
    optuna_timeout_sec: int = Field(default=2400)
    min_trials_before_early_stop: int = Field(default=15)
    distilbert_timeout_sec: int = Field(default=1800)
    optuna_topk_export: int = Field(default=20)

    # === MLflow ===
    mlflow_tracking_uri: str = Field(default="http://mlflow:5000")
    mlflow_experiment_name: str = Field(default="kindle_reviews")

    # === Spark ресурсы ===
    spark_driver_memory: str = Field(default="6g")
    spark_executor_memory: str = Field(default="6g")
    spark_num_cores: int = Field(default=4)

    # === Логирование ===
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="text")
    log_include_timestamp: bool = Field(default=True)

    @computed_field
    @property
    def selected_model_kinds(self) -> List[ModelKind]:
        """Модели для обучения."""
        return [
            ModelKind.logreg,
            ModelKind.rf,
            ModelKind.hist_gb,
            ModelKind.mlp,
            ModelKind.distilbert,
        ]


# Глобальный экземпляр настроек
settings = Settings()

# Обратная совместимость: экспортируем константы для старого кода
SEED = settings.seed
RAW_DATA_DIR = settings.raw_data_dir
PROCESSED_DATA_DIR = settings.processed_data_dir
MODEL_DIR = settings.model_dir
MODEL_FILE_DIR = settings.model_dir
MODEL_ARTEFACTS_DIR = settings.model_artefacts_dir
DRIFT_ARTEFACTS_DIR = settings.drift_artefacts_dir

KAGGLE_DATASET = settings.kaggle_dataset
CSV_NAME = settings.csv_name
JSON_NAME = settings.json_name

FORCE_DOWNLOAD = settings.force_download
FORCE_PROCESS = settings.force_process
FORCE_TRAIN = settings.force_train

INJECT_SYNTHETIC_DRIFT = settings.inject_synthetic_drift
RUN_DRIFT_MONITOR = settings.run_drift_monitor

PER_CLASS_LIMIT = settings.per_class_limit
HASHING_TF_FEATURES = settings.hashing_tf_features
SHUFFLE_PARTITIONS = settings.shuffle_partitions
MIN_DF = settings.min_df
MIN_TF = settings.min_tf

MEMORY_WARNING_MB = settings.memory_warning_mb
TFIDF_MAX_FEATURES_MIN = settings.tfidf_max_features_min
TFIDF_MAX_FEATURES_MAX = settings.tfidf_max_features_max
TFIDF_MAX_FEATURES_STEP = settings.tfidf_max_features_step
FORCE_SVD_THRESHOLD_MB = settings.force_svd_threshold_mb

SELECTED_MODEL_KINDS = settings.selected_model_kinds
OPTUNA_N_TRIALS = settings.optuna_n_trials
OPTUNA_STORAGE = settings.optuna_storage
STUDY_BASE_NAME = settings.study_base_name
N_FOLDS = settings.n_folds
TRAIN_DEVICE = settings.train_device
EARLY_STOP_PATIENCE = settings.early_stop_patience
OPTUNA_TIMEOUT_SEC = settings.optuna_timeout_sec
MIN_TRIALS_BEFORE_EARLY_STOP = settings.min_trials_before_early_stop
DISTILBERT_TIMEOUT_SEC = settings.distilbert_timeout_sec

SPARK_DRIVER_MEMORY = settings.spark_driver_memory
SPARK_EXECUTOR_MEMORY = settings.spark_executor_memory
SPARK_NUM_CORES = settings.spark_num_cores

# Логгер для совместимости
log = get_logger("kindle")
