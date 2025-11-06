"""Централизованная конфигурация проекта.

Единый источник истины для всех настроек приложения.
Загружает переменные из .env файла и переопределяет через os.environ.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from scripts.logging_config import get_logger
from scripts.models.kinds import ModelKind

# Загружаем .env только если не в Airflow (Airflow использует env_file в docker-compose)
if not os.environ.get("AIRFLOW_HOME"):
    load_dotenv(override=False)


def _getenv_bool(key: str, default: bool = False) -> bool:
    """Читает булеву переменную окружения."""
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off", ""):
        return default if val == "" else False
    return default


def _getenv_int(key: str, default: int) -> int:
    """Читает целочисленную переменную окружения."""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _getenv_path(key: str, default: str) -> Path:
    """Читает путь из переменной окружения."""
    return Path(os.environ.get(key, default))


# Seed
SEED = _getenv_int("SEED", 42)

# Директории
RAW_DATA_DIR = _getenv_path("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = _getenv_path("PROCESSED_DATA_DIR", "data/processed")
MODEL_DIR = _getenv_path("MODEL_DIR", "artefacts")
MODEL_FILE_DIR = MODEL_DIR
MODEL_ARTEFACTS_DIR = _getenv_path(
    "MODEL_ARTEFACTS_DIR", str(MODEL_DIR / "model_artefacts")
)
DRIFT_ARTEFACTS_DIR = _getenv_path(
    "DRIFT_ARTEFACTS_DIR", str(MODEL_DIR / "drift_artefacts")
)

# Датасет
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", "bharadwaj6/kindle-reviews")
CSV_NAME = os.environ.get("CSV_NAME", "kindle_reviews.csv")
JSON_NAME = os.environ.get("JSON_NAME", "kindle_reviews.json")

# Флаги
FORCE_DOWNLOAD = _getenv_bool("FORCE_DOWNLOAD", False)
FORCE_PROCESS = _getenv_bool("FORCE_PROCESS", False)
FORCE_TRAIN = _getenv_bool("FORCE_TRAIN", False)
KEEP_CANDIDATES = _getenv_bool("KEEP_CANDIDATES", False)

# Дрейф и валидация
INJECT_SYNTHETIC_DRIFT = _getenv_bool("INJECT_SYNTHETIC_DRIFT", False)
RUN_DRIFT_MONITOR = _getenv_bool("RUN_DRIFT_MONITOR", False)
RUN_DATA_VALIDATION = _getenv_bool("RUN_DATA_VALIDATION", True)

# Обработка данных
PER_CLASS_LIMIT = 100  # _getenv_int("PER_CLASS_LIMIT", 35000)
HASHING_TF_FEATURES = _getenv_int("HASHING_TF_FEATURES", 6144)
SHUFFLE_PARTITIONS = _getenv_int("SHUFFLE_PARTITIONS", 32)
MIN_DF = _getenv_int("MIN_DF", 3)
MIN_TF = _getenv_int("MIN_TF", 2)

# Оптимизация памяти
MEMORY_WARNING_MB = _getenv_int("MEMORY_WARNING_MB", 3072)
TFIDF_MAX_FEATURES_MIN = _getenv_int("TFIDF_MAX_FEATURES_MIN", 2000)
TFIDF_MAX_FEATURES_MAX = _getenv_int("TFIDF_MAX_FEATURES_MAX", 6000)
TFIDF_MAX_FEATURES_STEP = _getenv_int("TFIDF_MAX_FEATURES_STEP", 500)
FORCE_SVD_THRESHOLD_MB = _getenv_int("FORCE_SVD_THRESHOLD_MB", 4000)

# Обучение
OPTUNA_N_TRIALS = 1  # _getenv_int("OPTUNA_N_TRIALS", 30)
OPTUNA_STORAGE = os.environ.get(
    "OPTUNA_STORAGE", "postgresql+psycopg2://admin:admin@postgres:5432/optuna"
)
STUDY_BASE_NAME = os.environ.get("STUDY_BASE_NAME", "kindle_optuna")
N_FOLDS = _getenv_int("N_FOLDS", 1)
TRAIN_DEVICE = os.environ.get("TRAIN_DEVICE", "cpu")
EARLY_STOP_PATIENCE = _getenv_int("EARLY_STOP_PATIENCE", 8)
OPTUNA_TIMEOUT_SEC = _getenv_int("OPTUNA_TIMEOUT_SEC", 2400)
MIN_TRIALS_BEFORE_EARLY_STOP = _getenv_int("MIN_TRIALS_BEFORE_EARLY_STOP", 1)
DISTILBERT_TIMEOUT_SEC = _getenv_int("DISTILBERT_TIMEOUT_SEC", 1800)
OPTUNA_TOPK_EXPORT = _getenv_int("OPTUNA_TOPK_EXPORT", 20)

# DistilBERT гиперпараметры
DISTILBERT_MIN_EPOCHS = _getenv_int("DISTILBERT_MIN_EPOCHS", 2)
DISTILBERT_MAX_EPOCHS = _getenv_int("DISTILBERT_MAX_EPOCHS", 8)
DISTILBERT_EARLY_STOP_PATIENCE = _getenv_int("DISTILBERT_EARLY_STOP_PATIENCE", 3)

# Модели для обучения
SELECTED_MODEL_KINDS = [
    ModelKind.logreg,
    ModelKind.rf,
    ModelKind.hist_gb,
    ModelKind.mlp,
    ModelKind.distilbert,
]

# MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "kindle_reviews")
MODEL_PRODUCTION_THRESHOLD = 0.75

# Spark ресурсы
SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
SPARK_EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "6g")
SPARK_NUM_CORES = _getenv_int("SPARK_NUM_CORES", 4)

# Database
DB_POOL_SIZE = _getenv_int("DB_POOL_SIZE", 10)
DB_MAX_OVERFLOW = _getenv_int("DB_MAX_OVERFLOW", 20)
DB_POOL_TIMEOUT = _getenv_int("DB_POOL_TIMEOUT", 30)
DB_POOL_RECYCLE = _getenv_int("DB_POOL_RECYCLE", 3600)

# Логирование
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get("LOG_FORMAT", "text")
LOG_INCLUDE_TIMESTAMP = _getenv_bool("LOG_INCLUDE_TIMESTAMP", True)

# Логгер для совместимости
log = get_logger("kindle")


def get_tfidf_max_features_range(n_samples: int) -> tuple[int, int, int]:
    """Рассчитывает диапазон max_features для TF-IDF на основе размера выборки.

    Args:
        n_samples: Количество образцов в датасете.

    Returns:
        Кортеж (min_features, max_features, step) для динамической оптимизации.
    """
    min_val = TFIDF_MAX_FEATURES_MIN
    max_val = TFIDF_MAX_FEATURES_MAX
    step = TFIDF_MAX_FEATURES_STEP

    # Масштабируем диапазон в зависимости от размера датасета
    scaling_factor = max(0.5, min(1.0, n_samples / 100000))
    adjusted_min = max(500, int(min_val * scaling_factor))
    adjusted_max = int(max_val * scaling_factor)

    return adjusted_min, adjusted_max, step
