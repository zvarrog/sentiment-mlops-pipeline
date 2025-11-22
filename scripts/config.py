"""Централизованная конфигурация проекта.

Единый источник истины для всех настроек приложения.
Загружает переменные из .env файла и переопределяет через os.environ.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from scripts.models.kinds import ModelKind

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artefacts"
DEFAULT_MODEL_ARTEFACTS_SUBDIR = "model_artefacts"
DEFAULT_DRIFT_ARTEFACTS_SUBDIR = "drift_artefacts"

DEFAULT_KAGGLE_DATASET = "bharadwaj6/kindle-reviews"
DEFAULT_CSV_NAME = "kindle_reviews.csv"

BEST_MODEL_FILENAME = "best_model.joblib"
BEST_MODEL_META_FILENAME = "best_model_meta.json"
BASELINE_NUMERIC_STATS_FILENAME = "baseline_numeric_stats.json"

NUMERIC_COLS: list[str] = [
    "text_len",
    "word_count",
    "kindle_freq",
    "sentiment",
    "user_avg_len",
    "user_review_count",
    "item_avg_len",
    "item_review_count",
    "exclamation_count",
    "caps_ratio",
    "question_count",
    "avg_word_length",
]

# Загружаем .env только если не в Airflow (Airflow использует env_file в docker-compose)
if not os.environ.get("AIRFLOW_HOME"):
    load_dotenv(override=False)


def _getenv_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no", "off", ""):
        return default if val == "" else False
    return default


def _getenv_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _getenv_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _getenv_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


SEED = _getenv_int("SEED", 42)

RAW_DATA_DIR = _getenv_path("RAW_DATA_DIR", str(DEFAULT_RAW_DATA_DIR))
PROCESSED_DATA_DIR = _getenv_path("PROCESSED_DATA_DIR", str(DEFAULT_PROCESSED_DATA_DIR))
MODEL_DIR = _getenv_path("MODEL_DIR", str(DEFAULT_MODEL_DIR))
MODEL_ARTEFACTS_DIR = _getenv_path(
    "MODEL_ARTEFACTS_DIR", str(MODEL_DIR / DEFAULT_MODEL_ARTEFACTS_SUBDIR)
)
DRIFT_ARTEFACTS_DIR = _getenv_path(
    "DRIFT_ARTEFACTS_DIR", str(MODEL_DIR / DEFAULT_DRIFT_ARTEFACTS_SUBDIR)
)

BEST_MODEL_PATH = MODEL_DIR / BEST_MODEL_FILENAME
BEST_MODEL_META_PATH = MODEL_ARTEFACTS_DIR / BEST_MODEL_META_FILENAME
BASELINE_NUMERIC_STATS_PATH = MODEL_ARTEFACTS_DIR / BASELINE_NUMERIC_STATS_FILENAME

# Датасет
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET)
CSV_NAME = os.environ.get("CSV_NAME", DEFAULT_CSV_NAME)

# Флаги
FORCE_DOWNLOAD = _getenv_bool("FORCE_DOWNLOAD", False)
FORCE_PROCESS = _getenv_bool("FORCE_PROCESS", False)
FORCE_TRAIN = _getenv_bool("FORCE_TRAIN", False)
KEEP_CANDIDATES = _getenv_bool("KEEP_CANDIDATES", False)

# Дрейф и валидация
INJECT_SYNTHETIC_DRIFT = _getenv_bool("INJECT_SYNTHETIC_DRIFT", False)
RUN_DRIFT_MONITOR = _getenv_bool("RUN_DRIFT_MONITOR", False)
RUN_DATA_VALIDATION = _getenv_bool("RUN_DATA_VALIDATION", True)

# Порог минимального размера выборок для стабильной оценки PSI метрики
MIN_SAMPLES_FOR_PSI = _getenv_int("MIN_SAMPLES_FOR_PSI", 10)

# Лимит сэмплирования на класс для балансировки датасета (компромисс между
# скоростью обработки и качеством модели: ~35k/класс = ~175k total для 5 классов)
PER_CLASS_LIMIT = 10000 #_getenv_int("PER_CLASS_LIMIT", 35000)

# Размер словаря TF-IDF — кратный 1024 для выравнивания в памяти
HASHING_TF_FEATURES = _getenv_int("HASHING_TF_FEATURES", 6144)
SHUFFLE_PARTITIONS = _getenv_int("SHUFFLE_PARTITIONS", 32)
MIN_DF = _getenv_int("MIN_DF", 3)
MIN_TF = _getenv_int("MIN_TF", 2)

# TF-IDF параметры
TFIDF_MAX_FEATURES_MIN = _getenv_int("TFIDF_MAX_FEATURES_MIN", 2000)
TFIDF_MAX_FEATURES_MAX = _getenv_int("TFIDF_MAX_FEATURES_MAX", 6000)
TFIDF_MAX_FEATURES_STEP = _getenv_int("TFIDF_MAX_FEATURES_STEP", 500)

# Обучение
OPTUNA_N_TRIALS = _getenv_int(
    "OPTUNA_N_TRIALS", 30
)  # Компромисс между качеством оптимизации и временем
OPTUNA_STORAGE = os.environ.get(
    "OPTUNA_STORAGE", "postgresql+psycopg2://admin:admin@postgres:5432/optuna"
)
STUDY_BASE_NAME = os.environ.get("STUDY_BASE_NAME", "kindle_optuna")
N_FOLDS = _getenv_int("N_FOLDS", 1)  # 1 = holdout validation, >1 = cross-validation
TRAIN_DEVICE = os.environ.get("TRAIN_DEVICE", "cpu")
EARLY_STOP_PATIENCE = _getenv_int("EARLY_STOP_PATIENCE", 8)
OPTUNA_TIMEOUT_SEC = _getenv_int("OPTUNA_TIMEOUT_SEC", 2400)
MIN_TRIALS_BEFORE_EARLY_STOP = _getenv_int("MIN_TRIALS_BEFORE_EARLY_STOP", 15)
DISTILBERT_TIMEOUT_SEC = _getenv_int("DISTILBERT_TIMEOUT_SEC", 1800)
OPTUNA_TOPK_EXPORT = _getenv_int("OPTUNA_TOPK_EXPORT", 20)

# DistilBERT гиперпараметры
DISTILBERT_MIN_EPOCHS = _getenv_int("DISTILBERT_MIN_EPOCHS", 2)
DISTILBERT_MAX_EPOCHS = _getenv_int("DISTILBERT_MAX_EPOCHS", 8)

# Модели для обучения
SELECTED_MODEL_KINDS = [
    ModelKind.logreg,
    ModelKind.rf,
    ModelKind.hist_gb,
    ModelKind.mlp,
    # ModelKind.distilbert,
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


@dataclass
class DataPaths:
    train: Path = PROCESSED_DATA_DIR / "train.parquet"
    val: Path = PROCESSED_DATA_DIR / "val.parquet"
    test: Path = PROCESSED_DATA_DIR / "test.parquet"


DATA_PATHS = DataPaths()
