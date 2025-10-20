import os
from pathlib import Path
from types import SimpleNamespace

from scripts.models.kinds import ModelKind

from .logging_config import get_logger

# Опциональная загрузка .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# === Функция для булевых переменных ===
def _getenv_bool(key: str, default: str = "false") -> bool:
    """Конвертирует переменную окружения в bool."""
    return os.getenv(key, default).lower() in {"1", "true", "yes"}


# === Директории ===
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
# Корень для всех артефактов (модель и вспомогательные файлы)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artefacts"))
MODEL_FILE_DIR = MODEL_DIR
MODEL_ARTEFACTS_DIR = Path(
    os.getenv("MODEL_ARTEFACTS_DIR", str(MODEL_DIR / "model_artefacts"))
)
DRIFT_ARTEFACTS_DIR = Path(
    os.getenv("DRIFT_ARTEFACTS_DIR", str(MODEL_DIR / "drift_artefacts"))
)

# === Датасет ===
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "bharadwaj6/kindle-reviews")
CSV_NAME = os.getenv("CSV_NAME", "kindle_reviews.csv")
JSON_NAME = os.getenv("JSON_NAME", "kindle_reviews.json")

# === Флаги ===
FORCE_DOWNLOAD = _getenv_bool("FORCE_DOWNLOAD")
FORCE_PROCESS = _getenv_bool("FORCE_PROCESS")
FORCE_TRAIN = _getenv_bool("FORCE_TRAIN")

# === Флаги дрейфа ===
INJECT_SYNTHETIC_DRIFT = _getenv_bool("INJECT_SYNTHETIC_DRIFT")
RUN_DRIFT_MONITOR = _getenv_bool("RUN_DRIFT_MONITOR")

# === Обработка данных ===
PER_CLASS_LIMIT = int(os.getenv("PER_CLASS_LIMIT", "35000"))
HASHING_TF_FEATURES = int(os.getenv("HASHING_TF_FEATURES", "6144"))
SHUFFLE_PARTITIONS = int(os.getenv("SHUFFLE_PARTITIONS", "32"))
MIN_DF = int(os.getenv("MIN_DF", "3"))
MIN_TF = int(os.getenv("MIN_TF", "2"))
SEED = int(os.getenv("SEED", "42"))

# === Модели и обучение ===
SELECTED_MODEL_KINDS = [
    ModelKind.logreg,
    ModelKind.rf,
    ModelKind.hist_gb,
    ModelKind.mlp,
    ModelKind.distilbert,
]
OPTUNA_N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "30"))
OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///optuna_study.db")
STUDY_BASE_NAME = os.getenv("STUDY_BASE_NAME", "kindle_optuna")

# === Connection Pooling для PostgreSQL ===
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# === Дополнительные параметры ===
N_FOLDS = int(os.getenv("N_FOLDS", "1"))
TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "cpu")  # "cpu" или "cuda"
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "8"))
OPTUNA_TIMEOUT_SEC = 2400
MIN_TRIALS_BEFORE_EARLY_STOP = 15
DISTILBERT_TIMEOUT_SEC = int(os.getenv("DISTILBERT_TIMEOUT_SEC", "1200"))

# === Настройки памяти ===
MEMORY_WARNING_MB = int(
    os.getenv("MEMORY_WARNING_MB", "2048")
)  # лимит предупреждения о памяти

# Адаптивный TF-IDF max_features на основе размера датасета
# Для малых датасетов (<100K) используем меньше фичей, для больших (>500K) - больше
def get_tfidf_max_features_range(dataset_size: int) -> tuple[int, int, int]:
    """Вычисляет диапазон max_features для TF-IDF на основе размера датасета."""
    if dataset_size < 100_000:
        return (1000, 3000, 500)  # min, max, step
    elif dataset_size < 500_000:
        return (2000, 6000, 500)
    else:
        return (3000, 10000, 1000)


TFIDF_MAX_FEATURES_MIN = int(os.getenv("TFIDF_MAX_FEATURES_MIN", "2000"))
TFIDF_MAX_FEATURES_MAX = int(os.getenv("TFIDF_MAX_FEATURES_MAX", "6000"))
TFIDF_MAX_FEATURES_STEP = int(os.getenv("TFIDF_MAX_FEATURES_STEP", "500"))

# FORCE_SVD_THRESHOLD_MB — порог, после которого включается SVD для снижения размерности. Чем выше порог, тем выше качество, но медленнее обучение.
FORCE_SVD_THRESHOLD_MB = int(os.getenv("FORCE_SVD_THRESHOLD_MB", "4000"))

# === Spark ресурсы ===
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g")
SPARK_NUM_CORES = int(os.getenv("SPARK_NUM_CORES", "4"))

log = get_logger("kindle")

# === Объект settings для обратной совместимости ===
settings = SimpleNamespace(
    force_download=FORCE_DOWNLOAD,
    force_process=FORCE_PROCESS,
    force_train=FORCE_TRAIN,
    kaggle_dataset=KAGGLE_DATASET,
    csv_name=CSV_NAME,
    json_name=JSON_NAME,
    model_dir=str(MODEL_DIR),
    raw_data_dir=str(RAW_DATA_DIR),
    processed_data_dir=str(PROCESSED_DATA_DIR),
    per_class_limit=PER_CLASS_LIMIT,
    hashing_tf_features=HASHING_TF_FEATURES,
    shuffle_partitions=SHUFFLE_PARTITIONS,
    min_df=MIN_DF,
    min_tf=MIN_TF,
    selected_models=",".join(m.value for m in SELECTED_MODEL_KINDS),
    optuna_n_trials=OPTUNA_N_TRIALS,
    optuna_storage=OPTUNA_STORAGE,
    optuna_timeout_sec=OPTUNA_TIMEOUT_SEC,
    early_stop_patience=EARLY_STOP_PATIENCE,
    min_trials_before_early_stop=MIN_TRIALS_BEFORE_EARLY_STOP,
    distilbert_timeout_sec=DISTILBERT_TIMEOUT_SEC,
    spark_driver_memory=SPARK_DRIVER_MEMORY,
    spark_executor_memory=SPARK_EXECUTOR_MEMORY,
)
