DEFAULT_RAW_DATA_DIR = "data/raw"
DEFAULT_PROCESSED_DATA_DIR = "data/processed"
DEFAULT_MODEL_DIR = "artefacts"
DEFAULT_MODEL_ARTEFACTS_SUBDIR = "model_artefacts"
DEFAULT_DRIFT_ARTEFACTS_SUBDIR = "drift_artefacts"

DEFAULT_KAGGLE_DATASET = "bharadwaj6/kindle-reviews"
DEFAULT_CSV_NAME = "kindle_reviews.csv"
DEFAULT_JSON_NAME = "kindle_reviews.json"

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
