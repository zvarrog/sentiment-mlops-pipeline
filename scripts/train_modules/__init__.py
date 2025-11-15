"""Модули для обучения моделей."""

from .artifact_manager import (
    calculate_baseline_statistics,
    log_mlflow_artifact_safe,
    save_csv_artifact,
    save_json_artifact,
    save_model_artifact,
    save_text_artifact,
)
from .data_loading import load_splits
from .evaluation import compute_metrics, log_confusion_matrix
from .models import SimpleMLP
from .optuna_optimizer import optimize_model
from .pipeline_builders import ModelBuilderFactory

