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
from .feature_space import build_feature_space
from .models import build_model
from .optuna_optimizer import optimize_model
from .pipeline_builders import ModelBuilderFactory

__all__ = [
    "load_splits",
    "compute_metrics",
    "log_confusion_matrix",
    "build_feature_space",
    "build_model",
    "ModelBuilderFactory",
    "optimize_model",
    "save_model_artifact",
    "save_json_artifact",
    "save_csv_artifact",
    "save_text_artifact",
    "log_mlflow_artifact_safe",
    "calculate_baseline_statistics",
]
