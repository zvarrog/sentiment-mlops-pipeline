"""Модуль для построения и сохранения визуализаций.

Выносит дублированную логику построения ROC/PR кривых из train.py и evaluation_reporter.py.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from scripts.logging_config import get_logger
from scripts.utils import get_model_input

log = get_logger(__name__)


def plot_roc_pr_curves(
    pipeline: Any,
    x_test: Any,
    y_test: Any,
    output_dir: Path,
    is_distilbert: bool = False,
) -> tuple[Path | None, Path | None]:
    """Строит и сохраняет ROC и PR кривые.

    Возвращает пути к сохранённым файлам графиков или (None, None) при ошибке.
    """
    try:
        if not hasattr(pipeline, "predict_proba"):
            log.debug("Модель не поддерживает predict_proba, пропуск ROC/PR кривых")
            return None, None

        # Определяем входные данные для predict_proba
        x_for_proba = get_model_input(pipeline, x_test)

        y_score = pipeline.predict_proba(x_for_proba)
        classes = sorted(set(y_test.tolist()))
        y_true_bin = label_binarize(y_test, classes=classes)

        # Выравнивание размерности если нужно
        if y_score.shape[1] != y_true_bin.shape[1]:
            y_score = _align_probability_shape(pipeline, y_score, y_true_bin, classes)

        # ROC кривая
        roc_path = _plot_roc_curve(y_true_bin, y_score, output_dir)

        # PR кривая
        pr_path = _plot_pr_curve(y_true_bin, y_score, output_dir)

        return roc_path, pr_path

    except (ValueError, OSError, RuntimeError, AttributeError) as e:
        log.warning("Не удалось построить ROC/PR кривые: %s", e)
        return None, None


def _align_probability_shape(
    pipeline: Any, y_score: np.ndarray, y_true_bin: np.ndarray, classes: list
) -> np.ndarray:
    """Выравнивает размерность вероятностей с истинными метками."""
    proba_aligned = np.zeros((y_score.shape[0], y_true_bin.shape[1]), dtype=float)

    try:
        model_classes = list(getattr(pipeline, "classes_", []))
    except AttributeError:
        model_classes = []

    for j, c in enumerate(classes):
        if model_classes and c in model_classes:
            src_idx = model_classes.index(c)
            if src_idx < y_score.shape[1]:
                proba_aligned[:, j] = y_score[:, src_idx]

    return proba_aligned


def _plot_roc_curve(y_true_bin: np.ndarray, y_score: np.ndarray, output_dir: Path) -> Path:
    """Строит и сохраняет ROC кривую."""
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"micro-avg ROC (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (micro-average)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    roc_path = output_dir / "roc_curve_test.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=100)
    plt.close(fig)

    log.debug("ROC кривая сохранена: %s", roc_path)
    return roc_path


def _plot_pr_curve(y_true_bin: np.ndarray, y_score: np.ndarray, output_dir: Path) -> Path:
    """Строит и сохраняет Precision-Recall кривую."""
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_true_bin, y_score, average="micro")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"micro-avg PR (AP={ap_micro:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (micro-average)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    pr_path = output_dir / "pr_curve_test.png"
    fig.tight_layout()
    fig.savefig(pr_path, dpi=100)
    plt.close(fig)

    log.debug("PR кривая сохранена: %s", pr_path)
    return pr_path
