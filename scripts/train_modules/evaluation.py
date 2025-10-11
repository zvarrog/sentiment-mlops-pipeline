"""
Модуль метрик, confusion matrix и отчётов для оценки моделей.
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

log = logging.getLogger("evaluation")


def compute_metrics(y_true, y_pred):
    """
    Вычисляет метрики качества модели.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def log_confusion_matrix(y_true, y_pred, path):
    """
    Сохраняет confusion matrix в файл.
    """
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(path, cm, fmt="%d")
    log.info(f"Confusion matrix сохранена: {path}")


def get_classification_report(y_true, y_pred):
    """
    Возвращает текстовый отчёт по классификации.
    """
    return classification_report(y_true, y_pred)
