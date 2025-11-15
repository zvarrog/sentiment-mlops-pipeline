"""Общее пространство признаков для обучения и тестов.

Содержит:
- NUMERIC_COLS — список числовых фичей, используемых моделью
- DenseTransformer — sklearn совместимый трансформер, приводящий sparse к dense
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from scripts.config import NUMERIC_COLS as _NUMERIC_COLS

# Единый источник истины — берём список числовых фичей из конфигурации
NUMERIC_COLS: list[str] = list(_NUMERIC_COLS)


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Преобразует sparse матрицы в dense numpy.ndarray.

    Используется в пайплайнах перед моделями, которые не поддерживают sparse.
    """

    def fit(self, x: Any, y: Any | None = None) -> DenseTransformer:
        return self

    def transform(self, x: Any) -> np.ndarray:
        if x is None:
            return np.empty((0, 0), dtype=float)
        if sp.issparse(x):
            return x.toarray()
        # Если уже плотная матрица/массив — возвращаем как есть
        return np.asarray(x)
