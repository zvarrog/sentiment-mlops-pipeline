"""
Модуль обработки признаков для обучения моделей.
"""

import gc

from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from scripts.config import MEMORY_WARNING_MB, log

NUMERIC_COLS = [
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

MEM_WARN_MB = MEMORY_WARNING_MB


class DenseTransformer(TransformerMixin, BaseEstimator):
    """
    Превращает scipy sparse в dense numpy.
    """

    def fit(self, _X, _y=None):
        """Фиктивный fit для совместимости со sklearn API.

        Args:
            _X: Входные данные (не используются)
            _y: Целевая переменная (не используется)

        Returns:
            self
        """
        # fit(X, y=None) — обязательная сигнатура по API sklearn даже для stateless-трансформеров;
        return self

    def transform(self, X):
        """Преобразует разреженную матрицу в плотную с контролем памяти.

        Args:
            X: Входная матрица (sparse или dense)

        Returns:
            np.ndarray: Плотный numpy массив с предупреждением при превышении лимита памяти
        """
        arr = X.toarray() if sp.issparse(X) else X
        size_mb = arr.nbytes / (1024 * 1024)
        if size_mb > MEM_WARN_MB:
            log.warning(
                "DenseTransformer: размер dense=%.1f MB > %.0f MB (риск памяти)",
                size_mb,
                MEM_WARN_MB,
            )
            # Принудительная очистка памяти для критически больших матриц
            if size_mb > MEM_WARN_MB * 1.5:
                gc.collect()
                log.info("Выполнена принудительная очистка памяти (garbage collection)")
        return arr
