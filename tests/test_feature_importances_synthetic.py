"""
Быстрый синтетический тест извлечения важностей признаков из пайплайна.

Не требует наличия parquet-файлов в data/processed.
"""

import numpy as np
import pandas as pd

from scripts.train import (
    _extract_feature_importances,
    build_pipeline,
)
from scripts.train_modules.feature_space import NUMERIC_COLS


class DummyTrial:
    """Упрощённый trial с фиксированными гиперпараметрами.

    Компоненты подобраны так, чтобы быстро обучить логистическую регрессию
    на TF-IDF без SVD.
    """

    def __init__(self, numeric_cols):
        self.params = {
            "tfidf_max_features": 500,
            "use_svd": False,
            "model": "logreg",
            "logreg_C": 1.0,
            "logreg_solver": "liblinear",
            "logreg_penalty": "l2",
        }
        # Ограничиваемся реально присутствующими колонками
        self._attrs = {"numeric_cols": numeric_cols}

    def suggest_int(self, name, low, high, step=1):
        return self.params.get(name, low)

    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])

    def suggest_float(self, name, low, high, log=False):
        return self.params.get(name, low)

    @property
    def user_attrs(self):
        return self._attrs


def _make_synthetic_df(n=40):
    # Тексты из 5 классов, повторяем шаблоны
    base_texts = [
        "I loved this kindle book so much",
        "Terrible experience, waste of time",
        "Average quality, could be better",
        "Excellent read, highly recommend",
        "Not great, not terrible",
    ]
    texts = (base_texts * ((n // len(base_texts)) + 1))[:n]
    # Числовые признаки (заполним часть доступных колонок)
    num_cols = ["text_len", "word_count", "kindle_freq", "sentiment"]
    data = {
        "reviewText": texts,
    }
    for c in num_cols:
        if c == "text_len":
            data[c] = [len(t) for t in texts]
        elif c == "word_count":
            data[c] = [len(t.split()) for t in texts]
        elif c == "kindle_freq":
            data[c] = [t.lower().count("kindle") for t in texts]
        else:  # sentiment (фиктивная метрика)
            data[c] = np.random.randn(n).tolist()
    df = pd.DataFrame(data)
    # Метки 1..5
    y = np.array([(i % 5) + 1 for i in range(n)], dtype=int)
    present_num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    return df, y, present_num_cols


def test_feature_importances_synthetic_non_empty():
    X, y, present_cols = _make_synthetic_df(50)
    trial = DummyTrial(present_cols)
    pipe = build_pipeline(trial, "logreg")
    pipe.fit(X, y)
    fi = _extract_feature_importances(pipe, use_svd=False)
    assert fi, "Ожидаем непустой список важностей признаков"
