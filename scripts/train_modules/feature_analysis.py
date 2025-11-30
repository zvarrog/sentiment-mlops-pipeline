"""Анализ важности признаков обученных моделей.

Единый источник истины для извлечения feature importances.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from scripts.logging_config import get_logger

log = get_logger(__name__)


def get_feature_names(preprocessor: ColumnTransformer, use_svd: bool) -> list[str]:
    """Извлекает имена признаков из препроцессора.

    Args:
        preprocessor: ColumnTransformer с text и numeric трансформерами.
        use_svd: Флаг использования SVD для текстовых признаков.

    Returns:
        Список имён признаков в порядке, соответствующем выходу препроцессора.
    """
    feature_names: list[str] = []

    try:
        text_pipe: Pipeline = preprocessor.named_transformers_["text"]
        tfidf: TfidfVectorizer = text_pipe.named_steps["tfidf"]
        vocab_inv = (
            {idx: tok for tok, idx in tfidf.vocabulary_.items()}
            if hasattr(tfidf, "vocabulary_")
            else {}
        )
        text_dim = len(vocab_inv) if vocab_inv else 0
        numeric_cols = preprocessor.transformers_[1][2]

        if not use_svd and vocab_inv:
            feature_names.extend([vocab_inv.get(i, f"tok_{i}") for i in range(text_dim)])
        elif use_svd and "svd" in text_pipe.named_steps:
            svd_model = text_pipe.named_steps["svd"]
            n_components = svd_model.n_components

            for comp_idx in range(n_components):
                component = svd_model.components_[comp_idx]
                top_indices = np.argsort(np.abs(component))[-10:][::-1]
                top_terms = [vocab_inv.get(i, f"tok_{i}") for i in top_indices]
                feature_name = f"svd_{comp_idx}[{','.join(top_terms[:3])}...]"
                feature_names.append(feature_name)

        feature_names.extend(list(numeric_cols))
    except (KeyError, AttributeError, IndexError) as e:
        log.warning("Ошибка при извлечении имен признаков: %s", e)

    return feature_names


def get_model_coefficients(model) -> np.ndarray | None:
    """Извлекает coef_ или feature_importances_ из модели."""
    if hasattr(model, "coef_"):
        coef_result: np.ndarray = np.mean(np.abs(model.coef_), axis=0)
        return coef_result
    if hasattr(model, "feature_importances_"):
        importances: np.ndarray = model.feature_importances_
        return importances
    return None


def extract_feature_importances(pipeline: Pipeline, use_svd: bool) -> list[dict[str, float]]:
    """Извлекает топ-50 наиболее важных признаков из обученной модели.

    Args:
        pipeline: Обученный sklearn Pipeline с 'pre' и 'model' шагами.
        use_svd: Флаг использования SVD для текстовых признаков.

    Returns:
        Список словарей с ключами 'feature' и 'importance'.
    """
    res: list[dict[str, float]] = []
    try:
        if "pre" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            return res

        model = pipeline.named_steps["model"]
        pre: ColumnTransformer = pipeline.named_steps["pre"]

        feature_names = get_feature_names(pre, use_svd)
        coefs = get_model_coefficients(model)

        if coefs is None:
            return res

        top_idx = np.argsort(coefs)[::-1][:50]
        for i in top_idx:
            if i < len(feature_names):
                res.append({"feature": feature_names[i], "importance": float(coefs[i])})
    except (KeyError, AttributeError, ValueError) as e:
        log.warning("Не удалось извлечь feature importances: %s", e)
    return res
