"""Единый модуль генерации фичей для API и Spark.

Минимальные соглашения:
- Все расчёты делаются по уже очищенному тексту.
- Отсутствующие числовые признаки заполняются нулями (без data leakage).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import cast

import pandas as pd
from textblob import TextBlob

from scripts.logging_config import get_logger

log = get_logger(__name__)


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_sentiment(text: str) -> float:
    """Рассчитывает показатель тональности (sentiment polarity) через TextBlob.

    Возвращает:
        float: Значение в диапазоне [-1.0, 1.0]; при ошибке возвращается 0.0.
    """
    if not text or len(text.strip()) < 3:
        return 0.0
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        return float(max(-1.0, min(1.0, round(polarity, 4))))
    except (AttributeError, ValueError, TypeError) as e:
        log.debug("Ошибка расчёта sentiment: %s", e)
        return 0.0


def _compute_single_text_features(text: str) -> dict[str, float]:
    """Вычисляет признаки для одного текста. Единственный источник истины."""
    if not text or not isinstance(text, str):
        return {
            "text_len": 0.0,
            "word_count": 0.0,
            "kindle_freq": 0.0,
            "exclamation_count": 0.0,
            "caps_ratio": 0.0,
            "question_count": 0.0,
            "avg_word_length": 0.0,
        }

    text_len = float(len(text))
    words = text.split()
    word_count = float(len(words))
    kindle_freq = float(text.lower().count("kindle"))
    exclamation_count = float(text.count("!"))
    question_count = float(text.count("?"))
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(text_len, 1.0)
    avg_word_length = text_len / max(word_count, 1.0)

    return {
        "text_len": text_len,
        "word_count": word_count,
        "kindle_freq": kindle_freq,
        "exclamation_count": exclamation_count,
        "caps_ratio": caps_ratio,
        "question_count": question_count,
        "avg_word_length": avg_word_length,
    }


def extract_text_features(text: str) -> dict[str, float]:
    """Извлекает признаки из одного текста (для Spark UDF/единичных запросов)."""
    return _compute_single_text_features(text)


def get_spark_clean_udf():
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    @udf(returnType=StringType())
    def clean_udf(text):
        return clean_text(text) if text else ""

    return clean_udf


def get_spark_feature_extraction_udf():
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType, MapType, StringType

    @udf(returnType=MapType(StringType(), FloatType()))
    def feature_udf(text):
        return extract_text_features(text) if text else {}

    return feature_udf


def calculate_sentiment_series(s: pd.Series) -> pd.Series:
    return s.fillna("").apply(calculate_sentiment).astype(float)


def extract_basic_text_features(clean_series: pd.Series) -> pd.DataFrame:
    """Векторизованное извлечение признаков для pandas (для обучения/батч-предикта).

    Возвращаем высокопроизводительную реализацию на базе pandas-операций,
    эквивалентную логике _compute_single_text_features.
    """
    s = clean_series.fillna("")

    text_len = s.str.len().astype(float)
    word_count = s.str.split().str.len().fillna(0).astype(float)
    kindle_freq = s.str.count("kindle", flags=re.IGNORECASE).astype(float)
    exclamation_count = s.str.count("!").astype(float)
    question_count = s.str.count(r"\?").astype(float)

    # Подсчёт заглавных букв и доли заглавных
    caps_count = s.str.replace(r"[^A-Z]", "", regex=True).str.len().astype(float)
    caps_ratio = (caps_count / text_len.clip(lower=1.0)).fillna(0.0)

    avg_word_length = (text_len / word_count.clip(lower=1.0)).astype(float)

    return pd.DataFrame(
        {
            "text_len": text_len,
            "word_count": word_count,
            "kindle_freq": kindle_freq,
            "exclamation_count": exclamation_count,
            "caps_ratio": caps_ratio,
            "question_count": question_count,
            "avg_word_length": avg_word_length,
        }
    )


def transform_features(
    texts: Iterable[str],
    numeric_features: dict[str, list[float]] | None,
    expected_numeric_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Основная точка входа для подготовки фичей.

    Порядок обработки:
    1. Извлечение текстовых признаков (caps_ratio, exclamation_count) из сырого текста
    2. Очистка текста для TF-IDF
    3. Расчёт sentiment по очищенному тексту для консистентности с train pipeline
    """
    df = pd.DataFrame({"reviewText": list(texts)})

    raw_texts = df["reviewText"].fillna("")
    txt_df = extract_basic_text_features(cast(pd.Series, raw_texts))

    df["reviewText"] = raw_texts.apply(clean_text)
    df["sentiment"] = calculate_sentiment_series(cast(pd.Series, df["reviewText"]))

    df = pd.concat([df, txt_df], axis=1)

    ignored: list[str] = []
    if numeric_features:
        n = len(df)
        for col, values in numeric_features.items():
            if col not in expected_numeric_cols:
                ignored.append(f"{col} (неизвестный признак)")
                continue
            if not isinstance(values, list) or len(values) != n:
                ignored.append(
                    f"{col} (длина {len(values) if isinstance(values, list) else 'n/a'} != {n})"
                )
                continue
            try:
                df[col] = pd.Series(values, index=df.index).astype(float)
            except (ValueError, TypeError):
                ignored.append(f"{col} (нечисловые значения)")

    # Заполняем пропуски для ожидаемых колонок
    for col in expected_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df, ignored
