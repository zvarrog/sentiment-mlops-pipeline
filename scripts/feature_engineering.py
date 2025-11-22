"""Единый модуль генерации фичей для API и Spark.

Минимальные соглашения:
- Все расчёты делаются по уже очищенному тексту.
- Отсутствующие числовые признаки заполняются нулями (без data leakage).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    import re

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_sentiment(text: str) -> float:
    if not text or len(text.strip()) < 3:
        return 0.0
    try:
        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        return float(max(-1.0, min(1.0, round(polarity, 4))))
    except Exception:
        return 0.0


def extract_text_features(text: str) -> dict[str, float]:
    if not text or not isinstance(text, str):
        return {
            "text_len": 0.0,
            "word_count": 0.0,
            "kindle_freq": 0.0,
            "exclamation_count": 0.0,
            "caps_ratio": 0.0,
            "question_count": 0.0,
        }
    text_len = float(len(text))
    words = text.split()
    word_count = float(len(words))
    kindle_freq = float(text.lower().count("kindle"))
    exclamation_count = float(text.count("!"))
    question_count = float(text.count("?"))
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(text_len, 1.0)
    return {
        "text_len": text_len,
        "word_count": word_count,
        "kindle_freq": kindle_freq,
        "exclamation_count": exclamation_count,
        "caps_ratio": caps_ratio,
        "question_count": question_count,
    }


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
    s = clean_series.fillna("")

    text_len = s.str.len().astype(float)
    word_count = s.str.split().str.len().fillna(0).astype(float)
    import re

    kindle_freq = s.str.count("kindle", flags=re.IGNORECASE).astype(float)
    exclamation_count = s.str.count("!").astype(float)
    caps_ratio = (
        s.str.replace(r"[^A-Z]", "", regex=True).str.len().astype(float)
        / text_len.clip(lower=1.0)
    ).fillna(0.0)
    question_count = s.str.count(r"\?").astype(float)
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
    df = pd.DataFrame({"reviewText": list(texts)})
    raw_texts = df["reviewText"].fillna("")
    txt_df = extract_basic_text_features(cast(pd.Series, raw_texts))

    df["reviewText"] = raw_texts.apply(clean_text)

    df = pd.concat([df, txt_df], axis=1)
    df["sentiment"] = calculate_sentiment_series(cast(pd.Series, df["reviewText"]))

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

    for col in expected_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df, ignored
