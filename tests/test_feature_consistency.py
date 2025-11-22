import pandas as pd

from scripts.feature_engineering import (
    clean_text,
    extract_basic_text_features,
    transform_features,
)


def test_clean_text_logic():
    """Проверяем, что очистка работает как ожидается (lower + remove punctuation)."""
    raw = "HELLO World!!! http://link.com"
    cleaned = clean_text(raw)
    assert cleaned == "hello world"


def test_features_on_raw_text():
    """Проверяем, что признаки считаются по СЫРОМУ тексту (до очистки)."""
    raw = "HELLO World!!!"
    # Если бы считали по чистому ("hello world"), то caps_ratio=0, exclamation=0

    df_raw = pd.Series([raw])
    features = extract_basic_text_features(df_raw)

    # Проверки
    assert features.iloc[0]["exclamation_count"] == 3.0
    assert features.iloc[0]["caps_ratio"] > 0.0
    assert features.iloc[0]["text_len"] == len(raw)


def test_transform_features_integration():
    """Проверяем интеграционную функцию transform_features."""
    texts = ["HELLO World!!!", "simple text"]
    df, ignored = transform_features(
        texts, numeric_features=None, expected_numeric_cols=[]
    )

    # Проверяем, что в df есть и очищенный текст, и правильные фичи
    assert df.iloc[0]["reviewText"] == "hello world"
    assert df.iloc[0]["exclamation_count"] == 3.0
    assert df.iloc[0]["caps_ratio"] > 0.0

    assert df.iloc[1]["reviewText"] == "simple text"
    assert df.iloc[1]["exclamation_count"] == 0.0
    assert df.iloc[1]["caps_ratio"] == 0.0
