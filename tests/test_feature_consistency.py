import pandas as pd
import pytest

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
    df, ignored = transform_features(texts, numeric_features=None, expected_numeric_cols=[])

    # Проверяем, что в df есть и очищенный текст, и правильные фичи
    assert df.iloc[0]["reviewText"] == "hello world"
    assert df.iloc[0]["exclamation_count"] == 3.0
    assert df.iloc[0]["caps_ratio"] > 0.0

    assert df.iloc[1]["reviewText"] == "simple text"
    assert df.iloc[1]["exclamation_count"] == 0.0
    assert df.iloc[1]["caps_ratio"] == 0.0


# ============================================================
# Edge Cases
# ============================================================


class TestEdgeCases:
    """Граничные случаи для feature engineering."""

    def test_empty_string(self):
        """Пустая строка не должна ломать пайплайн."""
        result = clean_text("")
        assert result == ""

        features = extract_basic_text_features(pd.Series([""]))
        assert features.iloc[0]["text_len"] == 0.0
        assert features.iloc[0]["word_count"] == 0.0

    def test_none_value(self):
        """None должен обрабатываться как пустая строка."""
        result = clean_text(None)
        assert result == ""

    def test_whitespace_only(self):
        """Строка только из пробелов."""
        result = clean_text("   \t\n  ")
        assert result == ""

        features = extract_basic_text_features(pd.Series(["   "]))
        assert features.iloc[0]["word_count"] == 0.0

    def test_unicode_emoji(self):
        """Unicode и emoji не должны ломать обработку."""
        text = "Great product! 🔥🔥 Отличный товар! Ценa: €100"
        result = clean_text(text)
        # Должен содержать только латиницу после очистки
        assert "great" in result
        assert "product" in result
        # Кириллица и спецсимволы удаляются
        assert "🔥" not in result
        assert "€" not in result

    def test_very_long_text(self):
        """Очень длинный текст (10KB+)."""
        long_text = "word " * 5000  # ~25KB
        result = clean_text(long_text)
        assert len(result) > 0

        features = extract_basic_text_features(pd.Series([long_text]))
        assert features.iloc[0]["word_count"] == 5000.0

    def test_special_characters_only(self):
        """Текст только из спецсимволов."""
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = clean_text(text)
        assert result == ""

        features = extract_basic_text_features(pd.Series([text]))
        assert features.iloc[0]["exclamation_count"] == 1.0
        assert features.iloc[0]["question_count"] == 1.0

    def test_mixed_case_consistency(self):
        """Регистр должен влиять на caps_ratio, но не на очищенный текст."""
        lower = "hello world"
        upper = "HELLO WORLD"
        mixed = "HeLLo WoRLd"

        assert clean_text(lower) == clean_text(upper) == clean_text(mixed)

        features_lower = extract_basic_text_features(pd.Series([lower]))
        features_upper = extract_basic_text_features(pd.Series([upper]))

        assert features_lower.iloc[0]["caps_ratio"] == 0.0
        assert features_upper.iloc[0]["caps_ratio"] > 0.5

    def test_numbers_and_urls(self):
        """Числа и URL должны удаляться."""
        text = "Check http://example.com and call 123-456-7890"
        result = clean_text(text)
        assert "http" not in result
        assert "123" not in result
        assert "check" in result
        assert "call" in result

    @pytest.mark.parametrize(
        "text,expected_exclaim,expected_question",
        [
            ("Hello!", 1.0, 0.0),
            ("What?", 0.0, 1.0),
            ("Really?!", 1.0, 1.0),
            ("Wow!!! Amazing!!!", 6.0, 0.0),
            ("???", 0.0, 3.0),
        ],
    )
    def test_punctuation_counting(self, text, expected_exclaim, expected_question):
        """Подсчёт пунктуации должен быть точным."""
        features = extract_basic_text_features(pd.Series([text]))
        assert features.iloc[0]["exclamation_count"] == expected_exclaim
        assert features.iloc[0]["question_count"] == expected_question
