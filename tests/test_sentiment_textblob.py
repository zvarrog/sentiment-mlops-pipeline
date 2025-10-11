"""Тест sentiment анализа с TextBlob в Spark окружении."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, StringType, StructField, StructType


@pytest.fixture(scope="module")
def spark():
    """Создаём Spark сессию для тестов."""
    spark = (
        SparkSession.builder.appName("TestSentiment")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_sentiment_textblob_basic(spark):
    """Тестируем базовую функциональность sentiment анализа."""
    from pyspark.sql.functions import udf

    def calculate_sentiment_textblob(text):
        """Копия функции из spark_process.py для тестирования."""
        if not text or len(text.strip()) < 3:
            return 0.0

        try:
            from textblob import TextBlob

            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return float(max(-1.0, min(1.0, round(polarity, 4))))

        except ImportError:
            # Fallback для случая отсутствия TextBlob
            positive_indicators = ["good", "great", "excellent", "love", "amazing"]
            negative_indicators = ["bad", "terrible", "awful", "worst", "hate"]

            pos_count = sum(1 for word in positive_indicators if word in text)
            neg_count = sum(1 for word in negative_indicators if word in text)

            if pos_count > neg_count:
                return min(0.8, pos_count * 0.2)
            elif neg_count > pos_count:
                return max(-0.8, -neg_count * 0.2)
            else:
                return 0.0
        except Exception:
            return 0.0

    # Создаём UDF
    sentiment_udf = udf(calculate_sentiment_textblob, FloatType())

    # Тестовые данные
    test_data = [
        ("This book is absolutely amazing and wonderful!", "positive"),
        ("I hate this terrible and awful story", "negative"),
        ("The book was okay, nothing special", "neutral"),
        ("", "empty"),
        ("a", "short"),
    ]

    schema = StructType(
        [
            StructField("text", StringType(), True),
            StructField("expected_type", StringType(), True),
        ]
    )

    df = spark.createDataFrame(test_data, schema)

    # Применяем sentiment анализ
    result_df = df.withColumn("sentiment", sentiment_udf(col("text")))

    # Собираем результаты
    results = result_df.collect()

    # Проверяем результаты
    assert len(results) == 5

    # Позитивный текст должен иметь положительный sentiment
    positive_result = [r for r in results if r.expected_type == "positive"][0]
    assert positive_result.sentiment > 0.1

    # Негативный текст должен иметь отрицательный sentiment
    negative_result = [r for r in results if r.expected_type == "negative"][0]
    assert negative_result.sentiment < -0.1

    # Пустой текст должен быть нейтральным
    empty_result = [r for r in results if r.expected_type == "empty"][0]
    assert empty_result.sentiment == 0.0

    # Короткий текст должен быть нейтральным
    short_result = [r for r in results if r.expected_type == "short"][0]
    assert short_result.sentiment == 0.0


def test_sentiment_categories(spark):
    """Тестируем категоризацию sentiment."""
    from pyspark.sql.functions import udf, when

    def calculate_sentiment_textblob(text):
        """Simplified version for testing."""
        if "amazing" in text.lower():
            return 0.8
        elif "terrible" in text.lower():
            return -0.8
        elif "okay" in text.lower():
            return 0.05
        else:
            return 0.0

    sentiment_udf = udf(calculate_sentiment_textblob, FloatType())

    test_data = [
        ("This is amazing",),
        ("This is terrible",),
        ("This is okay",),
        ("This is neutral",),
    ]

    schema = StructType([StructField("text", StringType(), True)])
    df = spark.createDataFrame(test_data, schema)

    # Применяем sentiment и категории
    result_df = (
        df.withColumn("sentiment", sentiment_udf(col("text")))
        .withColumn(
            "sentiment_category",
            when(col("sentiment") > 0.1, "positive")
            .when(col("sentiment") < -0.1, "negative")
            .otherwise("neutral"),
        )
        .withColumn(
            "sentiment_strength",
            when(col("sentiment").abs() > 0.5, "strong")
            .when(col("sentiment").abs() > 0.2, "moderate")
            .otherwise("weak"),
        )
    )

    results = result_df.collect()

    # Проверяем категории
    amazing_row = [r for r in results if "amazing" in r.text][0]
    assert amazing_row.sentiment_category == "positive"
    assert amazing_row.sentiment_strength == "strong"

    terrible_row = [r for r in results if "terrible" in r.text][0]
    assert terrible_row.sentiment_category == "negative"
    assert terrible_row.sentiment_strength == "strong"

    okay_row = [r for r in results if "okay" in r.text][0]
    assert okay_row.sentiment_category == "neutral"
    assert okay_row.sentiment_strength == "weak"


def test_sentiment_edge_cases():
    """Тестируем граничные случаи sentiment анализа."""
    from textblob import TextBlob

    def calculate_sentiment_textblob(text):
        """Основная функция sentiment анализа (как в продакшене)."""
        if not text or len(str(text).strip()) < 3:
            return 0.0
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        return float(max(-1.0, min(1.0, round(polarity, 4))))

    # Граничные случаи
    assert calculate_sentiment_textblob("") == 0.0
    assert calculate_sentiment_textblob("  ") == 0.0
    assert calculate_sentiment_textblob("ab") == 0.0  # слишком короткий
    assert isinstance(calculate_sentiment_textblob("This is great!"), float)
    assert -1.0 <= calculate_sentiment_textblob("Absolutely terrible and awful") <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
