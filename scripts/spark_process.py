"""Обработка kindle_reviews.csv c оптимизациями.

Изменения/особенности:
* Автоматическое удаление искусственного индексного столбца (leading comma / _c0)
* Балансировка по классам (звёздам) с настраиваемым лимитом по классу
* Минимизация числа shuffle (агрегации пользователя и товара в одном проходе каждая)
* Контролируемое число shuffle партиций для малых/средних объёмов
"""

from pathlib import Path

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml.feature import IDF, CountVectorizer, Tokenizer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    abs,
    avg,
    col,
    count,
    greatest,
    lit,
    pandas_udf,
    rand,
    row_number,
    when,
)
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window

from scripts.config import (
    CSV_NAME,
    DATA_PATHS,
    HASHING_TF_FEATURES,
    MIN_DF,
    MIN_TF,
    PARQUET_TRAIN_PARTITIONS,
    PARQUET_VAL_TEST_PARTITIONS,
    PER_CLASS_LIMIT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RUN_DATA_VALIDATION,
    SHUFFLE_PARTITIONS,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_NUM_CORES,
)
from scripts.feature_engineering import (
    calculate_sentiment,
    get_spark_clean_udf,
    get_spark_feature_extraction_udf,
)
from scripts.logging_config import get_logger

log = get_logger(__name__)


@pandas_udf(FloatType())
def sentiment_udf(texts: pd.Series) -> pd.Series:
    return texts.fillna("").apply(calculate_sentiment)


def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("KindleReviews")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.executor.cores", str(SPARK_NUM_CORES))
        .config("spark.driver.cores", str(SPARK_NUM_CORES))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    log.info(
        "Конфигурация: driverMemory=%s, executorMemory=%s, cores=%s, shuffle.partitions=%s",
        spark.sparkContext.getConf().get("spark.driver.memory"),
        spark.sparkContext.getConf().get("spark.executor.memory"),
        spark.sparkContext.getConf().get("spark.executor.cores", "n/a"),
        spark.conf.get("spark.sql.shuffle.partitions"),
    )
    return spark


def load_data(spark: SparkSession) -> DataFrame:
    return spark.read.csv(
        str(RAW_DATA_DIR / CSV_NAME),
        header=True,
        inferSchema=True,
        quote='"',
        escape='"',
        multiLine=True,
    )


def clean_and_balance_data(df: DataFrame) -> DataFrame:
    clean_udf = get_spark_clean_udf()
    feature_udf = get_spark_feature_extraction_udf()

    df = df.withColumn("features", feature_udf(col("reviewText")))
    df = df.withColumn("reviewText", clean_udf(col("reviewText")))

    feature_names = [
        "text_len",
        "word_count",
        "kindle_freq",
        "exclamation_count",
        "caps_ratio",
        "question_count",
    ]
    for feature_name in feature_names:
        df = df.withColumn(feature_name, col("features").getItem(feature_name))
    df = df.drop("features")

    # Чистим данные: валидные тексты и оценки, исключаем пустые/пробельные тексты
    clean = df.filter(
        (col("reviewText").isNotNull()) & (col("overall").isNotNull()) & (col("text_len") > 0)
    )
    log.info("Фильтрация null и пустых текстов применена")

    # Балансировка через Window-функцию
    window_spec = Window.partitionBy("overall").orderBy(rand(seed=42))
    clean = (
        clean.withColumn("rn", row_number().over(window_spec))
        .filter(col("rn") <= PER_CLASS_LIMIT)
        .drop("rn")
    )
    log.info("Балансировка применена: <= %d на класс", PER_CLASS_LIMIT)
    return clean


def add_sentiment_features(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("avg_word_length", col("text_len") / greatest(col("word_count"), lit(1)))
        .withColumn("sentiment", sentiment_udf(col("reviewText")))
        .withColumn(
            "sentiment_category",
            when(col("sentiment") > 0.1, "positive")
            .when(col("sentiment") < -0.1, "negative")
            .otherwise("neutral"),
        )
        .withColumn(
            "sentiment_strength",
            when(abs(col("sentiment")) > 0.5, "strong")
            .when(abs(col("sentiment")) > 0.2, "moderate")
            .otherwise("weak"),
        )
    )


def apply_tfidf(
    train: DataFrame, val: DataFrame, test: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:
    tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
    vectorizer = CountVectorizer(
        inputCol="words",
        outputCol="rawFeatures",
        vocabSize=HASHING_TF_FEATURES,
        minDF=MIN_DF,
        minTF=MIN_TF,
    )
    idf = IDF(inputCol="rawFeatures", outputCol="tfidfFeatures")

    # Fit on train
    train_words = tokenizer.transform(train)
    vec_model = vectorizer.fit(train_words)
    train_feat = vec_model.transform(train_words)
    idf_model = idf.fit(train_feat)
    train_res = idf_model.transform(train_feat)

    # Transform val
    val_words = tokenizer.transform(val)
    val_feat = vec_model.transform(val_words)
    val_res = idf_model.transform(val_feat)

    # Transform test
    test_words = tokenizer.transform(test)
    test_feat = vec_model.transform(test_words)
    test_res = idf_model.transform(test_feat)

    log.info(
        "CountVectorizer: vocabSize<=%d, minDF=%s, minTF=%s",
        HASHING_TF_FEATURES,
        str(MIN_DF),
        str(MIN_TF),
    )
    return train_res, val_res, test_res


def add_aggregations(
    train: DataFrame, val: DataFrame, test: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:
    # Агрегации на train — для избежания data leakage
    user_stats = (
        train.groupBy("reviewerID")
        .agg(
            avg("text_len").alias("user_avg_len"),
            count("reviewText").alias("user_review_count"),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    item_stats = (
        train.groupBy("asin")
        .agg(
            avg("text_len").alias("item_avg_len"),
            count("reviewText").alias("item_review_count"),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    train_res = train.join(user_stats, on="reviewerID", how="left").join(
        item_stats, on="asin", how="left"
    )
    val_res = val.join(user_stats, on="reviewerID", how="left").join(
        item_stats, on="asin", how="left"
    )
    test_res = test.join(user_stats, on="reviewerID", how="left").join(
        item_stats, on="asin", how="left"
    )

    log.info("После добавления агрегатов кол-во колонок в train: %d", len(train_res.columns))

    # Принудительная материализация статистик перед возвратом для гарантии вычисления.
    # При ленивых вычислениях это не строго обязательно до записи,
    # но persist помогает при многократном использовании (как здесь).
    return train_res, val_res, test_res


def validate_data() -> None:
    """Валидирует сохраненные parquet файлы."""
    if not RUN_DATA_VALIDATION:
        return

    try:
        from scripts.data_validation import (
            log_validation_results,
            validate_parquet_dataset,
        )

        validation_results = validate_parquet_dataset(Path(PROCESSED_DATA_DIR))
        all_valid = log_validation_results(validation_results)

        if not all_valid:
            log.warning("Обнаружены проблемы в сохранённых данных")
    except (OSError, ValueError) as e:
        log.warning("Ошибка валидации сохранённых данных: %s", e)


def process_data(force: bool = False) -> None:
    """Основная функция обработки данных Spark."""
    if (
        not force
        and DATA_PATHS.train.exists()
        and DATA_PATHS.val.exists()
        and DATA_PATHS.test.exists()
    ):
        log.warning(
            "Обработанные данные уже существуют в %s. Используйте force=True.",
            str(PROCESSED_DATA_DIR),
        )
        return

    spark = create_spark_session()
    clean = None
    train = val = test = None

    try:
        df = load_data(spark)
        clean = clean_and_balance_data(df)
        clean = add_sentiment_features(clean)

        clean = clean.persist(StorageLevel.MEMORY_AND_DISK)
        log.info("Данные с фичами закешированы")

        # Split
        train, val, test = clean.randomSplit([0.7, 0.15, 0.15], seed=42)

        # TF-IDF
        train, val, test = apply_tfidf(train, val, test)

        # Persist intermediate results
        train = train.persist(StorageLevel.MEMORY_AND_DISK)
        val = val.persist(StorageLevel.MEMORY_AND_DISK)
        test = test.persist(StorageLevel.MEMORY_AND_DISK)

        # Aggregations
        train, val, test = add_aggregations(train, val, test)

        # Save
        train.repartition(PARQUET_TRAIN_PARTITIONS).write.mode("overwrite").parquet(
            str(DATA_PATHS.train)
        )
        val.repartition(PARQUET_VAL_TEST_PARTITIONS).write.mode("overwrite").parquet(
            str(DATA_PATHS.val)
        )
        test.repartition(PARQUET_VAL_TEST_PARTITIONS).write.mode("overwrite").parquet(
            str(DATA_PATHS.test)
        )

        log.info("Данные сохранены в %s", str(PROCESSED_DATA_DIR.resolve()))

    finally:
        # Освобождаем память: unpersist всех закешированных DataFrame
        if clean is not None:
            clean.unpersist()
        if train is not None:
            train.unpersist()
        if val is not None:
            val.unpersist()
        if test is not None:
            test.unpersist()

        spark.stop()

    validate_data()


if __name__ == "__main__":
    process_data()
