"""Обработка kindle_reviews.csv c оптимизациями.

Изменения/особенности:
* Автоматическое удаление искусственного индексного столбца (leading comma / _c0)
* Балансировка по классам (звёздам) с настраиваемым лимитом по классу
* Минимизация числа shuffle (агрегации пользователя и товара в одном проходе каждая)
* Контролируемое число shuffle партиций для малых/средних объёмов
"""

from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml.feature import IDF, CountVectorizer, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    abs,
    avg,
    col,
    count,
    greatest,
    lit,
    rand,
    row_number,
    when,
)
from pyspark.sql.window import Window

from scripts.config import (
    CSV_NAME,
    DATA_PATHS,
    HASHING_TF_FEATURES,
    MIN_DF,
    MIN_TF,
    PER_CLASS_LIMIT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SHUFFLE_PARTITIONS,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_NUM_CORES,
)

from .logging_config import setup_auto_logging
from .feature_engineering import calculate_sentiment

log = setup_auto_logging()


def process_data(force: bool = False) -> None:
    """Основная функция обработки данных Spark.

    Args:
        force: Если True, обрабатывает заново даже если файлы существуют

    Выполняет:
    - Загрузку CSV
    - Очистку и нормализацию текста
    - Балансировку по классам
    - Добавление признаков
    - Sentiment анализ
    - TF-IDF векторизацию
    - Сохранение в parquet

    Вызывается только на driver, никогда на worker-ах.
    """
    if (
        not force
        and DATA_PATHS.train.exists()
        and DATA_PATHS.val.exists()
        and DATA_PATHS.test.exists()
    ):
        log.warning(
            "Обработанные данные уже существуют в %s. Для форсированной обработки используйте force=True.",
            str(PROCESSED_DATA_DIR),
        )
        return

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

    df = spark.read.csv(
        str(RAW_DATA_DIR / CSV_NAME),
        header=True,
        inferSchema=True,
        quote='"',
        escape='"',
        multiLine=True,
    )

    from scripts.feature_engineering import (
        get_spark_clean_udf,
        get_spark_feature_extraction_udf,
    )

    clean_udf = get_spark_clean_udf()
    feature_udf = get_spark_feature_extraction_udf()

    # Нормализуем текст
    df = df.withColumn("reviewText", clean_udf(col("reviewText")))

    # Извлекаем признаки в MapType и разворачиваем в колонки
    df = df.withColumn("features", feature_udf(col("reviewText")))
    for feature_name in [
        "text_len",
        "word_count",
        "kindle_freq",
        "exclamation_count",
        "caps_ratio",
        "question_count",
    ]:
        df = df.withColumn(feature_name, col("features").getItem(feature_name))
    df = df.drop("features")

    # Чистим данные: валидные тексты и оценки, исключаем пустые/пробельные тексты
    clean = df.filter(
        (col("reviewText").isNotNull())
        & (col("overall").isNotNull())
        & (col("text_len") > 0)
    )
    log.info("После фильтрации null и пустых текстов: %s", clean.count())

    # Балансировка через Window-функцию (эффективнее цикла по классам)
    window_spec = Window.partitionBy("overall").orderBy(rand(seed=42))
    clean = (
        clean.withColumn("rn", row_number().over(window_spec))
        .filter(col("rn") <= PER_CLASS_LIMIT)
        .drop("rn")
    )
    log.info(
        "После балансировки (<= %d на класс) строк: %d", PER_CLASS_LIMIT, clean.count()
    )

    # sentiment через Pandas UDF
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import FloatType

    @pandas_udf(FloatType())
    def sentiment_udf(texts: pd.Series) -> pd.Series:
        return texts.fillna("").apply(calculate_sentiment)

    clean = (
        clean.withColumn(
            "avg_word_length", col("text_len") / greatest(col("word_count"), lit(1))
        )
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

    clean = clean.persist(StorageLevel.MEMORY_AND_DISK)

    # Материализуем кэш через action
    features_count = clean.count()
    log.info("После всех фич строк: %d", features_count)

    # Делим на выборки
    train, val, test = clean.randomSplit([0.7, 0.15, 0.15], seed=42)
    tr_c, v_c, te_c = train.count(), val.count(), test.count()
    log.info(
        "Размеры выборок: train=%d, val=%d, test=%d (total=%d)",
        tr_c,
        v_c,
        te_c,
        tr_c + v_c + te_c,
    )

    # TF-IDF: применяем векторизатор и IDF только на train и применяем к val/test тем же моделям
    tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
    vectorizer = CountVectorizer(
        inputCol="words",
        outputCol="rawFeatures",
        vocabSize=HASHING_TF_FEATURES,
        minDF=MIN_DF,
        minTF=MIN_TF,
    )
    idf = IDF(inputCol="rawFeatures", outputCol="tfidfFeatures")

    train_words = tokenizer.transform(train)
    vec_model = vectorizer.fit(train_words)
    train_feat = vec_model.transform(train_words)
    idf_model = idf.fit(train_feat)
    train = idf_model.transform(train_feat)

    val_words = tokenizer.transform(val)
    val_feat = vec_model.transform(val_words)
    val = idf_model.transform(val_feat)

    test_words = tokenizer.transform(test)
    test_feat = vec_model.transform(test_words)
    test = idf_model.transform(test_feat)

    train = train.persist(StorageLevel.MEMORY_AND_DISK)
    val = val.persist(StorageLevel.MEMORY_AND_DISK)
    test = test.persist(StorageLevel.MEMORY_AND_DISK)
    log.info(
        "CountVectorizer: vocabSize<=%d, minDF=%s, minTF=%s",
        HASHING_TF_FEATURES,
        str(MIN_DF),
        str(MIN_TF),
    )

    try:
        # Агрегации на train — для избежания data leakage
        # Статистики для val и test будут заполнены только для известных ID
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

        train = train.join(user_stats, on="reviewerID", how="left").join(
            item_stats, on="asin", how="left"
        )
        val = val.join(user_stats, on="reviewerID", how="left").join(
            item_stats, on="asin", how="left"
        )
        test = test.join(user_stats, on="reviewerID", how="left").join(
            item_stats, on="asin", how="left"
        )

        log.info(
            "После добавления агрегатов кол-во колонок в train: %d", len(train.columns)
        )

        train.repartition(4).write.mode("overwrite").parquet(str(DATA_PATHS.train))
        val.repartition(2).write.mode("overwrite").parquet(str(DATA_PATHS.val))
        test.repartition(2).write.mode("overwrite").parquet(str(DATA_PATHS.test))
    finally:
        if 'clean' in locals() and hasattr(clean, 'unpersist'):
            clean.unpersist()
        train.unpersist()
        val.unpersist()
        test.unpersist()
        # Очищаем кэши агрегатов (если переменные определены)
        if "user_stats" in locals() and hasattr(user_stats, "unpersist"):
            user_stats.unpersist()
        if "item_stats" in locals() and hasattr(item_stats, "unpersist"):
            item_stats.unpersist()

    log.info(
        "Данные сохранены в %s",
        str(PROCESSED_DATA_DIR.resolve()),
    )

    from scripts.config import RUN_DATA_VALIDATION

    if RUN_DATA_VALIDATION:
        try:
            from .data_validation import (
                log_validation_results,
                validate_parquet_dataset,
            )

            validation_results = validate_parquet_dataset(Path(PROCESSED_DATA_DIR))
            all_valid = log_validation_results(validation_results)

            if not all_valid:
                log.warning("Обнаружены проблемы в сохранённых данных")

        except (OSError, ValueError, ImportError) as e:
            log.warning("Ошибка валидации сохранённых данных: %s", e)

    spark.stop()


if __name__ == "__main__":
    process_data()
