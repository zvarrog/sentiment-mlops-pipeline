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
    length,
    lit,
    lower,
    rand,
    regexp_replace,
    size,
    split,
    substring,
    when,
)

from scripts.config import (
    CSV_NAME,
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
from .text_features import calculate_sentiment

TRAIN_PATH = PROCESSED_DATA_DIR / "train.parquet"
VAL_PATH = PROCESSED_DATA_DIR / "val.parquet"
TEST_PATH = PROCESSED_DATA_DIR / "test.parquet"

log = setup_auto_logging()


def process_data() -> None:
    """Основная функция обработки данных Spark.

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
    # Проверяем флаг форсированной обработки
    from scripts.config import FORCE_PROCESS

    force_process = FORCE_PROCESS

    if (
        not force_process
        and TRAIN_PATH.exists()
        and VAL_PATH.exists()
        and TEST_PATH.exists()
    ):
        log.warning(
            "Обработанные данные уже существуют в %s. Для форсированной обработки установите флаг FORCE_PROCESS = True.",
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

    max_text_chars = 5000
    # Фичи пунктуации
    df = df.withColumn("text_len", length(col("reviewText")))
    df = df.withColumn("word_count", size(split(col("reviewText"), " ")))
    df = df.withColumn(
        "kindle_freq", (size(split(lower(col("reviewText")), "kindle")) - 1)
    )
    df = df.withColumn("exclamation_count", (size(split(col("reviewText"), "!")) - 1))
    df = df.withColumn(
        "caps_ratio",
        length(regexp_replace(col("reviewText"), "[^A-Z]", ""))
        / greatest(length(col("reviewText")), lit(1)),
    )
    df = df.withColumn("question_count", (size(split(col("reviewText"), "\\?")) - 1))
    # Чистка текста
    text_col = lower(substring(col("reviewText"), 1, max_text_chars))
    text_col = regexp_replace(text_col, r"http\S+", " ")
    text_col = regexp_replace(text_col, r"[^a-z ]+", " ")
    text_col = regexp_replace(text_col, r"\s+", " ")
    df = df.withColumn("reviewText", text_col)

    # Чистим данные: валидные тексты и оценки
    clean = df.filter((col("reviewText").isNotNull()) & (col("overall").isNotNull()))
    log.info("После фильтрации null: %s", clean.count())

    class_counts = clean.groupBy("overall").count().collect()
    min_class_size = min([row["count"] for row in class_counts])
    sample_size = min(min_class_size, PER_CLASS_LIMIT)

    balanced_dfs = []
    for rating in [1, 2, 3, 4, 5]:
        class_df = clean.filter(col("overall") == rating)
        sampled = class_df.orderBy(rand(seed=42)).limit(sample_size)
        balanced_dfs.append(sampled)

    clean = balanced_dfs[0]
    for df_part in balanced_dfs[1:]:
        clean = clean.union(df_part)
    log.info(
        "После балансировки (<= %d на класс) строк: %d", PER_CLASS_LIMIT, clean.count()
    )

    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType

    @udf(FloatType())
    def calculate_sentiment_textblob(text: str) -> float:
        """UDF обёртка для общей функции sentiment из text_features."""
        return calculate_sentiment(text)

    clean = (
        clean.withColumn(
            "avg_word_length", col("text_len") / greatest(col("word_count"), lit(1))
        )
        .withColumn("sentiment", calculate_sentiment_textblob(col("reviewText")))
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
        # Агрегации на train
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

        # Батчируем join для всех выборок
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

        train.write.mode("overwrite").parquet(str(TRAIN_PATH))
        val.write.mode("overwrite").parquet(str(VAL_PATH))
        test.write.mode("overwrite").parquet(str(TEST_PATH))
    finally:
        train.unpersist()
        val.unpersist()
        test.unpersist()
        # Очищаем кэши агрегатов (если переменные определены)
        try:
            user_stats.unpersist()
            item_stats.unpersist()
        except Exception:
            pass

    log.info(
        "Данные сохранены в %s",
        str(PROCESSED_DATA_DIR.resolve()),
    )

    from scripts.config import RUN_DATA_VALIDATION

    if RUN_DATA_VALIDATION:
        log.info("Проверка: запуск валидации сохранённых parquet файлов...")
        try:
            # Используем pandas для валидации после записи Spark
            from .data_validation import (
                log_validation_results,
                validate_parquet_dataset,
            )

            validation_results = validate_parquet_dataset(Path(PROCESSED_DATA_DIR))
            all_valid = log_validation_results(validation_results)

            if all_valid:
                log.info("Валидация сохранённых данных успешно завершена")
            else:
                log.warning("Обнаружены проблемы в сохранённых данных")

        except Exception as e:
            log.warning("Ошибка валидации сохранённых данных: %s", e)

    log.info("Обработка завершена.")
    spark.stop()


if __name__ == "__main__":
    process_data()
