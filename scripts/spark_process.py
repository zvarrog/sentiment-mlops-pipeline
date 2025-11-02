"""Обработка kindle_reviews.csv c оптимизациями.

Изменения/особенности:
* Автоматическое удаление искусственного индексного столбца (leading comma / _c0)
* Балансировка по классам (звёздам) с настраиваемым лимитом по классу
* Минимизация числа shuffle (агрегации пользователя и товара в одном проходе каждая)
* Контролируемое число shuffle партиций для малых/средних объёмов
"""

import contextlib
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
    regexp_replace,
    row_number,
    size,
    split,
    substring,
    when,
)
from pyspark.sql.window import Window

from scripts.config import (
    CSV_NAME,
    FORCE_PROCESS,
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
    if (
        not FORCE_PROCESS
        and TRAIN_PATH.exists()
        and VAL_PATH.exists()
        and TEST_PATH.exists()
    ):
        log.warning(
            "Обработанные данные уже существуют в %s. Для форсированной обработки установите флаг FORCE_PROCESS = True.",
            str(PROCESSED_DATA_DIR),
        )
        return

    # Создаём SparkSession (если скрипт запускается сам по себе)
    try:
        spark = (
            SparkSession.builder.appName("KindleReviews")
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .config("spark.executor.cores", str(SPARK_NUM_CORES))
            .config("spark.driver.cores", str(SPARK_NUM_CORES))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
    except Exception:
        # если SparkSession уже создан в окружении, просто получим текущий
        spark = SparkSession.builder.getOrCreate()

    # Снижаем число shuffle партиций для средних объёмов
    with contextlib.suppress(Exception):
        spark.conf.set("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))

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
    text_col = lower(substring(col("reviewText"), 1, max_text_chars))
    text_col = regexp_replace(text_col, r"[^a-z ]+", " ")
    df = df.withColumn("reviewText", text_col)

    # Чистим данные: валидные тексты и оценки
    clean = df.filter((col("reviewText").isNotNull()) & (col("overall").isNotNull()))
    log.info("После фильтрации null: %s", clean.count())

    # Балансировка: берём последние n по каждому классу
    window = Window.partitionBy("overall").orderBy(col("unixReviewTime").desc())
    clean = (
        clean.withColumn("row_num", row_number().over(window))
        .filter(col("row_num") <= PER_CLASS_LIMIT)
        .drop("row_num")
    )
    balanced_count = clean.count()
    log.info(
        "После балансировки (<= %d на класс) строк: %d", PER_CLASS_LIMIT, balanced_count
    )

    # Батчируем добавление признаков в один .select() вместо цепи .withColumn()
    clean = clean.select(
        "*",
        length(col("reviewText")).alias("text_len"),
        size(split(col("reviewText"), " ")).alias("word_count"),
        (size(split(lower(col("reviewText")), "kindle")) - 1).alias("kindle_freq"),
        (size(split(col("reviewText"), "!")) - 1).alias("exclamation_count"),
        (
            length(regexp_replace(col("reviewText"), "[^A-Z]", ""))
            / greatest(length(col("reviewText")), lit(1))
        ).alias("caps_ratio"),
        (size(split(col("reviewText"), "\\?")) - 1).alias("question_count"),
    )

    # Добавляем производную колонку avg_word_length после того как есть text_len и word_count
    clean = clean.withColumn(
        "avg_word_length", col("text_len") / greatest(col("word_count"), lit(1))
    )

    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType
    from textblob import TextBlob

    @udf(FloatType())
    def calculate_sentiment_textblob(text: str) -> float:
        """Вычисляет sentiment score с помощью TextBlob.

        Возвращает polarity от -1 (негативный) до +1 (позитивный).
        Использует встроенные модели и словари TextBlob.

        Эта функция запускается на worker-ах Spark,
        поэтому НЕ должна создавать SparkSession или зависеть от внешних ресурсов.
        """
        if not text or len(text.strip()) < 3:
            return 0.0

        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return float(max(-1.0, min(1.0, round(polarity, 4))))
        except Exception:
            # На случай ошибки парсинга — возвращаем нейтральный sentiment
            return 0.0

    clean = clean.withColumn("sentiment", calculate_sentiment_textblob(col("reviewText")))

    # Дополнительные sentiment метрики для анализа
    clean = clean.withColumn(
        "sentiment_category",
        when(col("sentiment") > 0.1, "positive")
        .when(col("sentiment") < -0.1, "negative")
        .otherwise("neutral"),
    )
    clean = clean.withColumn(
        "sentiment_strength",
        when(abs(col("sentiment")) > 0.5, "strong")
        .when(abs(col("sentiment")) > 0.2, "moderate")
        .otherwise("weak"),
    )

    clean = clean.persist(StorageLevel.MEMORY_AND_DISK)

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
        str(Path(PROCESSED_DATA_DIR).resolve()),
    )

    # Валидация сохранённых данных
    import os

    _val_raw = os.environ.get("RUN_DATA_VALIDATION", "1").strip().lower()
    run_validation = _val_raw in {"1", "true", "yes", "y", "on"}

    if run_validation:
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
    else:
        log.info("Валидация сохранённых данных пропущена: RUN_DATA_VALIDATION=0")

    log.info("Обработка завершена.")
    # Безопасная остановка Spark и закрытие Py4J gateway
    try:
        spark.stop()
    except Exception as _e:
        log.warning("Spark JVM уже остановлена или недоступна: %s", _e)
    # Пытаемся закрыть Py4J gateway, чтобы не оставлять фоновые процессы
    try:
        sc = getattr(spark, "_sc", None)
        if sc is not None and getattr(sc, "_gateway", None) is not None:
            sc._gateway.close()
    except Exception as _e:
        log.debug("Не удалось закрыть Py4J gateway: %s", _e)


if __name__ == "__main__":
    process_data()
