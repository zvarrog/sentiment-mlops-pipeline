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

from scripts.settings import (
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
else:
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

    # Устойчивое удаление искусственного индексного столбца (leading comma / _c0 / BOM)
    first_col = df.columns[0]
    # нормализуем имя: убираем BOM и пробелы
    if isinstance(first_col, str):
        cleaned = first_col.strip()
        if cleaned.startswith("\ufeff"):
            cleaned = cleaned.lstrip("\ufeff")
    else:
        cleaned = first_col

    expected_cols = [
        "asin",
        "helpful",
        "overall",
        "reviewText",
        "reviewTime",
        "reviewerID",
        "reviewerName",
        "summary",
        "unixReviewTime",
    ]

    should_drop = False
    # явно пустое имя или стандартное имя парсера
    if (
        cleaned in ("", "_c0")
        or len(df.columns) == len(expected_cols) + 1
        and cleaned not in expected_cols
    ):
        should_drop = True

    if should_drop:
        log.info("Удаление индексного столбца: raw=%r cleaned=%r", first_col, cleaned)
        df = df.drop(first_col)
        log.info("Новый header: %s", ", ".join(df.columns))
    else:
        log.info(
            "Первый столбец валидный (raw=%r cleaned=%r) — пропускаю удаление",
            first_col,
            cleaned,
        )

    # Оставляем только нужные колонки
    cols = [
        "reviewerID",
        "asin",
        "reviewText",
        "overall",
        "unixReviewTime",
        "reviewTime",
    ]
    df = df.select(*cols)
    # Легкая очистка текста (truncate, lower, normalize symbols, remove html/url/non-latin, collapse spaces)
    MAX_TEXT_CHARS = 2000
    text_expr = lower(substring(col("reviewText"), 1, MAX_TEXT_CHARS))
    # Удаляем невидимые пробелы/марки: zero-width space, BOM, NBSP
    text_expr = regexp_replace(text_expr, r"[\u200b\ufeff\u00A0]", " ")
    # Убираем HTML/URL
    text_expr = regexp_replace(text_expr, r"<[^>]+>", " ")
    text_expr = regexp_replace(text_expr, r"http\S+", " ")
    # Нормализация типографских кавычек и тире
    text_expr = regexp_replace(text_expr, r"[\u2018\u2019]", "'")
    text_expr = regexp_replace(text_expr, r"[\u201C\u201D]", '"')
    text_expr = regexp_replace(text_expr, r"[\u2013\u2014]", "-")
    # Убираем типичные Kindle/ebook-метки, не несущие смысловой нагрузки
    text_expr = regexp_replace(
        text_expr,
        r"\b(kindle edition|prime reading|whispersync|borrow(?:ed)? for free|free sample|look inside)\b",
        " ",
    )
    # Только латиница и пробелы, схлопывание пробелов
    text_expr = regexp_replace(text_expr, r"[^a-z ]", " ")
    text_expr = regexp_replace(text_expr, r"\s+", " ")
    df = df.withColumn("reviewText", text_expr)

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

    clean = clean.withColumn("text_len", length(col("reviewText")))
    clean = clean.withColumn("word_count", size(split(col("reviewText"), " ")))
    # Частота слова 'kindle' в отзыве
    clean = clean.withColumn(
        "kindle_freq", size(split(lower(col("reviewText")), "kindle")) - 1
    )

    # Дополнительные информативные признаки
    clean = clean.withColumn(
        "exclamation_count", size(split(col("reviewText"), "!")) - 1
    )
    clean = clean.withColumn(
        "caps_ratio",
        length(regexp_replace(col("reviewText"), "[^A-Z]", ""))
        / greatest(length(col("reviewText")), lit(1)),
    )
    clean = clean.withColumn(
        "question_count", size(split(col("reviewText"), "\\?")) - 1
    )
    clean = clean.withColumn(
        "avg_word_length", col("text_len") / greatest(col("word_count"), lit(1))
    )

    # Sentiment анализ с TextBlob для более точных результатов
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType

    def calculate_sentiment_textblob(text):
        """Вычисляет sentiment score с помощью TextBlob.

        Возвращает polarity от -1 (негативный) до +1 (позитивный).
        Использует встроенные модели и словари TextBlob.
        """
        if not text or len(text.strip()) < 3:
            return 0.0

        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        # Ограничиваем диапазон и округляем для стабильности
        return float(max(-1.0, min(1.0, round(polarity, 4))))

    # Регистрируем UDF для sentiment анализа
    sentiment_udf = udf(calculate_sentiment_textblob, FloatType())

    clean = clean.withColumn("sentiment", sentiment_udf(col("reviewText")))

    # Дополнительные sentiment метрики для анализа
    clean = clean.withColumn(
        "sentiment_category",
        when(col("sentiment") > 0.1, "positive")
        .when(col("sentiment") < -0.1, "negative")
        .otherwise("neutral"),
    )
    clean = clean.withColumn(
        "sentiment_strength",
        when(col("sentiment").abs() > 0.5, "strong")
        .when(col("sentiment").abs() > 0.2, "moderate")
        .otherwise("weak"),
    )

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

    # TF-IDF: фитим векторизатор и IDF только на train и применяем к val/test тем же моделям
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
    idfModel = idf.fit(train_feat)
    train = idfModel.transform(train_feat)

    val_words = tokenizer.transform(val)
    val_feat = vec_model.transform(val_words)
    val = idfModel.transform(val_feat)

    test_words = tokenizer.transform(test)
    test_feat = vec_model.transform(test_words)
    test = idfModel.transform(test_feat)

    train = train.persist(StorageLevel.MEMORY_AND_DISK)
    val = val.persist(StorageLevel.MEMORY_AND_DISK)
    test = test.persist(StorageLevel.MEMORY_AND_DISK)
    log.info(
        "CountVectorizer: vocabSize<=%d, minDF=%s, minTF=%s",
        HASHING_TF_FEATURES,
        str(MIN_DF),
        str(MIN_TF),
    )

    # Агрегации на train
    user_stats = train.groupBy("reviewerID").agg(
        avg("text_len").alias("user_avg_len"),
        count("reviewText").alias("user_review_count"),
    )
    item_stats = train.groupBy("asin").agg(
        avg("text_len").alias("item_avg_len"),
        count("reviewText").alias("item_review_count"),
    )

    # Присоединяем агрегаты к каждому датасету
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

    # Сохраняем данные (путь должен быть строкой для Py4J/Java)
    train.write.mode("overwrite").parquet(str(TRAIN_PATH))
    val.write.mode("overwrite").parquet(str(VAL_PATH))
    test.write.mode("overwrite").parquet(str(TEST_PATH))

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
    # Безопасная остановка Spark: JVM могла завершиться к этому моменту
    try:
        spark.stop()
    except Exception as _e:
        log.warning("Spark JVM уже остановлена или недоступна: %s", _e)
