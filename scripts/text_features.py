"""Общие функции обработки текста и извлечения признаков.

Используется как в Spark (UDF), так и в Pandas (API) для синхронизации обработки.
Все функции — чистые, работают с базовыми типами Python.
"""


def clean_text(text: str, max_length: int = 5000) -> str:
    """Нормализует текст: lowercase, удаление URL, пунктуации, лишних пробелов.

    Args:
        text: Исходный текст
        max_length: Максимальная длина текста после обрезки

    Returns:
        Очищенный текст
    """
    if not text or not isinstance(text, str):
        return ""

    # Обрезка до max_length
    text = text[:max_length]

    # Lowercase
    text = text.lower()

    # Удаление URL
    import re

    text = re.sub(r"http\S+", " ", text)

    # Только буквы и пробелы
    text = re.sub(r"[^a-z ]+", " ", text)

    # Нормализация пробелов
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_sentiment(text: str) -> float:
    """Вычисляет sentiment score с помощью TextBlob.

    Args:
        text: Текст для анализа

    Returns:
        Polarity от -1 (негативный) до +1 (позитивный)
    """
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
    """Извлекает базовые текстовые признаки из сырого текста.

    Признаки вычисляются ДО чистки текста для сохранения информации о пунктуации.

    Args:
        text: Исходный текст (сырой, до clean_text)

    Returns:
        Словарь с признаками: text_len, word_count, kindle_freq, exclamation_count,
        caps_ratio, question_count
    """
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

    # Частота упоминания "kindle"
    kindle_freq = float(text.lower().count("kindle"))

    # Знаки препинания
    exclamation_count = float(text.count("!"))
    question_count = float(text.count("?"))

    # Соотношение заглавных букв
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


def calculate_avg_word_length(text_len: float, word_count: float) -> float:
    """Вычисляет среднюю длину слова.

    Args:
        text_len: Длина текста в символах
        word_count: Количество слов

    Returns:
        Средняя длина слова
    """
    return text_len / max(word_count, 1.0)
