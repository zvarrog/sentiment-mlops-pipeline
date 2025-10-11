"""Улучшенный тест API с проверкой TextBlob sentiment анализа."""

import requests


def test_textblob_sentiment_consistency():
    """Проверяем, что API использует TextBlob аналогично обучению."""

    # Тестовые данные с известными sentiment характеристиками
    test_cases = [
        {
            "text": "This book is absolutely amazing and wonderful! I love it so much!",
            "expected_sentiment": "positive",
            "description": "Очень позитивный отзыв",
        },
        {
            "text": "This is terrible, awful and worst book ever. I hate it completely.",
            "expected_sentiment": "negative",
            "description": "Очень негативный отзыв",
        },
        {
            "text": "The book was okay, nothing special really.",
            "expected_sentiment": "neutral",
            "description": "Нейтральный отзыв",
        },
        {
            "text": "Great story but terrible ending. Love the characters but hate the plot.",
            "expected_sentiment": "mixed",
            "description": "Смешанный отзыв",
        },
    ]

    api_url = "http://localhost:8000"

    print("=== Тест TextBlob Sentiment в API ===")

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print(f"Текст: {case['text']}")

        # Запрос к API с увеличенным timeout
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"texts": [case["text"]]},
                timeout=30,  # Увеличили timeout до 30 секунд
            )

            if response.status_code == 200:
                data = response.json()
                prediction = data["labels"][0]
                probs = data.get("probs", [None])[0] if data.get("probs") else None
                warnings = data.get("warnings", {})

                print(f"Предсказание: {prediction} звёзд")
                if probs:
                    print(f"Вероятности: {[round(p, 3) for p in probs]}")

                # Проверяем, что sentiment был рассчитан (должен быть в warnings как auto_filled)
                auto_filled = (
                    warnings.get("auto_filled_numeric_columns", []) if warnings else []
                )
                if "sentiment" in auto_filled:
                    print("✓ Sentiment рассчитан автоматически через TextBlob")
                else:
                    print("⚠ Sentiment не найден в auto_filled колонках")
                    print(f"Warnings: {warnings}")
                    if warnings:
                        print(f"Available warning keys: {list(warnings.keys())}")

            else:
                print(f"❌ Ошибка API: {response.status_code}")
                print(response.text)

        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка соединения: {e}")

    print("\n=== Тест batch_predict с полными данными ===")

    # Тест с полными данными включая числовые признаки
    batch_data = [
        {
            "reviewText": "Amazing book with excellent story!",
            "user_review_count": 15.0,
            "text_len": 35.0,
        },
        {
            "reviewText": "Terrible waste of time and money.",
            "user_review_count": 8.0,
            "text_len": 29.0,
        },
    ]

    try:
        response = requests.post(
            f"{api_url}/batch_predict",
            json={"data": batch_data},
            timeout=30,  # Увеличили timeout
        )

        if response.status_code == 200:
            data = response.json()
            predictions = data["predictions"]
            warnings = data.get("warnings", {})

            for pred in predictions:
                idx = pred["index"]
                text = batch_data[idx]["reviewText"]
                label = pred["prediction"]
                probs = pred.get("probabilities", [])

                print(f"\nТекст: {text}")
                print(f"Предсказание: {label} звёзд")
                if probs:
                    print(f"Вероятности: {[round(p, 3) for p in probs]}")

        else:
            print(f"❌ Ошибка batch_predict: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка соединения: {e}")


def test_metadata_and_health():
    """Проверка metadata и health endpoints."""

    api_url = "http://localhost:8000"

    print("\n=== Проверка Health ===")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"Статус: {health.get('status')}")
            print(f"Модель загружена: {health.get('model_exists')}")
            print(f"Лучшая модель: {health.get('best_model')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка health check: {e}")

    print("\n=== Проверка Metadata ===")
    try:
        response = requests.get(f"{api_url}/metadata", timeout=5)
        if response.status_code == 200:
            metadata = response.json()

            # Информация о модели
            model_info = metadata.get("model_info", {})
            print(f"Модель: {model_info.get('best_model', 'N/A')}")
            print(
                f"Accuracy: {model_info.get('test_metrics', {}).get('accuracy', 'N/A')}"
            )

            # Feature contract
            feature_contract = metadata.get("feature_contract", {})
            expected_numeric = feature_contract.get("expected_numeric_columns", [])
            print(f"Ожидаемые числовые колонки: {len(expected_numeric)}")

            # Проверяем, что sentiment в числовых колонках
            if "sentiment" in expected_numeric:
                print("✓ sentiment входит в feature contract")
            else:
                print("⚠ sentiment НЕ найден в feature contract")
                print(f"Числовые колонки: {expected_numeric}")

        else:
            print(f"❌ Metadata failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка metadata: {e}")


def compare_sentiment_methods():
    """Сравнение старого и нового методов sentiment анализа."""

    test_texts = [
        "This book is absolutely amazing and I love it!",
        "Terrible book, worst story ever, complete waste of time.",
        "The book was okay, nothing really special about it.",
        "Good start but disappointing ending overall.",
    ]

    print("\n=== Сравнение Sentiment методов ===")

    for text in test_texts:
        print(f"\nТекст: {text}")

        # TextBlob sentiment
        try:
            from textblob import TextBlob

            blob = TextBlob(text)
            textblob_score = round(blob.sentiment.polarity, 4)
            print(f"TextBlob sentiment: {textblob_score}")
        except ImportError:
            print("TextBlob не установлен")

        # Простой словарный метод (старый)
        positive_words = ["good", "great", "excellent", "love", "amazing", "best"]
        negative_words = ["bad", "terrible", "awful", "worst", "hate", "horrible"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            simple_score = 1.0
        elif neg_count > pos_count:
            simple_score = -1.0
        else:
            simple_score = 0.0

        print(f"Простой sentiment: {simple_score} (pos:{pos_count}, neg:{neg_count})")


if __name__ == "__main__":
    print("Тестирование API с TextBlob sentiment анализом")
    print("Убедитесь, что API запущен на http://localhost:8000")

    test_textblob_sentiment_consistency()
    test_metadata_and_health()
    compare_sentiment_methods()

    print("\n=== Тест завершён ===")
