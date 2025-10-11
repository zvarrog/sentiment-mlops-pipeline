"""Простая проверка API через /predict эндпоинт.

Отправляет несколько тестовых текстов и выводит предсказания.
"""

import requests

url = "http://localhost:8000/predict"
resp = requests.post(
    url,
    json={
        "texts": [
            "Loved this Kindle!",
            "Short story.",
            "Terrible experience",
        ]
    },
)
print(resp.status_code, resp.json())
