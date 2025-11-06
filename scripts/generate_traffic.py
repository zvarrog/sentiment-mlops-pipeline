"""Генерация непрерывного трафика для накопления метрик в Prometheus/Grafana."""

import time

import requests

texts = [
    "Absolutely amazing product! Highly recommended!",
    "Terrible quality, waste of money.",
    "Average, nothing special.",
    "Love it! Best purchase ever!",
    "Disappointed, expected better.",
]

print("Генерация трафика для мониторинга метрик...")
print("Нажмите Ctrl+C для остановки\n")

counter = 0
try:
    while True:
        counter += 1

        # Запрос к /predict
        try:
            resp = requests.post(
                "http://localhost:8000/predict",
                json={"texts": texts},
                timeout=10
            )
            status = resp.status_code
            print(f"#{counter:3d} /predict -> {status} ({len(texts)} texts)")
        except Exception as e:
            print(f"#{counter:3d} /predict -> ERROR: {e}")
        
        time.sleep(5)  # Пауза 5 секунд между запросами
        
except KeyboardInterrupt:
    print(f"\n\nОстановлено. Всего отправлено: {counter} запросов")
