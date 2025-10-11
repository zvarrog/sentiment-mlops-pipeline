"""Простая проверка API через /batch_predict: полагаемся на автогенерацию признаков в API."""

import pandas as pd
import requests

SAMPLE_SIZE = 5

df = pd.read_csv("data/raw/kindle_reviews.csv").sample(
    n=SAMPLE_SIZE
)  # , random_state=42)

# Формируем минимальный payload — только тексты; признаки посчитает API
payload = {"data": [{"reviewText": str(t or "")} for t in df["reviewText"].fillna("")]}

resp = requests.post("http://localhost:8000/batch_predict", json=payload)
data = resp.json()
preds = data.get("predictions", [])

for i, (text, stars) in enumerate(zip(df["reviewText"], df.get("overall"))):
    preview = text[:200] + "..." if isinstance(text, str) and len(text) > 200 else text
    pred_item = preds[i]
    label = pred_item.get("prediction")
    print(f"📖 {preview}")
    print(f"⭐ {stars} -> predict: {label}\n")
