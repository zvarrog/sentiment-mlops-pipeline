import pandas as pd
import requests

SAMPLE_SIZE = 5

df = pd.read_csv("data/raw/kindle_reviews.csv").sample(n=SAMPLE_SIZE)

payload = {"data": [{"reviewText": str(t or "")} for t in df["reviewText"].fillna("")]}

resp = requests.post("http://localhost:8000/batch_predict", json=payload)
resp.raise_for_status()
data = resp.json()
preds = data.get("predictions", [])

for i, (text, stars) in enumerate(
    zip(df["reviewText"], df.get("overall"), strict=False)
):
    preview = text[:200] + "..." if isinstance(text, str) and len(text) > 200 else text
    label = int(preds[i]) if i < len(preds) else None
    print(f"TEXT: {preview}")
    print(f"RATING {stars} -> predict: {label}\n")
