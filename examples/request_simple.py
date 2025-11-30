import requests

# Пример простого запроса на /predict
texts = [
    "This book is absolutely amazing! Best read of the year!",
    "Terrible waste of money. Don't buy this.",
    "It's okay, nothing special but readable.",
    "I loved every page of this wonderful story!",
    "Boring and poorly written.",
]

payload = {"texts": texts}

resp = requests.post("http://localhost:8000/predict", json=payload)
resp.raise_for_status()
data = resp.json()

print("Predictions:")
for i, (text, label) in enumerate(zip(texts, data["labels"])):
    preview = text[:60] + "..." if len(text) > 60 else text
    print(f"{i + 1}. {preview} -> Rating: {label}")

if data.get("probs"):
    print("\nProbabilities available:", len(data["probs"]))
