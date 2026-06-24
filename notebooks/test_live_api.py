import requests

BASE = "https://sentimentops-api.thankfulmeadow-07d26880.uaenorth.azurecontainerapps.io"

print(f"Testing live API at: {BASE}\n")

# ── 1. Root ───────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/")
print("Root:", r.json())

# ── 2. Health ─────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/health")
print("Health:", r.json())

# ── 3. sklearn predict ────────────────────────────────────────────────
reviews = [
    "This movie was absolutely fantastic, I loved every minute",
    "Terrible waste of time, worst movie I have ever seen",
    "Not bad but not great either, just average",
]

print("\n── sklearn predictions ──────────────────────────")
for review in reviews:
    r = requests.post(f"{BASE}/predict", json={"text": review})
    data = r.json()
    print(f"{data['sentiment'].upper()} ({data['confidence']:.2%}) — {review[:50]}")

# ── 4. DistilBERT predict ─────────────────────────────────────────────
print("\n── distilbert predictions ───────────────────────")
for review in reviews:
    r = requests.post(f"{BASE}/predict/distilbert", json={"text": review})
    data = r.json()
    print(f"{data['sentiment'].upper()} ({data['confidence']:.2%}) — {review[:50]}")

# ── 5. Compare ────────────────────────────────────────────────────────
print("\n── compare on ambiguous review ──────────────────")
r = requests.post(f"{BASE}/predict/compare",
    json={"text": "Not bad but not great either, just average"})
print(r.json())