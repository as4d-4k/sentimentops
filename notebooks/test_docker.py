import requests

# health check
r = requests.get("http://127.0.0.1:8000/health")
print("Health:", r.json())

# positive review
r = requests.post("http://127.0.0.1:8000/predict",
    json={"text": "This movie was absolutely fantastic"})
print("Positive:", r.json())

# negative review
r = requests.post("http://127.0.0.1:8000/predict",
    json={"text": "Terrible waste of time, worst movie ever"})
print("Negative:", r.json())

# batch
r = requests.post("http://127.0.0.1:8000/predict/batch",
    json={"texts": [
        "Absolutely brilliant film",
        "Complete waste of time",
        "No Country for Old Men is a masterpiece"
    ]})
print("Batch:", r.json())