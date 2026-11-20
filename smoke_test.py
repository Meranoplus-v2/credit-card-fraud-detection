import pandas as pd
import requests
import random

df = pd.read_csv("creditcard.csv")  

BASE_URL = "http://localhost:8000"

# get 10 fraud and 10 legitimate samples
fraud_samples = df[df['Class'] == 1].sample(10).drop('Class', axis=1).to_dict(orient='records')
legit_samples = df[df['Class'] == 0].sample(10).drop('Class', axis=1).to_dict(orient='records')

for i in range(500): 
    is_fraud = random.random() < 0.2
    sample = random.choice(fraud_samples) if is_fraud else random.choice(legit_samples)

    # predict
    response = requests.post(f"{BASE_URL}/predict", json=sample)
    data = response.json()
    record_id = data["record_id"]
    print(f"{i+1}/500 | Predicted: {data['fraud']} | Actual: {is_fraud} | ID: {record_id}")

    # report outcome
    requests.post(f"{BASE_URL}/report/{record_id}?occurred={is_fraud}")

# check accuracy
accuracy = requests.get(f"{BASE_URL}/accuracy").json()
print(f"\nBrier Score: {accuracy['brier_score']}")
print(f"Observed Fraud Rate: {accuracy['observed_accident_rate']}")

# check drift
drift = requests.get(f"{BASE_URL}/drift").json()
print(f"\nDrift Detected: {drift['drift_detected']}")
print(f"Percent Change: {drift['percent_change']}%")
