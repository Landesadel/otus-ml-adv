import requests
import json

url = "http://localhost:8000/predict"
headers = {'Content-Type': 'application/json'}

data = {
    "age": 57,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 236,
    "fbs": 0,
    "restecg": 1,
    "thalach": 174,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 1,
    "ca": 1,
    "thal": 2
}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
