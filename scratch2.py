import requests

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": "Bearer hf_dummy_token"}
data = b"dummy"

response = requests.post(API_URL, headers=headers, data=data)
print("Status:", response.status_code)
print("Response:", response.text)
