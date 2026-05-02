import requests

API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
with open("c:/Users/Aryan/Downloads/cat and dog classifier/test.csv", "rb") as f:
    # Just sending a tiny dummy payload to see if it gives 400 Bad Request or "Cannot POST"
    data = b"dummy_image_data"
    
response = requests.post(API_URL, data=data)
print(response.status_code)
print(response.text)
