import requests
from src.secret import API_TOKEN

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"

# 使用token进行访问
headers = {"Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(API_URL, headers=headers, json={"inputs": "你好，huggingface"})

# 不成功： 404
print(response)
# print(response.json())