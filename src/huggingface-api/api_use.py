import requests

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"

#不使用token进行匿名访问
response = requests.post(API_URL,json={"inputs":"你好，huggingface"})

#不成功： {'error': 'Invalid username or password.'}
print(response.json())