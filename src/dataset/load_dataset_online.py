"""
load_dataset的方式必须联网，即使下载到本地之后，依然会联网查询资源
"""
from datasets import load_dataset

# 在线加载数据
cache_dir = r"E:/project/python/slm_model_train/src/dataset/datasets"
dataset = load_dataset(path="NousResearch/hermes-function-calling-v1", split="train", cache_dir=cache_dir)

print(dataset)
