"""
加载本地数据集
"""
from datasets import load_from_disk

# 加载本地数据：失败
cache_dir = r"E:/project/python/slm_model_train/src/dataset/datasets"
dataset = load_from_disk(cache_dir)
print(dataset)
