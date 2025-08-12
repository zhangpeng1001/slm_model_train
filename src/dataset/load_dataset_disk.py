"""
加载本地数据集
"""
from datasets import load_from_disk

cache_dir = r"E:/project/python/slm_model_train/src/dataset/data/ChnSentiCorp"

loaded_dataset = load_from_disk(cache_dir)
print(loaded_dataset)
