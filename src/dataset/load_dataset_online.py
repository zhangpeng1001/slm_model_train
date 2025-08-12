"""
load_dataset的方式必须联网，即使下载到本地之后，依然会联网查询资源
"""
from datasets import load_dataset

cache_dir = r"E:/project/python/slm_model_train/src/dataset/datasets"

# 在线加载数据
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1", split="train", cache_dir=cache_dir)
dataset = load_dataset(path="NousResearch/hermes-function-calling-v1", cache_dir=cache_dir)

"""
Dataset({
    features: ['id', 'conversations', 'category', 'subcategory', 'task'],
    num_rows: 1893
})
"""
print(dataset)

####################################################################################################################
# 加载情感分析数据集
# 1. 首先加载数据集
dataset = load_dataset("lansinuote/ChnSentiCorp", cache_dir=cache_dir)
# print(dataset)

# 2. 保存为磁盘格式
save_path = r"E:/project/python/slm_model_train/src/dataset/data/ChnSentiCorp"
dataset.save_to_disk(save_path)

# 3. 后续可直接加载磁盘格式
from datasets import load_from_disk

loaded_dataset = load_from_disk(save_path)
print(loaded_dataset)
