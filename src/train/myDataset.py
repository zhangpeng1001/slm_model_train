"""
使用pytorch进行数据加载
"""
from torch.utils.data import Dataset
from datasets import load_from_disk


class MyDataset(Dataset):
    # 初始化数据
    def __init__(self, split):
        cache_dir = r"E:/project/python/slm_model_train/src/dataset/data/ChnSentiCorp"
        self.dataset = load_from_disk(cache_dir)
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        else:
            raise Exception("参数有误")

    # 获取数据集大小
    def __len__(self):
        return len(self.dataset)

    # 对数据定制化处理
    def __getitem__(self, item):
        # 数据示例   {"text":"","label":"0"}
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label


if __name__ == '__main__':
    dataset = MyDataset("test")
    for data in dataset:
        print(data)
