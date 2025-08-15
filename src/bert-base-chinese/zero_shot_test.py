from transformers import BertTokenizer, BertModel
import torch
import numpy as np

"""
代码说明：
零样本原理：通过构建 "文本 [SEP] 标签" 的输入形式，让模型学习文本与标签之间的语义关联，然后通过计算关联度分数来判断文本最可能属于哪个类别。
预期结果：
与 "科技" 相关的文本应该被归类到 "科技"
与 "体育" 相关的文本应该被归类到 "体育"
与 "娱乐" 相关的文本应该被归类到 "娱乐"
局限性：
BERT 并非专门为零样本学习设计，效果可能不如 GPT 系列或专门的零样本模型（如 BART、T5）
分类准确性可能不如经过微调的模型
对于模糊或跨类别的文本，分类效果可能不理想
"""

model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


def zero_shot_classification(text, candidate_labels):
    """
    零样本文本分类：判断文本属于哪个候选类别
    """
    # 为每个候选标签创建一个"文本[SEP]标签"的输入
    inputs_list = []
    for label in candidate_labels:
        # 构建"文本[SEP]标签"形式的输入，让模型学习文本与标签的关联
        input_text = f"{text}[SEP]{label}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs_list.append(inputs)

    # 存储每个标签的分数
    scores = []

    # 对每个标签进行推理
    with torch.no_grad():
        for inputs in inputs_list:
            outputs = model(**inputs)
            # 使用[CLS] token的输出作为句子对的表示
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            # 计算一个简单的分数（可以理解为文本与标签的关联度）
            score = torch.norm(cls_embedding, p=2).item()  # L2范数作为分数
            scores.append(score)

    # 归一化分数
    scores = np.array(scores)
    normalized_scores = scores / scores.sum()

    # 返回分类结果
    result = {
        "text": text,
        "labels": [candidate_labels[i] for i in np.argsort(scores)[::-1]],
        "scores": [normalized_scores[i] for i in np.argsort(scores)[::-1]]
    }
    return result


# 测试文本
test_texts = [
    "中国选手在奥运会上获得了金牌",
    "新款智能手机配备了最新的处理器",
    "电影首映礼吸引了众多明星出席",
    "人工智能技术在医疗领域取得新突破",
    "国家队在世界杯预选赛中取得胜利"
]

# 候选类别
candidate_labels = ["科技", "体育", "娱乐"]

# 进行零样本分类并打印结果
for text in test_texts:
    result = zero_shot_classification(text, candidate_labels)
    print(f"文本: {text}")
    print("分类结果:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label}: {score:.4f}")
    print("-" * 50)
