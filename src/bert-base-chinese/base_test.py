from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


# 1. 测试基本编码能力
def test_basic_encoding():
    text = "我爱自然语言处理"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    print(f"输入文本: {text}")
    print(f"编码结果形状: {last_hidden_state.shape}")  # 应为 [1, 分词数, 768]


# 2. 测试语义相似度计算能力
def test_semantic_similarity():
    texts = [
        "猫坐在垫子上",
        "小猫在垫子上休息",
        "汽车行驶在马路上"
    ]

    # 获取句子向量
    sentence_vectors = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # 使用[CLS] token的向量作为句子表示
        cls_vector = outputs.last_hidden_state[:, 0, :].numpy()
        sentence_vectors.append(cls_vector)

    # 计算相似度
    sim1 = cosine_similarity(sentence_vectors[0], sentence_vectors[1])[0][0]
    sim2 = cosine_similarity(sentence_vectors[0], sentence_vectors[2])[0][0]

    print(f"句子1与句子2的相似度: {sim1:.4f}")  # 应该较高
    print(f"句子1与句子3的相似度: {sim2:.4f}")  # 应该较低


# 3. 测试一词多义理解能力
def test_word_sense_disambiguation():
    sentences = [
        "他买了一个苹果手机",
        "她正在吃苹果"
    ]

    # 查找"苹果"在两个句子中的位置
    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        inputs = tokenizer(sent, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # 找到"苹果"对应的token索引
        apple_idx = tokens.index("苹")  # 注意BERT的分词可能会将词语分开
        apple_vector = outputs.last_hidden_state[0, apple_idx + 1, :].numpy()  # +1是因为有[CLS]

        print(f"\n句子: {sent}")
        print(f"分词结果: {tokens}")
        print(f"苹果的向量维度: {apple_vector.shape}")  # 应为 (768,)


# 运行测试
test_basic_encoding()
test_semantic_similarity()
test_word_sense_disambiguation()
