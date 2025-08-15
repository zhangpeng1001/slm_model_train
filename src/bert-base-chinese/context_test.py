"""
以下是一个测试bert-base-chinese模型上下文指代关系理解能力的示例。这个测试通过计算代词与可能指代对象之间的语义相似度，来判断模型是否能正确捕捉指代关系。
"""
"""
代码说明：
测试原理：通过计算代词与候选指代对象在模型隐藏状态中的余弦相似度，来判断模型是否认为它们存在语义关联。相似度越高，说明模型越可能认为该候选对象是代词的指代对象。
预期结果：
第一个案例中，"他" 应该与 "小明" 的相似度更高
第二个案例中，"她" 应该与 "小丽" 的相似度更高
第三个案例中，"他们" 应该与 "学生们" 的相似度更高
结果解读：
如果模型计算出的最高相似度对应正确的指代对象，说明模型具备一定的上下文指代理解能力
由于bert-base-chinese没有专门针对指代消解任务进行微调，其表现可能不如专门的指代消解模型

这个测试展示了预训练语言模型对中文中常见指代关系的理解能力。对于更复杂的指代场景（如远距离指代、歧义指代），模型的表现可能会下降，这时候通常需要在专门的指代消解数据集上进行微调以获得更好的性能。
"""
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


def test_coreference_resolution(sentence, pronoun, candidates):
    """
    测试模型对指代关系的理解能力
    sentence: 包含指代关系的句子
    pronoun: 要测试的代词（如"他"、"她"、"它"）
    candidates: 可能的指代对象列表
    """
    # 对句子进行编码
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.tokenize(sentence)

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state[0].numpy()  # 取第一个样本的隐藏状态

    # 找到代词在分词结果中的位置
    try:
        pronoun_indices = [i for i, token in enumerate(tokens) if pronoun in token]
        if not pronoun_indices:
            print(f"在句子中未找到代词 '{pronoun}'")
            return

        # 取第一个出现的代词位置
        pronoun_idx = pronoun_indices[0]
        pronoun_embedding = hidden_states[pronoun_idx]
        print(f"句子: {sentence}")
        print(f"分词结果: {tokens}")
        print(f"代词 '{pronoun}' 的位置: {pronoun_idx}")

        # 计算代词与每个候选对象的相似度
        similarities = []
        for candidate in candidates:
            # 找到候选对象在分词结果中的位置
            candidate_indices = [i for i, token in enumerate(tokens) if candidate in token]
            if candidate_indices:
                # 取第一个出现的候选对象位置
                candidate_idx = candidate_indices[0]
                candidate_embedding = hidden_states[candidate_idx]

                # 计算余弦相似度
                sim = cosine_similarity([pronoun_embedding], [candidate_embedding])[0][0]
                similarities.append((candidate, sim, candidate_idx))
            else:
                similarities.append((candidate, 0.0, -1))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 输出结果
        print("\n代词与候选对象的语义相似度:")
        for candidate, sim, idx in similarities:
            if idx != -1:
                print(f"  {candidate} (位置: {idx}): {sim:.4f}")
            else:
                print(f"  {candidate}: 未在句子中找到")

        print(f"\n最可能的指代对象: {similarities[0][0]} (相似度: {similarities[0][1]:.4f})")

    except Exception as e:
        print(f"处理过程中出错: {e}")


# 测试案例1：明确的指代关系
test_coreference_resolution(
    sentence="小明告诉小红他明天会来",
    pronoun="他",
    candidates=["小明", "小红"]
)

print("\n" + "-" * 80 + "\n")

# 测试案例2：基于上下文的指代
test_coreference_resolution(
    sentence="妈妈给小丽买了一本书，她非常喜欢它",
    pronoun="她",
    candidates=["妈妈", "小丽", "书"]
)

print("\n" + "-" * 80 + "\n")

# 测试案例3：更复杂的指代
test_coreference_resolution(
    sentence="张教授给学生们布置了论文，他们需要在周五前完成它",
    pronoun="他们",
    candidates=["张教授", "学生们", "论文"]
)
