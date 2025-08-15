"""
测试 bert-base-chinese 模型的基础能力

"""

"""

中文语义理解能力：
字符级别的上下文语义编码
一词多义的语境区分（如 "苹果" 在 "吃苹果" 和 "苹果手机" 中的不同含义）
中文语法结构理解
文本表示能力：
将中文文本转换为固定维度的向量表示
捕捉句子间的语义关联和相似度
下游任务适配能力：
可通过微调适应多种中文 NLP 任务：
文本分类（情感分析、垃圾邮件识别等）
命名实体识别（识别人名、地名、机构名等）
问答系统
文本相似度计算
词性标注
句法分析

更全面的能力测试：
1.** 使用下游任务评估 **：

在公开中文数据集上进行微调测试（如 CLUE 基准）
评估在分类、NER 等任务上的表现

2.** 上下文理解测试 **：

设计包含指代关系的句子（如 "小明告诉小红他明天会来"）
检查模型是否能正确捕捉指代关系

3.** 零样本测试 **：

不进行微调，直接使用预训练模型的输出进行简单任务
观察其在语义检索等任务上的表现

BERT 的优势在于其强大的预训练上下文理解能力，通过这些测试可以初步了解其对中文语言的理解程度。对于特定任务，通常还需要进行微调以获得最佳性能。
"""

"""
BertModel是 "基础引擎"，BertForXXX是 "带任务配件的引擎"，后者在前者基础上针对特定任务做了封装，更便于直接使用
在 Hugging Face 的transformers库中，BertModel与BertForSequenceClassification（以及其他BertForXXX类）的核心区别在于用途和结构设计：
1. 核心区别：基础模型 vs 任务特定模型
BertModel：
这是 BERT 的基础预训练模型，仅包含 BERT 的主体结构（多层 Transformer 编码器），输出的是原始的隐藏状态（hidden states）。
它不包含针对特定任务的输出层，主要用于：
提取文本的上下文特征（如句子 / 词向量）
作为自定义任务模型的基础组件（需手动添加输出层）
用于特征工程或迁移学习的底层特征提取
输出示例：
last_hidden_state：所有 token 的最终隐藏状态（形状为[batch_size, seq_len, hidden_size]）
pooler_output：[CLS] token 经过池化后的输出（可粗略作为句子表示）
BertForSequenceClassification：
这是针对文本分类任务的专用模型，在BertModel的基础上额外添加了一个分类头（全连接层 + Softmax / 激活函数）。
它的设计目标是直接用于文本分类任务（如情感分析、垃圾邮件识别等），输出的是每个类别的预测概率。
输出示例：
logits：每个类别的原始预测分数（形状为[batch_size, num_labels]），可通过softmax转换为概率
2. 其他常见的BertForXXX任务特定模型
除了文本分类，transformers库还为不同 NLP 任务提供了预定义的 BERT 变体，它们都在基础模型上添加了对应的任务输出层：

模型类名	用途	典型输出
BertForQuestionAnswering	问答任务（抽取式）	答案的起始 / 结束位置概率
BertForTokenClassification	token 级别分类（如 NER、词性标注）	每个 token 的类别概率
BertForNextSentencePrediction	下一句预测（BERT 预训练任务之一）	两句话是否连续的概率
BertForMaskedLM	掩码语言模型（BERT 预训练任务之一）	被掩码 token 的预测概率
BertForSequenceClassification	句子级分类（如情感分析）	句子的类别概率
BertForMultipleChoice	多选任务（如阅读理解中的选项选择）	每个选项的概率
3. 如何选择？
若需提取文本特征（如计算句子相似度、作为其他模型的输入），用BertModel。
若需直接解决特定任务（如分类、NER、问答），用对应的BertForXXX，它们已经集成了任务所需的输出层，可直接微调。
"""
