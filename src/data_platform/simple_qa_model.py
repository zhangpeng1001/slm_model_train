"""
基于BERT的问答模型
结合知识库和BERT模型进行事实性问答
"""
import numpy as np
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

from knowledge_base import DataPlatformKnowledgeBase


class SimpleDataPlatformQAModel:
    def __init__(self, model_path=None):
        """
        初始化问答模型
        """
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型路径
        if model_path is None:
            self.model_path = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
        else:
            self.model_path = model_path

        # 加载BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.bert_model = BertModel.from_pretrained(self.model_path).to(self.device)
        self.bert_model.eval()

        # 加载知识库
        self.knowledge_base = DataPlatformKnowledgeBase()

        # 预计算知识库中问题的向量表示
        self._precompute_question_embeddings()

    def _precompute_question_embeddings(self):
        """
        预计算知识库中所有问题的向量表示
        """
        self.question_embeddings = {}
        questions = list(self.knowledge_base.qa_knowledge.keys())

        for question in questions:
            embedding = self._get_text_embedding(question)
            self.question_embeddings[question] = embedding

    def _count_chinese_chars(self, text):
        """
        统计中文字符数量
        """
        return sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')

    def _has_business_keyword(self, text):
        """
        是否包含业务关键词（来自知识库的键）
        """
        return any(key in text for key in self.knowledge_base.qa_knowledge.keys())

    def _is_noise_query(self, text):
        """
        判定是否为无意义/噪声输入
        规则（尽量简单稳健，最小侵入）：
        - 过短文本
        - 仅由字母/数字/下划线/连字符/点组成（无中文且无业务关键词）
        - 高比例 ASCII 字母数字，且无中文且无业务关键词
        - 仅由标点组成
        """
        t = (text or "").strip()
        if not t:
            return True
        if len(t) < 2:
            return True

        chinese_count = self._count_chinese_chars(t)
        if chinese_count >= 2:
            return False  # 中文字符较多，视为可能的有效业务问题

        if self._has_business_keyword(t):
            return False  # 包含业务关键词

        # 仅标点
        if re.fullmatch(r'[\W_]+', t):
            return True

        # 仅 ASCII 字母数字常见符号
        if re.fullmatch(r'[A-Za-z0-9_\-\.]+', t):
            return True

        # 高比例 ASCII 字母数字
        ascii_alnum = sum(c.isascii() and c.isalnum() for c in t)
        if ascii_alnum / max(len(t), 1) > 0.8:
            return True

        return False

    def _normalize_question(self, text):
        """
        简单的中文问题归一化，去除客套/助词/标点，保留核心业务词
        例如："我想进行数据入库，该怎么做" -> "数据入库"
        """
        t = (text or "").strip()
        if not t:
            return t

        # 去除常见客套/助词/疑问短语
        stop_phrases = [
            "我想进行", "我想要", "我想", "请问", "麻烦", "想要",
            "该怎么做", "怎么做", "要怎么做", "如何", "怎样", "怎么办",
            "需要怎么", "应该怎么", "需要如何", "应该如何"
        ]
        for sp in stop_phrases:
            t = t.replace(sp, "")

        # 去除标点与空白
        t = re.sub(r"[，。！？、,.!?；;：:\s]+", "", t)

        return t

    def _get_text_embedding(self, text):
        """
        获取文本的BERT向量表示
        """
        # 编码文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 获取BERT输出
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # 使用[CLS]标记的向量作为文本表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding.flatten()

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """
        计算两个向量的余弦相似度
        """
        # 计算点积
        dot_product = np.dot(vec1, vec2)

        # 计算向量的模长
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # 避免除零错误
        if norm1 == 0 or norm2 == 0:
            return 0

        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def find_most_similar_question(self, user_question, threshold=0.8):
        """
        找到与用户问题最相似的知识库问题，同时返回第二相似度用于一致性检验
        """
        user_embedding = self._get_text_embedding(user_question)

        best_match = None
        best_sim = 0.0
        second_sim = 0.0

        for kb_question, kb_embedding in self.question_embeddings.items():
            similarity = cosine_similarity(user_embedding.reshape(1, -1), kb_embedding.reshape(1, -1))[0][0]
            if similarity > best_sim:
                second_sim = best_sim
                best_sim = similarity
                best_match = kb_question
            elif similarity > second_sim:
                second_sim = similarity

        if best_sim >= threshold:
            return best_match, best_sim, second_sim
        else:
            return None, best_sim, second_sim

    def answer_question(self, question):
        """
        回答用户问题
        """
        # 1) 输入噪声拦截：无意义输入直接拒答
        if self._is_noise_query(question):
            return {
                "answer": "抱歉，未能识别您的问题。请用数据平台相关的业务术语重新提问，例如：数据入库流程、任务调度、数据安全等。",
                "method": "invalid_input",
                "confidence": 0.0
            }

        # 2) 先尝试直接匹配（高置信度）
        direct_answer = self.knowledge_base.get_answer(question)
        if direct_answer != "抱歉，我无法回答这个问题。请尝试询问关于数据清洗、数据入库、数据处理、数据监控或数据安全相关的问题。":
            return {
                "answer": direct_answer,
                "method": "direct_match",
                "confidence": 1.0
            }

        # 2.1) 归一化后再匹配（处理“我想...该怎么做”类提问）
        norm_q = self._normalize_question(question)
        if norm_q != (question or ""):
            norm_answer = self.knowledge_base.get_answer(norm_q)
            if norm_answer != "抱歉，我无法回答这个问题。请尝试询问关于数据清洗、数据入库、数据处理、数据监控或数据安全相关的问题。":
                return {
                    "answer": norm_answer,
                    "method": "direct_match",
                    "confidence": 0.98
                }

        # 3) 语义相似度匹配：阈值根据输入特征动态调整
        chinese_count = self._count_chinese_chars(question or "")
        has_kw = self._has_business_keyword(question or "")

        # 基准阈值策略：
        # - 非中文且无业务关键词：严格（0.92）
        # - 无关键词但中文较少：偏严格（0.85）
        # - 正常情况：0.80
        if chinese_count < 2 and not has_kw:
            threshold = 0.92
        elif not has_kw:
            threshold = 0.85
        else:
            threshold = 0.80

        best_match, best_sim, second_sim = self.find_most_similar_question(question, threshold=threshold)

        # 4) Top-2 一致性检验：当分差很小且整体分不高，视为不稳定匹配，拒答
        if best_match:
            if (best_sim - second_sim) < 0.05 and best_sim < 0.90:
                return {
                    "answer": "抱歉，我无法回答这个问题。请尝试询问关于数据清洗、数据入库、数据处理、数据监控或数据安全相关的问题。",
                    "method": "no_match",
                    "confidence": 0.0
                }
            answer = self.knowledge_base.qa_knowledge[best_match]
            return {
                "answer": answer,
                "method": "semantic_match",
                "confidence": best_sim,
                "matched_question": best_match
            }
        else:
            return {
                "answer": "抱歉，我无法回答这个问题。请尝试询问关于数据清洗、数据入库、数据处理、数据监控或数据安全相关的问题。",
                "method": "no_match",
                "confidence": 0.0
            }

    def get_available_topics(self):
        """
        获取可回答的主题列表
        """
        return self.knowledge_base.get_all_topics()

    def add_knowledge(self, question, answer):
        """
        添加新的知识到知识库
        """
        self.knowledge_base.qa_knowledge[question] = answer
        # 重新计算向量表示
        embedding = self._get_text_embedding(question)
        self.question_embeddings[question] = embedding

        return f"已成功添加新知识：{question}"
