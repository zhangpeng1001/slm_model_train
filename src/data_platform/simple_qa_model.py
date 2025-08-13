"""
简化版问答模型
不依赖sklearn，使用基础的文本匹配和BERT语义理解
"""
import torch
from transformers import BertTokenizer, BertModel
from knowledge_base import DataPlatformKnowledgeBase
import numpy as np
import math

class SimpleDataPlatformQAModel:
    def __init__(self, model_path=None):
        """
        初始化简化版问答模型
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
    
    def _cosine_similarity(self, vec1, vec2):
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
    
    def find_most_similar_question(self, user_question, threshold=0.7):
        """
        找到与用户问题最相似的知识库问题
        """
        user_embedding = self._get_text_embedding(user_question)
        
        max_similarity = 0
        best_match = None
        
        for kb_question, kb_embedding in self.question_embeddings.items():
            # 计算余弦相似度
            similarity = self._cosine_similarity(user_embedding, kb_embedding)
            
            if similarity > max_similarity and similarity > threshold:
                max_similarity = similarity
                best_match = kb_question
        
        return best_match, max_similarity
    
    def answer_question(self, question):
        """
        回答用户问题
        """
        # 首先尝试直接匹配
        direct_answer = self.knowledge_base.get_answer(question)
        if direct_answer != "抱歉，我无法回答这个问题。请尝试询问关于数据清洗、数据入库、数据处理、数据监控或数据安全相关的问题。":
            return {
                "answer": direct_answer,
                "method": "direct_match",
                "confidence": 1.0
            }
        
        # 如果直接匹配失败，使用语义相似度匹配
        best_match, similarity = self.find_most_similar_question(question)
        
        if best_match:
            answer = self.knowledge_base.qa_knowledge[best_match]
            return {
                "answer": answer,
                "method": "semantic_match",
                "confidence": similarity,
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
