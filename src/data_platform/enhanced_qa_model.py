"""
增强版数据平台问答模型
支持加载训练后的模型进行更智能的问答
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel

from src.data_platform.knowledge_base import DataPlatformKnowledgeBase


class EnhancedBertQAModel(nn.Module):
    """增强版BERT问答模型"""

    def __init__(self, model_path, hidden_size=768, vocab_size=21128):
        super(EnhancedBertQAModel, self).__init__()

        # 加载BERT模型
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = hidden_size

        # 问题编码器
        self.question_encoder = self.bert

        # 答案生成层
        self.answer_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, vocab_size)
        )

        # 相似度计算层
        self.similarity_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, question_input_ids, question_attention_mask,
                answer_input_ids=None, answer_attention_mask=None):

        # 编码问题
        question_outputs = self.question_encoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask
        )
        question_embedding = question_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if answer_input_ids is not None:
            # 训练模式：计算问答相似度
            answer_outputs = self.question_encoder(
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask
            )
            answer_embedding = answer_outputs.last_hidden_state[:, 0, :]

            # 计算相似度
            combined = torch.cat([question_embedding, answer_embedding], dim=1)
            similarity = self.similarity_layer(combined)

            # 生成答案logits
            answer_logits = self.answer_generator(question_embedding)

            return {
                'question_embedding': question_embedding,
                'answer_embedding': answer_embedding,
                'similarity': similarity,
                'answer_logits': answer_logits
            }
        else:
            # 推理模式：只返回问题编码和生成的答案
            answer_logits = self.answer_generator(question_embedding)
            return {
                'question_embedding': question_embedding,
                'answer_logits': answer_logits
            }


class EnhancedDataPlatformQAModel:
    """增强版数据平台问答系统"""

    def __init__(self, base_model_path=None, trained_model_path=None):
        """
        初始化增强版问答模型
        
        Args:
            base_model_path: 基础BERT模型路径
            trained_model_path: 训练后的模型路径
        """
        # 设备配置
        self.device = torch.device("cpu")

        # 设置模型路径
        if base_model_path is None:
            self.base_model_path = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
        else:
            self.base_model_path = base_model_path

        self.trained_model_path = trained_model_path

        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.base_model_path)

        # 加载知识库
        self.knowledge_base = DataPlatformKnowledgeBase()

        # 初始化模型
        self._load_model()

        # 预计算知识库问题的向量表示
        self._precompute_knowledge_embeddings()

    def _load_model(self):
        """加载模型"""
        try:
            if self.trained_model_path and os.path.exists(self.trained_model_path):
                # 加载训练后的模型
                print(f"加载训练后的模型: {self.trained_model_path}")

                # 读取配置
                config_path = os.path.join(self.trained_model_path, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    self.model = EnhancedBertQAModel(
                        self.base_model_path,
                        hidden_size=config.get('hidden_size', 768),
                        vocab_size=config.get('vocab_size', 21128)
                    )
                else:
                    self.model = EnhancedBertQAModel(self.base_model_path)

                # 加载模型权重
                model_state_path = os.path.join(self.trained_model_path, 'pytorch_model.bin')
                if os.path.exists(model_state_path):
                    state_dict = torch.load(model_state_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print("✓ 训练后的模型加载成功")
                    self.is_trained_model = True
                else:
                    raise FileNotFoundError("模型权重文件不存在")
            else:
                # 使用基础BERT模型
                print("使用基础BERT模型")
                self.model = EnhancedBertQAModel(self.base_model_path)
                self.is_trained_model = False

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"模型加载失败: {e}")
            print("回退到基础知识库模式")
            self.model = None
            self.is_trained_model = False

    def _precompute_knowledge_embeddings(self):
        """预计算知识库中问题的向量表示"""
        if self.model is None:
            return

        self.question_embeddings = {}
        questions = list(self.knowledge_base.qa_knowledge.keys())

        print("预计算知识库问题向量...")
        for question in questions:
            embedding = self._get_text_embedding(question)
            self.question_embeddings[question] = embedding
        print("✓ 知识库向量预计算完成")

    def _get_text_embedding(self, text):
        """获取文本的向量表示"""
        if self.model is None:
            return None

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

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(
                question_input_ids=inputs['input_ids'],
                question_attention_mask=inputs['attention_mask']
            )
            embedding = outputs['question_embedding'].cpu().numpy()

        return embedding.flatten()

    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0

        # 计算点积
        dot_product = np.dot(vec1, vec2)

        # 计算向量的模长
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # 避免除零错误
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def find_most_similar_question(self, user_question, threshold=0.7):
        """找到与用户问题最相似的知识库问题"""
        if self.model is None:
            return None, 0.0

        user_embedding = self._get_text_embedding(user_question)
        if user_embedding is None:
            return None, 0.0

        max_similarity = 0
        best_match = None

        for kb_question, kb_embedding in self.question_embeddings.items():
            if kb_embedding is None:
                continue

            similarity = self._cosine_similarity(user_embedding, kb_embedding)

            if similarity > max_similarity and similarity > threshold:
                max_similarity = similarity
                best_match = kb_question

        return best_match, max_similarity

    def generate_answer(self, question):
        """使用训练后的模型生成答案"""
        if self.model is None or not self.is_trained_model:
            return None

        try:
            # 编码问题
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 生成答案
            with torch.no_grad():
                outputs = self.model(
                    question_input_ids=inputs['input_ids'],
                    question_attention_mask=inputs['attention_mask']
                )

                answer_logits = outputs['answer_logits']

                # 获取最可能的token
                predicted_token_id = torch.argmax(answer_logits, dim=-1)[0]

                # 解码为文本（单个token）
                generated_text = self.tokenizer.decode(
                    [predicted_token_id.item()],
                    skip_special_tokens=True
                )

                return generated_text

        except Exception as e:
            print(f"答案生成失败: {e}")
            return None

    def answer_question(self, question):
        """回答用户问题"""
        # 首先尝试直接匹配
        # direct_answer = self.knowledge_base.get_answer(question)
        # if direct_answer != "抱歉，我无法回答这个问题。请尝试询问关于数据清洗、数据入库、数据处理、数据监控或数据安全相关的问题。":
        #     return {
        #         "answer": direct_answer,
        #         "method": "direct_match",
        #         "confidence": 1.0,
        #         "source": "knowledge_base"
        #     }

        # 如果有训练后的模型，尝试生成答案
        if self.is_trained_model:
            generated_answer = self.generate_answer(question)
            print("generated_answer:", generated_answer)
            if generated_answer and len(generated_answer.strip()) > 10:
                return {
                    "answer": generated_answer,
                    "method": "generation",
                    "confidence": 0.8,
                    "source": "trained_model"
                }

        # 使用语义相似度匹配
        best_match, similarity = self.find_most_similar_question(question)

        if best_match:
            answer = self.knowledge_base.qa_knowledge[best_match]
            return {
                "answer": answer,
                "method": "semantic_match",
                "confidence": similarity,
                "matched_question": best_match,
                "source": "knowledge_base"
            }
        else:
            return {
                "answer": "抱歉，我无法回答这个问题。请尝试询问关于数据清洗、数据入库、数据处理、数据监控或数据安全相关的问题。",
                "method": "no_match",
                "confidence": 0.0,
                "source": "fallback"
            }

    def get_available_topics(self):
        """获取可回答的主题列表"""
        return self.knowledge_base.get_all_topics()

    def add_knowledge(self, question, answer):
        """添加新的知识到知识库"""
        self.knowledge_base.qa_knowledge[question] = answer

        # 重新计算向量表示
        if self.model is not None:
            embedding = self._get_text_embedding(question)
            self.question_embeddings[question] = embedding

        return f"已成功添加新知识：{question}"

    def get_model_info(self):
        """获取模型信息"""
        return {
            "base_model_path": self.base_model_path,
            "trained_model_path": self.trained_model_path,
            "is_trained_model": self.is_trained_model,
            "device": str(self.device),
            "knowledge_base_size": len(self.knowledge_base.qa_knowledge)
        }
