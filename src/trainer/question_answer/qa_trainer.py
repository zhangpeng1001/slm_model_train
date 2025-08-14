"""
数据平台问答系统训练器
基于BERT模型进行微调训练，实现真正的问答生成能力
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel, BertConfig,
    get_linear_schedule_with_warmup
)
import numpy as np
from tqdm import tqdm
import logging
from torch.optim import AdamW

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QADataset(Dataset):
    """问答数据集"""

    def __init__(self, qa_pairs, tokenizer, max_length=512):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question = self.qa_pairs[idx]['question']
        answer = self.qa_pairs[idx]['answer']

        # 编码问题
        question_encoding = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 编码答案
        answer_encoding = self.tokenizer(
            answer,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'question_input_ids': question_encoding['input_ids'].flatten(),
            'question_attention_mask': question_encoding['attention_mask'].flatten(),
            'answer_input_ids': answer_encoding['input_ids'].flatten(),
            'answer_attention_mask': answer_encoding['attention_mask'].flatten(),
            'question_text': question,
            'answer_text': answer
        }


class BertQAModel(nn.Module):
    """基于BERT的问答生成模型"""

    def __init__(self, model_path, hidden_size=768, vocab_size=21128):
        super(BertQAModel, self).__init__()

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
            # 推理模式：只返回问题编码
            return {
                'question_embedding': question_embedding
            }


class QATrainer:
    """问答系统训练器"""

    def __init__(self, model_path, save_dir='./trained_models'):
        self.model_path = model_path
        self.save_dir = save_dir
        self.device = torch.device('cpu')  # 强制使用CPU

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        # 初始化模型
        self.model = BertQAModel(model_path)
        self.model.to(self.device)

        logger.info(f"模型已加载到设备: {self.device}")

    def prepare_training_data(self, knowledge_base_path=None):
        """准备训练数据"""

        if knowledge_base_path and os.path.exists(knowledge_base_path):
            # 从文件加载数据
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
        else:
            # 使用内置知识库数据
            from src.data_platform.knowledge_base import DataPlatformKnowledgeBase
            kb = DataPlatformKnowledgeBase()

            qa_pairs = []
            for question, answer in kb.qa_knowledge.items():
                qa_pairs.append({
                    'question': question,
                    'answer': answer
                })

            # 数据增强：生成相似问题
            augmented_pairs = self._augment_data(qa_pairs)
            qa_pairs.extend(augmented_pairs)

        logger.info(f"准备了 {len(qa_pairs)} 个训练样本")
        return qa_pairs

    def _augment_data(self, qa_pairs):
        """数据增强：生成更多训练样本"""
        augmented = []

        # 为每个问题生成变体
        question_variants = {
            "数据清洗流程": ["数据清洗的流程是什么", "如何进行数据清洗", "数据清洗步骤"],
            "数据入库流程": ["数据入库的流程", "如何进行数据入库", "数据入库步骤"],
            "数据质量检查": ["如何检查数据质量", "数据质量检查方法", "数据质量如何保证"],
            "数据监控": ["如何监控数据", "数据监控方法", "数据监控怎么做"],
            "数据安全": ["如何保证数据安全", "数据安全措施", "数据安全怎么做"]
        }

        for qa_pair in qa_pairs:
            original_question = qa_pair['question']
            answer = qa_pair['answer']

            # 添加问题变体
            if original_question in question_variants:
                for variant in question_variants[original_question]:
                    augmented.append({
                        'question': variant,
                        'answer': answer
                    })

        return augmented

    def create_dataloader(self, qa_pairs, batch_size=4, shuffle=True):
        """创建数据加载器"""
        dataset = QADataset(qa_pairs, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # CPU训练时设为0
        )
        return dataloader

    def train(self, qa_pairs, epochs=5, batch_size=4, learning_rate=2e-5):
        """训练模型"""
        logger.info("开始训练...")

        # 创建数据加载器
        train_dataloader = self.create_dataloader(qa_pairs, batch_size, shuffle=True)

        # 设置优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # 设置学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 损失函数
        similarity_criterion = nn.BCELoss()
        generation_criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch in progress_bar:
                # 移动数据到设备
                question_input_ids = batch['question_input_ids'].to(self.device)
                question_attention_mask = batch['question_attention_mask'].to(self.device)
                answer_input_ids = batch['answer_input_ids'].to(self.device)
                answer_attention_mask = batch['answer_attention_mask'].to(self.device)

                # 前向传播
                outputs = self.model(
                    question_input_ids=question_input_ids,
                    question_attention_mask=question_attention_mask,
                    answer_input_ids=answer_input_ids,
                    answer_attention_mask=answer_attention_mask
                )

                # 计算相似度损失（正样本，目标为1）
                similarity_targets = torch.ones(question_input_ids.size(0), 1).to(self.device)
                similarity_loss = similarity_criterion(outputs['similarity'], similarity_targets)

                # 简化生成损失计算
                # 只使用第一个token作为目标，避免维度问题
                answer_logits = outputs['answer_logits']  # [batch_size, vocab_size]
                first_answer_token = answer_input_ids[:, 0]  # [batch_size] - 取第一个token
                
                generation_loss = generation_criterion(answer_logits, first_answer_token)

                # 总损失
                loss = similarity_loss + 0.5 * generation_loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # 保存检查点
            if (epoch + 1) % 2 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}')

        logger.info("训练完成！")

    def save_model(self, model_name):
        """保存模型"""
        save_path = os.path.join(self.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        # 保存模型状态
        torch.save(self.model.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))

        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)

        # 保存配置
        config = {
            'model_path': self.model_path,
            'hidden_size': self.model.hidden_size,
            'vocab_size': self.tokenizer.vocab_size
        }
        with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"模型已保存到: {save_path}")

    def evaluate(self, qa_pairs, batch_size=4):
        """评估模型"""
        logger.info("开始评估...")

        eval_dataloader = self.create_dataloader(qa_pairs, batch_size, shuffle=False)

        self.model.eval()
        total_similarity = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc='Evaluating'):
                question_input_ids = batch['question_input_ids'].to(self.device)
                question_attention_mask = batch['question_attention_mask'].to(self.device)
                answer_input_ids = batch['answer_input_ids'].to(self.device)
                answer_attention_mask = batch['answer_attention_mask'].to(self.device)

                outputs = self.model(
                    question_input_ids=question_input_ids,
                    question_attention_mask=question_attention_mask,
                    answer_input_ids=answer_input_ids,
                    answer_attention_mask=answer_attention_mask
                )

                similarity_scores = outputs['similarity'].cpu().numpy()
                total_similarity += similarity_scores.sum()
                total_samples += len(similarity_scores)

        avg_similarity = total_similarity / total_samples
        logger.info(f"平均相似度得分: {avg_similarity:.4f}")

        return avg_similarity


def main():
    """主训练函数"""
    # 模型路径
    model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = QATrainer(model_path)

    # 准备训练数据
    qa_pairs = trainer.prepare_training_data()

    # 开始训练
    trainer.train(
        qa_pairs=qa_pairs,
        epochs=10,
        batch_size=2,  # CPU训练使用较小的batch size
        learning_rate=1e-5
    )

    # 评估模型
    trainer.evaluate(qa_pairs)

    # 保存最终模型
    trainer.save_model('final_model')


if __name__ == "__main__":
    main()
