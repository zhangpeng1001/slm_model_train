"""
问题分类器训练器
基于BERT训练三分类模型：数据平台相关、通用对话、无关问题
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionDataset(Dataset):
    """问题分类数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class QuestionClassifierTrainer:
    """问题分类器训练器"""

    def __init__(self, model_path, save_dir='./trained_classifiers'):
        self.model_path = model_path
        self.save_dir = save_dir
        self.device = torch.device('cpu')  # 强制使用CPU

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 类别定义
        self.label_map = {
            0: "data_platform",  # 数据平台相关
            1: "general_chat",  # 通用对话
            2: "irrelevant"  # 无关问题
        }

        # 初始化tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3
        )
        self.model.to(self.device)

        logger.info(f"分类器已加载到设备: {self.device}")

    def prepare_training_data(self):
        """准备训练数据"""

        # 数据平台相关问题（类别0）
        data_platform_questions = [
            "数据清洗流程是什么？",
            "如何进行数据清洗？",
            "数据清洗步骤有哪些？",
            "数据清洗的流程",
            "怎么清洗数据？",
            "数据入库流程",
            "如何进行数据入库？",
            "数据入库的步骤",
            "怎样入库数据？",
            "数据入库操作",
            "数据质量检查",
            "如何检查数据质量？",
            "数据质量检查方法",
            "数据质量如何保证？",
            "质量检查步骤",
            "数据监控",
            "如何监控数据？",
            "数据监控方法",
            "数据监控怎么做？",
            "监控数据流程",
            "数据安全",
            "如何保证数据安全？",
            "数据安全措施",
            "数据安全怎么做？",
            "数据加密方法",
            "数据预处理",
            "数据转换",
            "数据标准化",
            "数据备份",
            "权限管理",
            "任务调度",
            "异常处理",
            "数据库连接",
            "数据处理流程",
            "ETL流程",
            "数据仓库",
            "数据平台架构",
            "数据治理",
            "数据血缘",
            "元数据管理"
        ]

        # 通用对话（类别1）
        general_chat_questions = [
            "你好",
            "您好",
            "hi",
            "hello",
            "早上好",
            "下午好",
            "晚上好",
            "谢谢",
            "谢谢你",
            "感谢",
            "不客气",
            "再见",
            "拜拜",
            "bye",
            "好的",
            "知道了",
            "明白了",
            "收到",
            "OK",
            "可以",
            "没问题",
            "帮忙",
            "请问",
            "能否",
            "麻烦",
            "打扰了",
            "不好意思",
            "对不起",
            "抱歉"
        ]

        # 无关问题（类别2）
        irrelevant_questions = [
            "今天天气怎么样？",
            "北京有什么好吃的？",
            "如何学习英语？",
            "Python怎么安装？",
            "什么是机器学习？",
            "股票今天涨了吗？",
            "电影推荐一下",
            "音乐好听吗？",
            "游戏攻略",
            "旅游景点推荐",
            "美食制作方法",
            "健身计划",
            "减肥方法",
            "护肤品推荐",
            "汽车保养",
            "房价走势",
            "教育政策",
            "医疗保险",
            "法律咨询",
            "心理健康",
            "what is your name?",
            "how are you?",
            "where are you from?",
            "tell me a joke",
            "sing a song",
            "write a poem",
            "calculate 1+1",
            "what time is it?",
            "random text here",
            "asdfghjkl",
            "123456789",
            "测试文本",
            "随机输入",
            "无意义内容",
            "乱七八糟",
            "胡言乱语",
            "不知所云",
            "莫名其妙",
            "奇怪问题",
            "无关内容"
        ]

        # 组合数据
        texts = data_platform_questions + general_chat_questions + irrelevant_questions
        labels = ([0] * len(data_platform_questions) +
                  [1] * len(general_chat_questions) +
                  [2] * len(irrelevant_questions))

        logger.info(f"准备了 {len(texts)} 个训练样本")
        logger.info(f"数据平台相关: {len(data_platform_questions)} 个")
        logger.info(f"通用对话: {len(general_chat_questions)} 个")
        logger.info(f"无关问题: {len(irrelevant_questions)} 个")

        return texts, labels

    def create_dataloader(self, texts, labels, batch_size=8, shuffle=True):
        """创建数据加载器"""
        dataset = QuestionDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # CPU训练时设为0
        )
        return dataloader

    def train(self, epochs=5, batch_size=8, learning_rate=2e-5):
        """训练分类器"""
        logger.info("开始训练分类器...")

        # 准备训练数据
        texts, labels = self.prepare_training_data()

        # 创建数据加载器
        train_dataloader = self.create_dataloader(texts, labels, batch_size, shuffle=True)

        # 设置优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # 设置学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch in progress_bar:
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

        logger.info("训练完成！")

    def evaluate(self, texts, labels):
        """评估分类器"""
        logger.info("开始评估分类器...")

        eval_dataloader = self.create_dataloader(texts, labels, batch_size=8, shuffle=False)

        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 获取预测结果
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1)

                predictions.extend(predicted_labels.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # 计算准确率
        accuracy = accuracy_score(true_labels, predictions)
        logger.info(f"准确率: {accuracy:.4f}")

        # 打印分类报告
        target_names = ['数据平台相关', '通用对话', '无关问题']
        report = classification_report(true_labels, predictions, target_names=target_names)
        logger.info(f"分类报告:\n{report}")

        return accuracy

    def save_model(self, model_name='question_classifier'):
        """保存分类器模型"""
        save_path = os.path.join(self.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        # 保存模型和tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # 保存标签映射
        label_map_path = os.path.join(save_path, 'label_map.json')
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        logger.info(f"分类器已保存到: {save_path}")
        return save_path


def main():
    """主训练函数"""
    # 模型路径
    model_path = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = QuestionClassifierTrainer(model_path)

    # 开始训练
    trainer.train(
        epochs=3,  # 分类任务通常不需要太多轮次
        batch_size=4,  # CPU训练使用较小的batch size
        learning_rate=2e-5
    )

    # 评估模型
    texts, labels = trainer.prepare_training_data()
    trainer.evaluate(texts, labels)

    # 保存模型
    trainer.save_model('question_classifier')


if __name__ == "__main__":
    main()
