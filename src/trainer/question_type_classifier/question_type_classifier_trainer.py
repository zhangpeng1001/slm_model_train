"""
问题类型分类训练器
基于BERT训练二分类模型：问题回答 vs 任务处理
用于识别用户的提问意图
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionTypeDataset(Dataset):
    """问题类型分类数据集"""

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


class QuestionTypeClassifierTrainer:
    """问题类型分类训练器"""

    def __init__(self, model_path,
                 save_dir=r'E:\project\python\slm_model_train\src\train_data\trained_question_type_classifier'):
        self.model_path = model_path
        self.save_dir = save_dir
        self.device = torch.device('cpu')  # 强制使用CPU

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 类别定义
        self.label_map = {
            0: "问题回答",  # 用户想要获取信息、知识或解答
            1: "任务处理"  # 用户想要执行某个具体操作或处理任务
        }

        # 初始化tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2  # 二分类
        )
        self.model.to(self.device)

        logger.info(f"问题类型分类器已加载到设备: {self.device}")

    def prepare_training_data(self):
        """准备训练数据"""

        # 问题回答类型（类别0）- 用户想要获取信息、知识或解答
        question_answer_samples = [
            "我有一批西安市地类图斑的数据，怎么进行发服务",
            "DEM高程数据怎么进行质量检查",
            "矢量地形图数据如何入库",
            "我的点云数据处理完了，如何发布服务",
            "地理信息数据库的数据怎么备份",
            "卫星遥感数据的预处理流程是什么",
            "无人机航拍数据如何进行几何纠正",
            "土地利用现状数据怎么更新",
            "基础地理信息数据如何管理",
            # 扩展更多问题回答类型的样本
            "遥感影像数据的配准方法有哪些",
            "如何选择合适的坐标系统",
            "数据质量检查的标准是什么",
            "什么是数据血缘关系",
            "元数据管理的最佳实践",
            "空间数据索引如何建立",
            "GIS数据格式转换工具推荐",
            "地理编码的准确性如何提高",
            "空间分析的常用方法",
            "地图投影变换的原理",
            "栅格数据重采样方法",
            "矢量数据拓扑检查规则",
            "数据库性能优化策略",
            "空间数据压缩技术",
            "地理信息系统架构设计",
            "数据安全防护措施",
            "版本控制如何实现",
            "数据标准化规范",
            "质量控制流程设计",
            "异常数据处理方法",
            "数据迁移注意事项",
            "备份恢复策略制定",
            "权限管理体系建设",
            "监控告警机制设置",
            "性能调优方案",
            "容灾备份方案",
            "数据治理框架",
            "合规性检查要求",
            "审计日志管理",
            "接口文档规范"
        ]

        # 任务处理类型（类别1）- 用户想要执行某个具体操作
        task_processing_samples = [
            "请帮我把实景三维模型成果数据进行治理",
            "我已经上传了单波段浮点投影的数据，现在想进行入库",
            "有一些遥感影像数据需要处理",
            "正射影像数据需要进行坐标转换",
            "激光雷达数据进行格式转换",
            "地籍调查数据需要标准化处理",
            "处理地下管线探测数据",
            "倾斜摄影测量数据进行建模",
            # 扩展更多任务处理类型的样本
            "帮我清洗这批地形数据",
            "需要对影像数据进行几何校正",
            "请处理这些GPS轨迹数据",
            "批量转换CAD文件格式",
            "执行数据质量检查任务",
            "启动数据同步作业",
            "运行地理编码程序",
            "执行空间分析计算",
            "进行数据备份操作",
            "开始数据迁移任务",
            "处理异常数据记录",
            "执行数据更新操作",
            "运行ETL数据处理流程",
            "启动定时调度任务",
            "执行数据压缩操作",
            "进行数据加密处理",
            "运行数据验证程序",
            "执行索引重建任务",
            "启动数据监控服务",
            "处理数据导入请求",
            "执行数据导出操作",
            "运行数据清理脚本",
            "启动服务发布流程",
            "执行权限配置更新",
            "进行系统配置修改",
            "运行性能优化程序",
            "执行日志清理任务",
            "启动健康检查服务",
            "处理告警信息",
            "执行故障恢复操作",
            "运行数据修复程序"
        ]

        # 组合数据
        texts = question_answer_samples + task_processing_samples
        labels = ([0] * len(question_answer_samples) +
                  [1] * len(task_processing_samples))

        logger.info(f"准备了 {len(texts)} 个训练样本")
        logger.info(f"问题回答类型: {len(question_answer_samples)} 个")
        logger.info(f"任务处理类型: {len(task_processing_samples)} 个")

        return texts, labels

    def create_dataloader(self, texts, labels, batch_size=8, shuffle=True):
        """创建数据加载器"""
        dataset = QuestionTypeDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # CPU训练时设为0
        )
        return dataloader

    def train(self, epochs=5, batch_size=8, learning_rate=2e-5):
        """训练分类器"""
        logger.info("开始训练问题类型分类器...")

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
        logger.info("开始评估问题类型分类器...")

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
        target_names = ['问题回答', '任务处理']
        report = classification_report(true_labels, predictions, target_names=target_names)
        logger.info(f"分类报告:\n{report}")

        # 打印混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        logger.info(f"混淆矩阵:\n{cm}")

        return accuracy

    def predict(self, text):
        """预测单个文本的类型"""
        self.model.eval()

        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_label].item()

        return {
            'text': text,
            'predicted_type': self.label_map[predicted_label],
            'confidence': confidence,
            'probabilities': {
                '问题回答': probabilities[0][0].item(),
                '任务处理': probabilities[0][1].item()
            }
        }

    def batch_predict(self, texts):
        """批量预测文本类型"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

    def save_model(self, model_name='question_type_classifier'):
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

        logger.info(f"问题类型分类器已保存到: {save_path}")
        return save_path

    def load_model(self, model_path):
        """加载已训练的模型"""
        logger.info(f"从 {model_path} 加载模型...")

        # 加载模型和tokenizer
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

        # 加载标签映射
        label_map_path = os.path.join(model_path, 'label_map.json')
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                loaded_label_map = json.load(f)
                # 确保键是整数类型
                self.label_map = {int(k): v for k, v in loaded_label_map.items()}
        else:
            # 如果没有找到label_map.json，使用默认映射
            logger.warning("未找到label_map.json，使用默认标签映射")
            self.label_map = {
                0: "问题回答",
                1: "任务处理"
            }

        logger.info("模型加载完成！")
        logger.info(f"标签映射: {self.label_map}")


def main():
    """主训练函数"""
    # BERT模型路径（请根据实际路径修改）
    model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = QuestionTypeClassifierTrainer(model_path)

    # 开始训练
    trainer.train(
        epochs=5,  # 二分类任务，适当增加训练轮次
        batch_size=4,  # CPU训练使用较小的batch size
        learning_rate=2e-5
    )

    # 评估模型
    texts, labels = trainer.prepare_training_data()
    trainer.evaluate(texts, labels)

    # 保存模型
    model_save_path = trainer.save_model()

    # 测试预测功能
    logger.info("\n=== 预测测试 ===")
    test_samples = [
        "我有一批西安市地类图斑的数据，怎么进行发服务",
        "请帮我把实景三维模型成果数据进行治理",
        "DEM高程数据怎么进行质量检查",
        "激光雷达数据进行格式转换",
        "矢量地形图数据如何入库",
        "处理地下管线探测数据"
    ]

    for sample in test_samples:
        result = trainer.predict(sample)
        logger.info(f"文本: {result['text']}")
        logger.info(f"预测类型: {result['predicted_type']}")
        logger.info(f"置信度: {result['confidence']:.4f}")
        logger.info(f"概率分布: {result['probabilities']}")
        logger.info("-" * 50)


if __name__ == "__main__":
    main()
