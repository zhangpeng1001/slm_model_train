"""
数据提取训练器
基于BERT模型进行命名实体识别(NER)训练，从用户问题中提取数据类型信息
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
from sklearn.metrics import classification_report
import re

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractionDataset(Dataset):
    """数据提取数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_sequence = self.labels[idx]

        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 处理标签序列，确保与token对齐
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # 创建标签序列，与tokenized的长度一致
        labels = torch.zeros(self.max_length, dtype=torch.long)
        
        # 简化标签对齐：将字符级别标签映射到token级别
        # 这里使用简化的方法，将实体标签映射到对应的token位置
        tokens = self.tokenizer.tokenize(text)
        
        # 创建字符到token的映射
        char_to_token = self._create_char_to_token_mapping(text, tokens)
        
        # 将字符级别的标签转换为token级别
        for char_idx, label in enumerate(label_sequence):
            if char_idx < len(char_to_token):
                token_idx = char_to_token[char_idx]
                if token_idx is not None and token_idx < self.max_length:
                    labels[token_idx] = label

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': text
        }
    
    def _create_char_to_token_mapping(self, text, tokens):
        """创建字符到token的映射"""
        char_to_token = [None] * len(text)
        char_idx = 0
        
        for token_idx, token in enumerate(tokens):
            # 处理子词标记
            token_text = token.replace('##', '')
            
            # 在文本中查找token对应的字符位置
            if char_idx < len(text):
                # 寻找token在剩余文本中的位置
                remaining_text = text[char_idx:]
                token_pos = remaining_text.find(token_text)
                
                if token_pos != -1:
                    # 映射字符到token索引（+1是因为[CLS] token）
                    start_pos = char_idx + token_pos
                    end_pos = start_pos + len(token_text)
                    
                    for i in range(start_pos, min(end_pos, len(char_to_token))):
                        char_to_token[i] = token_idx + 1  # +1 for [CLS]
                    
                    char_idx = end_pos
                else:
                    # 如果找不到，跳过这个token
                    char_idx += 1
        
        return char_to_token


class BertDataExtractionModel(nn.Module):
    """基于BERT的数据提取模型"""

    def __init__(self, model_path, num_labels=3):
        super(BertDataExtractionModel, self).__init__()
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        
        # 分类层：B-DATA(开始), I-DATA(内部), O(其他)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # 分类
        logits = self.classifier(sequence_output)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 只计算有效位置的损失
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs['loss'] = loss
            
        return outputs


class DataExtractionTrainer:
    """数据提取训练器"""

    def __init__(self, model_path, save_dir='./trained_data_extractors'):
        self.model_path = model_path
        self.save_dir = save_dir
        self.device = torch.device('cpu')  # 强制使用CPU

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 标签定义：BIO标注
        self.label_map = {
            0: "O",      # 其他
            1: "B-DATA", # 数据类型开始
            2: "I-DATA"  # 数据类型内部
        }
        
        self.id2label = {v: k for k, v in enumerate(self.label_map.values())}

        # 初始化tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertDataExtractionModel(model_path, num_labels=len(self.label_map))
        self.model.to(self.device)

        logger.info(f"数据提取模型已加载到设备: {self.device}")

    def prepare_training_data(self):
        """准备训练数据"""
        
        # 训练样本：(问题文本, 提取目标)
        training_samples = [
            ("我有一批西安市地类图斑的数据，怎么进行发服务", "西安市地类图斑"),
            ("请帮我把实景三维模型成果数据进行治理", "实景三维模型成果"),
            ("我已经上传了单波段浮点投影的数据，现在想进行入库", "单波段浮点投影"),
            ("有一些遥感影像数据需要处理", "遥感影像"),
            ("DEM高程数据怎么进行质量检查", "DEM高程"),
            ("矢量地形图数据如何入库", "矢量地形图"),
            ("我的点云数据处理完了，如何发布服务", "点云"),
            ("正射影像数据需要进行坐标转换", "正射影像"),
            ("地理信息数据库的数据怎么备份", "地理信息数据库"),
            ("卫星遥感数据的预处理流程是什么", "卫星遥感"),
            ("无人机航拍数据如何进行几何纠正", "无人机航拍"),
            ("激光雷达数据的格式转换", "激光雷达"),
            ("地籍调查数据需要标准化处理", "地籍调查"),
            ("土地利用现状数据怎么更新", "土地利用现状"),
            ("基础地理信息数据如何管理", "基础地理信息"),
            ("城市三维建模数据的质量控制", "城市三维建模"),
            ("地下管线探测数据处理", "地下管线探测"),
            ("倾斜摄影测量数据建模", "倾斜摄影测量"),
            ("地质勘探数据入库流程", "地质勘探"),
            ("水文监测数据的时序分析", "水文监测"),
            ("气象观测数据处理", "气象观测"),
            ("土壤调查数据分析", "土壤调查"),
            ("森林资源调查数据", "森林资源调查"),
            ("海洋测绘数据处理", "海洋测绘"),
            ("地震监测数据分析", "地震监测"),
            ("环境监测数据入库", "环境监测"),
            ("交通流量数据统计", "交通流量"),
            ("人口普查数据处理", "人口普查"),
            ("经济统计数据分析", "经济统计"),
            ("建筑信息模型数据", "建筑信息模型"),
            ("管网拓扑数据维护", "管网拓扑"),
            ("电力设施数据更新", "电力设施"),
            ("通信基站数据管理", "通信基站"),
            ("道路网络数据处理", "道路网络"),
            ("地名地址数据标准化", "地名地址"),
            ("行政区划数据更新", "行政区划"),
            ("地价监测数据分析", "地价监测"),
            ("规划用地数据管理", "规划用地")
        ]
        
        texts = []
        labels = []
        
        for text, target in training_samples:
            # 为每个样本创建BIO标注
            text_labels = self._create_bio_labels(text, target)
            texts.append(text)
            labels.append(text_labels)
            
            # 数据增强：创建变体
            augmented_samples = self._augment_sample(text, target)
            for aug_text, aug_target in augmented_samples:
                aug_labels = self._create_bio_labels(aug_text, aug_target)
                texts.append(aug_text)
                labels.append(aug_labels)

        logger.info(f"准备了 {len(texts)} 个训练样本")
        return texts, labels

    def _create_bio_labels(self, text, target):
        """创建BIO标注序列"""
        labels = [0] * len(text)  # 默认为O标签
        
        # 查找目标实体在文本中的位置
        target_start = text.find(target)
        if target_start != -1:
            target_end = target_start + len(target)
            
            # 标注第一个字符为B-DATA
            labels[target_start] = 1
            
            # 标注后续字符为I-DATA
            for i in range(target_start + 1, target_end):
                labels[i] = 2
                
        return labels

    def _augment_sample(self, text, target):
        """数据增强：生成样本变体"""
        augmented = []
        
        # 问题模板变体
        question_templates = [
            f"如何处理{target}数据？",
            f"{target}数据怎么入库？",
            f"我需要对{target}数据进行处理",
            f"{target}数据的质量检查怎么做？",
            f"请帮我处理{target}数据",
            f"{target}数据如何发布服务？",
            f"我有{target}数据需要治理",
            f"{target}数据的标准化流程",
            f"如何管理{target}数据？",
            f"{target}数据处理完成后如何使用？"
        ]
        
        for template in question_templates:
            augmented.append((template, target))
            
        return augmented

    def create_dataloader(self, texts, labels, batch_size=4, shuffle=True):
        """创建数据加载器"""
        dataset = DataExtractionDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # CPU训练时设为0
        )
        return dataloader

    def train(self, epochs=5, batch_size=4, learning_rate=2e-5):
        """训练数据提取模型"""
        logger.info("开始训练数据提取模型...")

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
                labels = batch['labels'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs['loss']

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # 每2个epoch保存一次检查点
            if (epoch + 1) % 2 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}')

        logger.info("训练完成！")

    def evaluate(self, texts, labels):
        """评估数据提取模型"""
        logger.info("开始评估数据提取模型...")

        eval_dataloader = self.create_dataloader(texts, labels, batch_size=4, shuffle=False)

        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 获取预测结果
                logits = outputs['logits']
                predicted_labels = torch.argmax(logits, dim=2)

                # 只考虑有效位置的预测
                for i in range(input_ids.size(0)):
                    valid_length = attention_mask[i].sum().item()
                    pred_seq = predicted_labels[i][:valid_length].cpu().numpy()
                    true_seq = labels[i][:valid_length].cpu().numpy()
                    
                    predictions.extend(pred_seq)
                    true_labels.extend(true_seq)

        # 计算分类报告
        target_names = ['O', 'B-DATA', 'I-DATA']
        report = classification_report(true_labels, predictions, target_names=target_names, zero_division=0)
        logger.info(f"分类报告:\n{report}")

        # 计算实体级别的准确率
        entity_accuracy = self._calculate_entity_accuracy(texts, labels)
        logger.info(f"实体提取准确率: {entity_accuracy:.4f}")

        return entity_accuracy

    def _calculate_entity_accuracy(self, texts, labels):
        """计算实体级别的准确率"""
        correct_entities = 0
        total_entities = 0
        
        for text, label_seq in zip(texts, labels):
            # 提取真实实体
            true_entities = self._extract_entities_from_labels(text, label_seq)
            
            # 预测实体
            pred_entities = self.predict_entities(text)
            
            total_entities += len(true_entities)
            for entity in true_entities:
                if entity in pred_entities:
                    correct_entities += 1
                    
        if total_entities == 0:
            return 0.0
            
        return correct_entities / total_entities

    def _extract_entities_from_labels(self, text, labels):
        """从标签序列中提取实体"""
        entities = []
        current_entity = ""
        in_entity = False
        
        for i, label in enumerate(labels):
            if i >= len(text):
                break
                
            if label == 1:  # B-DATA
                if in_entity and current_entity:
                    entities.append(current_entity)
                current_entity = text[i]
                in_entity = True
            elif label == 2 and in_entity:  # I-DATA
                current_entity += text[i]
            else:  # O
                if in_entity and current_entity:
                    entities.append(current_entity)
                current_entity = ""
                in_entity = False
                
        if in_entity and current_entity:
            entities.append(current_entity)
            
        return entities

    def predict_entities(self, text):
        """预测文本中的实体"""
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
            
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=2)
            
        # 转换预测结果为实体
        pred_labels = predictions[0].cpu().numpy()
        valid_length = attention_mask[0].sum().item()
        
        # 将token级别的预测转换为字符级别
        char_labels = [0] * len(text)
        tokens = self.tokenizer.tokenize(text)
        
        # 简化处理：直接映射前面的字符
        token_idx = 1  # 跳过[CLS]
        char_idx = 0
        
        for token in tokens:
            if token_idx >= valid_length - 1:  # 跳过[SEP]
                break
                
            token_clean = token.replace('##', '')
            token_len = len(token_clean)
            
            if char_idx + token_len <= len(text):
                for i in range(token_len):
                    if char_idx + i < len(text):
                        char_labels[char_idx + i] = pred_labels[token_idx]
                char_idx += token_len
                
            token_idx += 1
        
        # 提取实体
        entities = self._extract_entities_from_labels(text, char_labels)
        return entities

    def save_model(self, model_name='data_extractor'):
        """保存数据提取模型"""
        save_path = os.path.join(self.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        # 保存模型状态
        torch.save(self.model.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))

        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)

        # 保存配置
        config = {
            'model_path': self.model_path,
            'num_labels': len(self.label_map),
            'label_map': self.label_map
        }
        with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"数据提取模型已保存到: {save_path}")
        return save_path

    def load_model(self, model_path):
        """加载训练好的模型"""
        # 加载配置
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 重新初始化模型
        self.model = BertDataExtractionModel(
            self.model_path, 
            num_labels=config['num_labels']
        )

        # 加载模型权重
        model_weights_path = os.path.join(model_path, 'pytorch_model.bin')
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)

        logger.info(f"模型已从 {model_path} 加载")

    def extract_data_from_question(self, question):
        """从问题中提取数据类型"""
        entities = self.predict_entities(question)
        
        if entities:
            # 返回最长的实体作为主要数据类型
            main_entity = max(entities, key=len)
            return {
                'main_data_type': main_entity,
                'all_entities': entities,
                'question': question
            }
        else:
            return {
                'main_data_type': None,
                'all_entities': [],
                'question': question
            }


def main():
    """主训练函数"""
    # 模型路径
    model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = DataExtractionTrainer(model_path)

    # 准备训练数据
    texts, labels = trainer.prepare_training_data()

    # 开始训练
    trainer.train(
        epochs=8,
        batch_size=2,  # CPU训练使用较小的batch size
        learning_rate=1e-5
    )

    # 评估模型
    trainer.evaluate(texts, labels)

    # 保存最终模型
    trainer.save_model('final_data_extractor')

    # 测试数据提取功能
    logger.info("\n=== 测试数据提取功能 ===")
    test_questions = [
        "我有一批西安市地类图斑的数据，怎么进行发服务",
        "请帮我把实景三维模型成果数据进行治理",
        "我已经上传了单波段浮点投影的数据，现在想进行入库",
        "有一些遥感影像数据需要处理",
        "DEM高程数据怎么进行质量检查"
    ]

    for question in test_questions:
        result = trainer.extract_data_from_question(question)
        logger.info(f"问题: {question}")
        logger.info(f"提取结果: {result['main_data_type']}")
        logger.info(f"所有实体: {result['all_entities']}")
        logger.info("-" * 50)


if __name__ == "__main__":
    main()
