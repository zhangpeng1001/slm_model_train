"""
统一PEFT训练器
使用LoRA技术对BERT模型进行微调，整合四个功能：
1. 问题分类器
2. 问题类型分类器  
3. 数据平台问答系统
4. 数据提取器

通过提示词区分不同任务，实现一个模型多种功能
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    get_linear_schedule_with_warmup, TrainingArguments, Trainer
)
from peft import (
    get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
)
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDataset(Dataset):
    """统一数据集，包含所有任务的数据"""

    def __init__(self, samples, tokenizer, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构建带提示词的输入文本
        prompt_text = self._build_prompt_text(sample)
        
        # 编码输入文本
        encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'task_type': sample['task_type'],
            'original_text': sample['text'],
            'prompt_text': prompt_text
        }

        # 根据任务类型添加标签
        if sample['task_type'] in ['question_classifier', 'question_type_classifier']:
            result['labels'] = torch.tensor(sample['label'], dtype=torch.long)
        elif sample['task_type'] == 'qa_system':
            # 对于问答任务，编码目标答案
            answer_encoding = self.tokenizer(
                sample['answer'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            result['answer_input_ids'] = answer_encoding['input_ids'].flatten()
            result['answer_attention_mask'] = answer_encoding['attention_mask'].flatten()
        elif sample['task_type'] == 'data_extraction':
            # 对于NER任务，使用序列标签
            result['ner_labels'] = torch.tensor(sample['ner_labels'], dtype=torch.long)

        return result

    def _build_prompt_text(self, sample):
        """构建带提示词的文本"""
        task_type = sample['task_type']
        text = sample['text']
        
        if task_type == 'question_classifier':
            return f"[任务：问题分类] 请判断以下问题属于哪个类别（数据平台相关/通用对话/无关问题）：{text}"
        elif task_type == 'question_type_classifier':
            return f"[任务：问题类型分类] 请判断以下问题的意图类型（问题回答/任务处理）：{text}"
        elif task_type == 'qa_system':
            return f"[任务：问答系统] 请回答以下关于数据平台的问题：{text}"
        elif task_type == 'data_extraction':
            return f"[任务：数据提取] 请从以下文本中提取数据类型信息：{text}"
        else:
            return text


class UnifiedPeftModel(nn.Module):
    """统一的PEFT模型"""

    def __init__(self, model_path):
        super(UnifiedPeftModel, self).__init__()
        
        # 加载基础BERT模型
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=16,  # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value", "key", "dense"]  # 对注意力层应用LoRA
        )
        
        # 应用LoRA到BERT模型
        self.bert = get_peft_model(self.bert, lora_config)
        
        # 任务特定的头部
        self.dropout = nn.Dropout(0.1)
        
        # 问题分类器头部（3分类）
        self.question_classifier_head = nn.Linear(self.hidden_size, 3)
        
        # 问题类型分类器头部（2分类）
        self.question_type_classifier_head = nn.Linear(self.hidden_size, 2)
        
        # 问答系统头部
        self.qa_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # 数据提取头部（NER，3分类：O, B-DATA, I-DATA）
        self.ner_head = nn.Linear(self.hidden_size, 3)

    def forward(self, input_ids, attention_mask, task_type, **kwargs):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # 根据任务类型选择对应的头部
        if task_type == 'question_classifier':
            logits = self.question_classifier_head(pooled_output)
            return {'logits': logits}
        elif task_type == 'question_type_classifier':
            logits = self.question_type_classifier_head(pooled_output)
            return {'logits': logits}
        elif task_type == 'qa_system':
            qa_output = self.qa_head(pooled_output)
            return {'qa_output': qa_output}
        elif task_type == 'data_extraction':
            # 对于NER任务，使用序列输出
            sequence_output = self.dropout(sequence_output)
            logits = self.ner_head(sequence_output)
            return {'logits': logits}
        else:
            raise ValueError(f"Unsupported task type: {task_type}")


class UnifiedPeftTrainer:
    """统一PEFT训练器"""

    def __init__(self, model_path, save_dir=r'E:\project\llm\lora\peft'):
        self.model_path = model_path
        self.save_dir = save_dir
        self.device = torch.device('cpu')  # 强制使用CPU

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 初始化tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = UnifiedPeftModel(model_path)
        self.model.to(self.device)

        # 标签映射
        self.label_maps = {
            'question_classifier': {
                0: "data_platform",  # 数据平台相关
                1: "general_chat",   # 通用对话
                2: "irrelevant"      # 无关问题
            },
            'question_type_classifier': {
                0: "问题回答",  # 问题回答
                1: "任务处理"   # 任务处理
            },
            'data_extraction': {
                0: "O",      # 其他
                1: "B-DATA", # 数据类型开始
                2: "I-DATA"  # 数据类型内部
            }
        }

        logger.info(f"统一PEFT模型已加载到设备: {self.device}")

    def prepare_training_data(self):
        """准备所有任务的训练数据"""
        all_samples = []

        # 1. 问题分类器数据
        question_classifier_samples = self._prepare_question_classifier_data()
        all_samples.extend(question_classifier_samples)

        # 2. 问题类型分类器数据
        question_type_classifier_samples = self._prepare_question_type_classifier_data()
        all_samples.extend(question_type_classifier_samples)

        # 3. 问答系统数据
        qa_system_samples = self._prepare_qa_system_data()
        all_samples.extend(qa_system_samples)

        # 4. 数据提取器数据
        data_extraction_samples = self._prepare_data_extraction_data()
        all_samples.extend(data_extraction_samples)

        # 打乱数据
        random.shuffle(all_samples)

        logger.info(f"准备了总计 {len(all_samples)} 个训练样本")
        logger.info(f"问题分类器: {len(question_classifier_samples)} 个")
        logger.info(f"问题类型分类器: {len(question_type_classifier_samples)} 个")
        logger.info(f"问答系统: {len(qa_system_samples)} 个")
        logger.info(f"数据提取器: {len(data_extraction_samples)} 个")

        return all_samples

    def _prepare_question_classifier_data(self):
        """准备问题分类器数据"""
        samples = []

        # 数据平台相关问题（类别0）
        data_platform_questions = [
            "数据清洗流程是什么？",
            "如何进行数据清洗？",
            "数据清洗步骤有哪些？",
            "数据入库流程",
            "如何进行数据入库？",
            "数据质量检查",
            "如何检查数据质量？",
            "数据监控",
            "如何监控数据？",
            "数据安全",
            "如何保证数据安全？",
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
            "你好", "您好", "hi", "hello", "早上好", "下午好", "晚上好",
            "谢谢", "谢谢你", "感谢", "不客气", "再见", "拜拜", "bye",
            "好的", "知道了", "明白了", "收到", "OK", "可以", "没问题",
            "帮忙", "请问", "能否", "麻烦", "打扰了", "不好意思", "对不起", "抱歉"
        ]

        # 无关问题（类别2）
        irrelevant_questions = [
            "今天天气怎么样？", "北京有什么好吃的？", "如何学习英语？",
            "Python怎么安装？", "什么是机器学习？", "股票今天涨了吗？",
            "电影推荐一下", "音乐好听吗？", "游戏攻略", "旅游景点推荐",
            "美食制作方法", "健身计划", "减肥方法", "护肤品推荐",
            "汽车保养", "房价走势", "教育政策", "医疗保险", "法律咨询",
            "心理健康", "what is your name?", "how are you?", "where are you from?"
        ]

        # 转换为样本格式
        for text in data_platform_questions:
            samples.append({
                'task_type': 'question_classifier',
                'text': text,
                'label': 0
            })

        for text in general_chat_questions:
            samples.append({
                'task_type': 'question_classifier',
                'text': text,
                'label': 1
            })

        for text in irrelevant_questions:
            samples.append({
                'task_type': 'question_classifier',
                'text': text,
                'label': 2
            })

        return samples

    def _prepare_question_type_classifier_data(self):
        """准备问题类型分类器数据"""
        samples = []

        # 问题回答类型（类别0）
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
            "遥感影像数据的配准方法有哪些",
            "如何选择合适的坐标系统",
            "数据质量检查的标准是什么",
            "什么是数据血缘关系",
            "元数据管理的最佳实践",
            "空间数据索引如何建立"
        ]

        # 任务处理类型（类别1）
        task_processing_samples = [
            "请帮我把实景三维模型成果数据进行治理",
            "我已经上传了单波段浮点投影的数据，现在想进行入库",
            "有一些遥感影像数据需要处理",
            "正射影像数据需要进行坐标转换",
            "激光雷达数据进行格式转换",
            "地籍调查数据需要标准化处理",
            "处理地下管线探测数据",
            "倾斜摄影测量数据进行建模",
            "帮我清洗这批地形数据",
            "需要对影像数据进行几何校正",
            "请处理这些GPS轨迹数据",
            "批量转换CAD文件格式",
            "执行数据质量检查任务",
            "启动数据同步作业",
            "运行地理编码程序"
        ]

        # 转换为样本格式
        for text in question_answer_samples:
            samples.append({
                'task_type': 'question_type_classifier',
                'text': text,
                'label': 0
            })

        for text in task_processing_samples:
            samples.append({
                'task_type': 'question_type_classifier',
                'text': text,
                'label': 1
            })

        return samples

    def _prepare_qa_system_data(self):
        """准备问答系统数据"""
        samples = []

        # 问答对数据
        qa_pairs = [
            {
                'question': '数据清洗流程是什么？',
                'answer': '数据清洗流程包括：1.数据检查和分析 2.识别和处理缺失值 3.检测和处理异常值 4.数据格式标准化 5.数据一致性检查 6.数据验证和确认。整个过程需要确保数据的准确性、完整性和一致性。'
            },
            {
                'question': '如何进行数据入库？',
                'answer': '数据入库流程：1.数据格式验证 2.数据质量检查 3.建立数据库连接 4.创建数据表结构 5.数据导入和加载 6.建立索引和约束 7.数据完整性验证 8.权限设置和访问控制。'
            },
            {
                'question': '数据质量检查包括哪些方面？',
                'answer': '数据质量检查包括：1.完整性检查-检查数据是否缺失 2.准确性检查-验证数据的正确性 3.一致性检查-确保数据格式统一 4.有效性检查-验证数据范围和格式 5.及时性检查-确保数据的时效性 6.唯一性检查-避免重复数据。'
            },
            {
                'question': '如何进行数据监控？',
                'answer': '数据监控方案：1.建立监控指标体系 2.设置数据质量阈值 3.实时监控数据流 4.异常检测和告警 5.监控报表和仪表板 6.定期数据健康检查 7.监控日志记录和分析。'
            },
            {
                'question': '数据安全措施有哪些？',
                'answer': '数据安全措施包括：1.访问控制和权限管理 2.数据加密存储和传输 3.数据备份和恢复策略 4.审计日志和监控 5.网络安全防护 6.数据脱敏和匿名化 7.安全培训和管理制度。'
            },
            {
                'question': '什么是ETL流程？',
                'answer': 'ETL是Extract(提取)、Transform(转换)、Load(加载)的缩写。ETL流程包括：1.Extract-从源系统提取数据 2.Transform-对数据进行清洗、转换、聚合等处理 3.Load-将处理后的数据加载到目标系统。这是数据仓库建设的核心流程。'
            }
        ]

        for qa_pair in qa_pairs:
            samples.append({
                'task_type': 'qa_system',
                'text': qa_pair['question'],
                'answer': qa_pair['answer']
            })

        return samples

    def _prepare_data_extraction_data(self):
        """准备数据提取器数据"""
        samples = []

        # NER训练数据
        ner_samples = [
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
            ("基础地理信息数据如何管理", "基础地理信息")
        ]

        for text, target in ner_samples:
            # 创建BIO标注
            ner_labels = self._create_bio_labels(text, target)
            samples.append({
                'task_type': 'data_extraction',
                'text': text,
                'target': target,
                'ner_labels': ner_labels
            })

        return samples

    def _create_bio_labels(self, text, target):
        """创建BIO标注序列"""
        # 简化处理：为整个序列创建标签，长度与max_length一致
        labels = [0] * 256  # 默认为O标签，与max_length一致
        
        # 查找目标实体在文本中的位置
        target_start = text.find(target)
        if target_start != -1:
            target_end = target_start + len(target)
            
            # 由于tokenization的复杂性，这里使用简化的映射
            # 在实际应用中，需要更精确的字符到token的映射
            if target_start < len(labels):
                labels[target_start] = 1  # B-DATA
                for i in range(target_start + 1, min(target_end, len(labels))):
                    labels[i] = 2  # I-DATA
                    
        return labels

    def create_dataloader(self, samples, batch_size=4, shuffle=True):
        """创建数据加载器"""
        dataset = UnifiedDataset(samples, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self._collate_fn
        )
        return dataloader

    def _collate_fn(self, batch):
        """自定义批处理函数"""
        # 按任务类型分组
        task_groups = {}
        for item in batch:
            task_type = item['task_type']
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(item)

        # 为了简化，这里返回第一个任务类型的批次
        # 在实际应用中，可能需要更复杂的批处理策略
        first_task = list(task_groups.keys())[0]
        items = task_groups[first_task]

        # 构建批次
        batch_data = {
            'input_ids': torch.stack([item['input_ids'] for item in items]),
            'attention_mask': torch.stack([item['attention_mask'] for item in items]),
            'task_type': first_task,
            'original_texts': [item['original_text'] for item in items],
            'prompt_texts': [item['prompt_text'] for item in items]
        }

        # 根据任务类型添加相应的标签
        if first_task in ['question_classifier', 'question_type_classifier']:
            batch_data['labels'] = torch.stack([item['labels'] for item in items])
        elif first_task == 'qa_system':
            batch_data['answer_input_ids'] = torch.stack([item['answer_input_ids'] for item in items])
            batch_data['answer_attention_mask'] = torch.stack([item['answer_attention_mask'] for item in items])
        elif first_task == 'data_extraction':
            batch_data['ner_labels'] = torch.stack([item['ner_labels'] for item in items])

        return batch_data

    def train(self, epochs=5, batch_size=4, learning_rate=2e-5):
        """训练统一模型"""
        logger.info("开始训练统一PEFT模型...")

        # 准备训练数据
        all_samples = self.prepare_training_data()

        # 创建数据加载器
        train_dataloader = self.create_dataloader(all_samples, batch_size, shuffle=True)

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
        classification_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch in progress_bar:
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                task_type = batch['task_type']

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type=task_type
                )

                # 计算损失
                loss = None
                if task_type in ['question_classifier', 'question_type_classifier']:
                    labels = batch['labels'].to(self.device)
                    logits = outputs['logits']
                    loss = classification_criterion(logits, labels)
                elif task_type == 'qa_system':
                    # 对于问答任务，使用简化的损失计算
                    qa_output = outputs['qa_output']
                    # 这里可以添加更复杂的问答损失计算
                    # 暂时使用一个简单的目标
                    target = torch.zeros_like(qa_output)
                    loss = mse_criterion(qa_output, target)
                elif task_type == 'data_extraction':
                    ner_labels = batch['ner_labels'].to(self.device)
                    logits = outputs['logits']
                    # 重塑logits和labels以匹配CrossEntropyLoss的要求
                    loss = classification_criterion(
                        logits.view(-1, logits.size(-1)),
                        ner_labels.view(-1)
                    )

                if loss is not None:
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': loss.item(), 'task': task_type})

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # 每2个epoch保存一次检查点
            if (epoch + 1) % 2 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}')

        logger.info("训练完成！")

    def predict(self, text, task_type):
        """预测单个文本"""
        self.model.eval()

        # 构建提示词
        sample = {'task_type': task_type, 'text': text}
        prompt_text = self._build_prompt_text(sample)

        # 编码文本
        encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_type=task_type
            )

            if task_type in ['question_classifier', 'question_type_classifier', 'data_extraction']:
                logits = outputs['logits']
                if task_type == 'data_extraction':
                    # 对于NER任务，返回序列预测
                    predictions = torch.argmax(logits, dim=2)
                    return {
                        'task_type': task_type,
                        'text': text,
                        'predictions': predictions[0].cpu().numpy(),
                        'entities': self._extract_entities_from_predictions(text, predictions[0].cpu().numpy())
                    }
                else:
                    # 对于分类任务
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_label = torch.argmax(logits, dim=1).item()
                    confidence = probabilities[0][predicted_label].item()

                    return {
                        'task_type': task_type,
                        'text': text,
                        'predicted_label': predicted_label,
                        'predicted_class': self.label_maps[task_type][predicted_label],
                        'confidence': confidence,
                        'probabilities': probabilities[0].cpu().numpy()
                    }
            elif task_type == 'qa_system':
                qa_output = outputs['qa_output']
                return {
                    'task_type': task_type,
                    'text': text,
                    'qa_output': qa_output[0].cpu().numpy()
                }

    def _build_prompt_text(self, sample):
        """构建带提示词的文本（与数据集中的方法相同）"""
        task_type = sample['task_type']
        text = sample['text']
        
        if task_type == 'question_classifier':
            return f"[任务：问题分类] 请判断以下问题属于哪个类别（数据平台相关/通用对话/无关问题）：{text}"
        elif task_type == 'question_type_classifier':
            return f"[任务：问题类型分类] 请判断以下问题的意图类型（问题回答/任务处理）：{text}"
        elif task_type == 'qa_system':
            return f"[任务：问答系统] 请回答以下关于数据平台的问题：{text}"
        elif task_type == 'data_extraction':
            return f"[任务：数据提取] 请从以下文本中提取数据类型信息：{text}"
        else:
            return text

    def _extract_entities_from_predictions(self, text, predictions):
        """从NER预测中提取实体"""
        entities = []
        current_entity = ""
        in_entity = False
        
        # 简化处理：直接从预测序列中提取实体
        for i, pred in enumerate(predictions):
            if i >= len(text):
                break
                
            if pred == 1:  # B-DATA
                if in_entity and current_entity:
                    entities.append(current_entity)
                current_entity = text[i] if i < len(text) else ""
                in_entity = True
            elif pred == 2 and in_entity:  # I-DATA
                if i < len(text):
                    current_entity += text[i]
            else:  # O
                if in_entity and current_entity:
                    entities.append(current_entity)
                current_entity = ""
                in_entity = False
                
        if in_entity and current_entity:
            entities.append(current_entity)
            
        return entities

    def save_model(self, model_name='unified_peft_model'):
        """保存PEFT模型"""
        save_path = os.path.join(self.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        # 保存PEFT模型
        self.model.bert.save_pretrained(save_path)

        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)

        # 保存任务头部的权重
        task_heads = {
            'question_classifier_head': self.model.question_classifier_head.state_dict(),
            'question_type_classifier_head': self.model.question_type_classifier_head.state_dict(),
            'qa_head': self.model.qa_head.state_dict(),
            'ner_head': self.model.ner_head.state_dict()
        }
        torch.save(task_heads, os.path.join(save_path, 'task_heads.pth'))

        # 保存配置
        config = {
            'model_path': self.model_path,
            'hidden_size': self.model.hidden_size,
            'label_maps': self.label_maps
        }
        with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"统一PEFT模型已保存到: {save_path}")
        return save_path

    def load_model(self, model_path):
        """加载训练好的PEFT模型"""
        logger.info(f"从 {model_path} 加载模型...")

        # 加载配置
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 加载PEFT模型
        self.model.bert = PeftModel.from_pretrained(
            self.model.bert.base_model, 
            model_path
        )

        # 加载任务头部权重
        task_heads_path = os.path.join(model_path, 'task_heads.pth')
        if os.path.exists(task_heads_path):
            task_heads = torch.load(task_heads_path, map_location=self.device)
            self.model.question_classifier_head.load_state_dict(task_heads['question_classifier_head'])
            self.model.question_type_classifier_head.load_state_dict(task_heads['question_type_classifier_head'])
            self.model.qa_head.load_state_dict(task_heads['qa_head'])
            self.model.ner_head.load_state_dict(task_heads['ner_head'])

        self.model.to(self.device)
        logger.info("模型加载完成！")


def main():
    """主训练函数"""
    # 基础模型路径
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建统一PEFT训练器
    trainer = UnifiedPeftTrainer(bert_model_path)

    # 开始训练
    trainer.train(
        epochs=8,  # 增加训练轮次以确保所有任务都能充分学习
        batch_size=2,  # CPU训练使用较小的batch size
        learning_rate=1e-5  # 使用较小的学习率进行精细调整
    )

    # 保存最终模型
    final_model_path = trainer.save_model('final_unified_peft_model')

    # 测试各个任务的功能
    logger.info("\n=== 测试统一PEFT模型功能 ===")

    # 测试问题分类器
    logger.info("\n--- 测试问题分类器 ---")
    test_questions_classifier = [
        "数据清洗流程是什么？",
        "你好，请问可以帮忙吗？",
        "今天天气怎么样？"
    ]
    
    for question in test_questions_classifier:
        result = trainer.predict(question, 'question_classifier')
        logger.info(f"问题: {question}")
        logger.info(f"分类结果: {result['predicted_class']} (置信度: {result['confidence']:.4f})")
        logger.info("-" * 50)

    # 测试问题类型分类器
    logger.info("\n--- 测试问题类型分类器 ---")
    test_questions_type = [
        "我有一批西安市地类图斑的数据，怎么进行发服务",
        "请帮我把实景三维模型成果数据进行治理"
    ]
    
    for question in test_questions_type:
        result = trainer.predict(question, 'question_type_classifier')
        logger.info(f"问题: {question}")
        logger.info(f"意图类型: {result['predicted_class']} (置信度: {result['confidence']:.4f})")
        logger.info("-" * 50)

    # 测试数据提取器
    logger.info("\n--- 测试数据提取器 ---")
    test_questions_extraction = [
        "我有一批西安市地类图斑的数据，怎么进行发服务",
        "请帮我把实景三维模型成果数据进行治理",
        "有一些遥感影像数据需要处理"
    ]
    
    for question in test_questions_extraction:
        result = trainer.predict(question, 'data_extraction')
        logger.info(f"问题: {question}")
        logger.info(f"提取的实体: {result['entities']}")
        logger.info("-" * 50)

    # 测试问答系统
    logger.info("\n--- 测试问答系统 ---")
    test_questions_qa = [
        "数据清洗流程是什么？",
        "如何进行数据入库？",
        "数据质量检查包括哪些方面？"
    ]
    
    for question in test_questions_qa:
        result = trainer.predict(question, 'qa_system')
        logger.info(f"问题: {question}")
        logger.info(f"问答输出维度: {result['qa_output'].shape}")
        logger.info("-" * 50)

    logger.info(f"\n统一PEFT模型训练完成！模型已保存至: {final_model_path}")
    logger.info("该模型整合了四个功能：问题分类、问题类型分类、数据平台问答、数据提取")
    logger.info("使用时通过提示词指定任务类型，实现一个模型多种功能")


if __name__ == "__main__":
    main()
