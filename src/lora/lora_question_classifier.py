"""
基于LoRA技术的问题分类器训练器
使用BERT-base-chinese模型进行低秩适应微调，实现问题三分类：数据平台相关、通用对话、无关问题
保留模型原有能力的同时进行高效微调
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
import math

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) 层实现"""

    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 冻结原始层参数
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # 获取原始层的输入输出维度
        if hasattr(original_layer, 'in_features') and hasattr(original_layer, 'out_features'):
            # Linear层
            in_features = original_layer.in_features
            out_features = original_layer.out_features
        elif hasattr(original_layer, 'weight'):
            # 其他层，从权重形状推断
            weight_shape = original_layer.weight.shape
            if len(weight_shape) == 2:
                out_features, in_features = weight_shape
            else:
                raise ValueError(f"不支持的层类型: {type(original_layer)}")
        else:
            raise ValueError(f"无法确定层的输入输出维度: {type(original_layer)}")

        # LoRA的A和B矩阵
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 初始化LoRA权重
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 原始层的输出
        original_output = self.original_layer(x)

        # LoRA的输出
        lora_output = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling

        return original_output + lora_output


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


class LoRAQuestionClassifierTrainer:
    """基于LoRA的问题分类器训练器"""

    def __init__(self, model_path, save_dir='E:/project/llm/lora/lora_question_classifier',
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        self.model_path = model_path
        self.save_dir = save_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # 应用LoRA到模型
        self._apply_lora()

        self.model.to(self.device)
        logger.info(f"LoRA分类器已加载到设备: {self.device}")
        logger.info(f"LoRA配置: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")

    def _apply_lora(self):
        """将LoRA应用到BERT模型的关键层"""
        # 应用LoRA到BERT的attention层
        for layer in self.model.bert.encoder.layer:
            # Query, Key, Value投影层
            layer.attention.self.query = LoRALayer(
                layer.attention.self.query,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout
            )
            layer.attention.self.key = LoRALayer(
                layer.attention.self.key,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout
            )
            layer.attention.self.value = LoRALayer(
                layer.attention.self.value,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout
            )

            # 前馈网络的第一层
            layer.intermediate.dense = LoRALayer(
                layer.intermediate.dense,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout
            )

        # 分类头保持可训练
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"总参数数量: {total_params:,}")
        logger.info(f"可训练参数数量: {trainable_params:,}")
        logger.info(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")

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
            num_workers=0  # Windows系统建议设为0
        )
        return dataloader

    def train(self, epochs=5, batch_size=8, learning_rate=1e-4):
        """训练LoRA分类器"""
        logger.info("开始训练LoRA分类器...")

        # 准备训练数据
        texts, labels = self.prepare_training_data()

        # 创建数据加载器
        train_dataloader = self.create_dataloader(texts, labels, batch_size, shuffle=True)

        # 设置优化器 - 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

        # 设置学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10%的步数用于warmup
            num_training_steps=total_steps
        )

        self.model.train()
        best_loss = float('inf')

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

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_lora_weights('best_model')

        logger.info("LoRA训练完成！")

    def evaluate(self, texts, labels):
        """评估LoRA分类器"""
        logger.info("开始评估LoRA分类器...")

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

    def save_lora_weights(self, model_name='lora_question_classifier'):
        """保存LoRA权重"""
        save_path = os.path.join(self.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        # 保存LoRA权重
        lora_state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer):
                lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight
                lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight

        # 保存分类头权重
        classifier_state_dict = self.model.classifier.state_dict()

        # 合并保存
        torch.save({
            'lora_weights': lora_state_dict,
            'classifier_weights': classifier_state_dict,
            'lora_config': {
                'rank': self.lora_rank,
                'alpha': self.lora_alpha,
                'dropout': self.lora_dropout
            }
        }, os.path.join(save_path, 'lora_weights.pth'))

        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)

        # 保存标签映射
        label_map_path = os.path.join(save_path, 'label_map.json')
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        # 保存配置信息
        config_path = os.path.join(save_path, 'lora_config.json')
        config = {
            'base_model_path': self.model_path,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'num_labels': 3,
            'label_map': self.label_map
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"LoRA权重已保存到: {save_path}")
        return save_path

    def load_lora_weights(self, lora_path):
        """加载LoRA权重"""
        weights_path = os.path.join(lora_path, 'lora_weights.pth')

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"LoRA权重文件不存在: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device)

        # 加载LoRA权重
        lora_weights = checkpoint['lora_weights']
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer):
                if f"{name}.lora_A.weight" in lora_weights:
                    module.lora_A.weight.data = lora_weights[f"{name}.lora_A.weight"]
                if f"{name}.lora_B.weight" in lora_weights:
                    module.lora_B.weight.data = lora_weights[f"{name}.lora_B.weight"]

        # 加载分类头权重
        if 'classifier_weights' in checkpoint:
            self.model.classifier.load_state_dict(checkpoint['classifier_weights'])

        logger.info(f"LoRA权重已从 {lora_path} 加载")

    def predict(self, text):
        """使用LoRA模型进行预测"""
        self.model.eval()

        # 编码输入文本
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'category': self.label_map[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }


class LoRAQuestionClassifier:
    """LoRA问题分类器推理类"""

    def __init__(self, base_model_path, lora_path):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载配置
        config_path = os.path.join(lora_path, 'lora_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 加载标签映射
        label_map_path = os.path.join(lora_path, 'label_map.json')
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
            # 转换键为整数
            self.label_map = {int(k): v for k, v in self.label_map.items()}

        # 初始化模型
        self.tokenizer = BertTokenizer.from_pretrained(lora_path)
        self.model = BertForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=self.config['num_labels']
        )

        # 应用LoRA
        self._apply_lora()

        # 加载LoRA权重
        self._load_lora_weights()

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"LoRA分类器推理模型已加载到设备: {self.device}")

    def _apply_lora(self):
        """应用LoRA到模型"""
        lora_rank = self.config['lora_rank']
        lora_alpha = self.config['lora_alpha']
        lora_dropout = self.config['lora_dropout']

        # 应用LoRA到BERT的attention层
        for layer in self.model.bert.encoder.layer:
            # Query, Key, Value投影层
            layer.attention.self.query = LoRALayer(
                layer.attention.self.query,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            layer.attention.self.key = LoRALayer(
                layer.attention.self.key,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            layer.attention.self.value = LoRALayer(
                layer.attention.self.value,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )

            # 前馈网络的第一层
            layer.intermediate.dense = LoRALayer(
                layer.intermediate.dense,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )

    def _load_lora_weights(self):
        """加载LoRA权重"""
        weights_path = os.path.join(self.lora_path, 'lora_weights.pth')
        checkpoint = torch.load(weights_path, map_location=self.device)

        # 加载LoRA权重
        lora_weights = checkpoint['lora_weights']
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer):
                if f"{name}.lora_A.weight" in lora_weights:
                    module.lora_A.weight.data = lora_weights[f"{name}.lora_A.weight"]
                if f"{name}.lora_B.weight" in lora_weights:
                    module.lora_B.weight.data = lora_weights[f"{name}.lora_B.weight"]

        # 加载分类头权重
        if 'classifier_weights' in checkpoint:
            self.model.classifier.load_state_dict(checkpoint['classifier_weights'])

    def classify_question(self, text):
        """对问题进行分类"""
        # 编码输入文本
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'category': self.label_map[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'method': 'lora_bert'
        }


def main():
    """主训练函数"""
    # 模型路径 - 使用用户指定的路径
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    print("=" * 60)
    print("🚀 LoRA问题分类器训练系统")
    print("=" * 60)
    print("基于BERT-base-chinese模型，使用LoRA技术进行高效微调")
    print("保留模型原有能力的同时实现问题三分类")
    print("=" * 60)

    # 创建LoRA训练器
    trainer = LoRAQuestionClassifierTrainer(
        model_path=bert_model_path,
        save_dir='E:/project/llm/lora/lora_question_classifier',
        lora_rank=8,  # LoRA秩，控制参数量
        lora_alpha=16,  # LoRA缩放因子
        lora_dropout=0.1  # LoRA dropout率
    )

    # 开始训练
    print("\n开始LoRA训练...")
    trainer.train(
        epochs=5,  # 训练轮数
        batch_size=8,  # 批次大小
        learning_rate=1e-4  # 学习率，比全量微调稍高
    )

    # 评估模型
    print("\n开始模型评估...")
    texts, labels = trainer.prepare_training_data()
    accuracy = trainer.evaluate(texts, labels)

    # 保存最终模型
    print("\n保存最终模型...")
    save_path = trainer.save_lora_weights('final_model')

    print("\n" + "=" * 60)
    print("✅ LoRA训练完成！")
    print(f"📊 最终准确率: {accuracy:.4f}")
    print(f"💾 模型保存路径: {save_path}")
    print("=" * 60)

    # 测试推理
    print("\n🧪 测试LoRA推理...")
    try:
        # 创建推理器
        classifier = LoRAQuestionClassifier(
            base_model_path=bert_model_path,
            lora_path=os.path.join(save_path)
        )

        # 测试问题
        test_questions = [
            "数据清洗流程是什么？",  # 数据平台相关
            "你好",  # 通用对话
            "今天天气怎么样？",  # 无关问题
        ]

        print("\n测试结果：")
        print("-" * 40)
        for question in test_questions:
            result = classifier.classify_question(question)
            category_cn = {
                "data_platform": "数据平台相关",
                "general_chat": "通用对话",
                "irrelevant": "无关问题"
            }
            print(f"问题: {question}")
            print(f"分类: {category_cn[result['category']]} (置信度: {result['confidence']:.3f})")
            print("-" * 40)

    except Exception as e:
        print(f"推理测试失败: {e}")

    return save_path


def test_lora_classifier():
    """测试已训练的LoRA分类器"""
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    lora_path = r"E:\project\llm\lora\lora_question_classifier\final_model"

    if not os.path.exists(lora_path):
        print(f"LoRA模型不存在: {lora_path}")
        print("请先运行训练程序")
        return

    print("=" * 60)
    print("🧪 LoRA问题分类器测试")
    print("=" * 60)

    # 创建分类器
    classifier = LoRAQuestionClassifier(
        base_model_path=bert_model_path,
        lora_path=lora_path
    )

    # 测试问题
    test_questions = [
        # 数据平台相关
        "数据清洗流程是什么？",
        "如何进行数据入库？",
        "数据质量检查方法",
        "数据监控怎么做？",
        "数据安全措施",

        # 通用对话
        "你好",
        "谢谢",
        "再见",
        "不客气",
        "好的",

        # 无关问题
        "今天天气怎么样？",
        "北京有什么好吃的？",
        "如何学习英语？",
        "what is your name?",
        "asdfghjkl"
    ]

    print("测试结果：")
    print("-" * 60)

    category_cn = {
        "data_platform": "数据平台相关",
        "general_chat": "通用对话",
        "irrelevant": "无关问题"
    }

    for question in test_questions:
        result = classifier.classify_question(question)
        print(f"问题: {question}")
        print(f"分类: {category_cn[result['category']]} (置信度: {result['confidence']:.3f})")
        print(f"方法: {result['method']}")
        print("-" * 40)


def interactive_lora_test():
    """交互式LoRA分类器测试"""
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    lora_path = r"E:\project\llm\lora\lora_question_classifier\final_model"

    if not os.path.exists(lora_path):
        print(f"LoRA模型不存在: {lora_path}")
        print("请先运行训练程序")
        return

    print("=" * 60)
    print("🤖 交互式LoRA分类器测试")
    print("=" * 60)

    classifier = LoRAQuestionClassifier(
        base_model_path=bert_model_path,
        lora_path=lora_path
    )

    category_cn = {
        "data_platform": "数据平台相关",
        "general_chat": "通用对话",
        "irrelevant": "无关问题"
    }

    print("输入问题进行分类测试，输入 'quit' 退出")
    print("-" * 60)

    while True:
        try:
            question = input("请输入问题: ").strip()

            if question.lower() in ['quit', 'exit', '退出', 'q']:
                break

            if not question:
                continue

            result = classifier.classify_question(question)

            print(f"分类结果: {category_cn[result['category']]}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"分类方法: {result['method']}")
            print("-" * 40)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"处理错误: {e}")


if __name__ == "__main__":
    # 默认显示菜单
    while True:
        print("\n" + "=" * 60)
        print("🤖 LoRA问题分类器系统")
        print("=" * 60)
        print("1. 训练LoRA分类器")
        print("2. 测试LoRA分类器")
        print("3. 交互式测试")
        print("0. 退出")
        print("=" * 60)

        try:
            choice = input("请选择功能 (0-3): ").strip()

            if choice == "1":
                main()
            elif choice == "2":
                test_lora_classifier()
            elif choice == "3":
                interactive_lora_test()
            elif choice == "0":
                print("感谢使用！再见！")
                break
            else:
                print("无效选择，请重新输入")

        except KeyboardInterrupt:
            print("\n感谢使用！再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
