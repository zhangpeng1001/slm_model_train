"""
基于LoRA技术的问题分类器训练器 - 使用PEFT库实现
使用BERT-base-chinese模型进行低秩适应微调，实现问题三分类：数据平台相关、通用对话、无关问题
使用PEFT库进行高效微调，保留模型原有能力
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

# PEFT库导入
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)

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


class PEFTLoRAQuestionClassifierTrainer:
    """基于PEFT库的LoRA问题分类器训练器"""

    def __init__(self, model_path, save_dir='E:/project/llm/lora/lora_peft_question_classifier',
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

        # 初始化tokenizer和基础模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.base_model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3
        )

        # 配置PEFT LoRA
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # 序列分类任务
            inference_mode=False,  # 训练模式
            r=lora_rank,  # LoRA秩
            lora_alpha=lora_alpha,  # LoRA缩放参数
            lora_dropout=lora_dropout,  # LoRA dropout
            target_modules=[
                "query", "key", "value",  # attention层
                "dense"  # 前馈网络层
            ],
            bias="none",  # 不训练bias
        )

        # 应用PEFT LoRA到模型
        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.to(self.device)

        # 打印可训练参数信息
        self.model.print_trainable_parameters()
        
        logger.info(f"PEFT LoRA分类器已加载到设备: {self.device}")
        logger.info(f"LoRA配置: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")

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
        """训练PEFT LoRA分类器"""
        logger.info("开始训练PEFT LoRA分类器...")

        # 准备训练数据
        texts, labels = self.prepare_training_data()

        # 创建数据加载器
        train_dataloader = self.create_dataloader(texts, labels, batch_size, shuffle=True)

        # 设置优化器 - 只优化可训练参数
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_peft_model('best_model')

        logger.info("PEFT LoRA训练完成！")

    def evaluate(self, texts, labels):
        """评估PEFT LoRA分类器"""
        logger.info("开始评估PEFT LoRA分类器...")

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

    def save_peft_model(self, model_name='peft_lora_question_classifier'):
        """保存PEFT模型"""
        save_path = os.path.join(self.save_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        # 保存PEFT模型（只保存LoRA权重）
        self.model.save_pretrained(save_path)

        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)

        # 保存标签映射
        label_map_path = os.path.join(save_path, 'label_map.json')
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        # 保存配置信息
        config_path = os.path.join(save_path, 'training_config.json')
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

        logger.info(f"PEFT模型已保存到: {save_path}")
        return save_path

    def predict(self, text):
        """使用PEFT LoRA模型进行预测"""
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


class PEFTLoRAQuestionClassifier:
    """PEFT LoRA问题分类器推理类"""

    def __init__(self, base_model_path, peft_model_path):
        self.base_model_path = base_model_path
        self.peft_model_path = peft_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载配置
        config_path = os.path.join(peft_model_path, 'training_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 加载标签映射
        label_map_path = os.path.join(peft_model_path, 'label_map.json')
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
            # 转换键为整数
            self.label_map = {int(k): v for k, v in self.label_map.items()}

        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(peft_model_path)

        # 加载基础模型
        self.base_model = BertForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=self.config['num_labels']
        )

        # 加载PEFT模型
        self.model = PeftModel.from_pretrained(self.base_model, peft_model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"PEFT LoRA分类器推理模型已加载到设备: {self.device}")

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
            'method': 'peft_lora_bert'
        }


def main():
    """主训练函数"""
    # 模型路径 - 使用用户指定的路径
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    print("=" * 60)
    print("🚀 PEFT LoRA问题分类器训练系统")
    print("=" * 60)
    print("基于BERT-base-chinese模型，使用PEFT库进行LoRA高效微调")
    print("保留模型原有能力的同时实现问题三分类")
    print("=" * 60)

    # 创建PEFT LoRA训练器
    trainer = PEFTLoRAQuestionClassifierTrainer(
        model_path=bert_model_path,
        save_dir='E:/project/llm/lora/peft_lora_question_classifier',
        lora_rank=8,  # LoRA秩，控制参数量
        lora_alpha=16,  # LoRA缩放因子
        lora_dropout=0.1  # LoRA dropout率
    )

    # 开始训练
    print("\n开始PEFT LoRA训练...")
    trainer.train(
        epochs=5,  # 训练轮数
        batch_size=8,  # 批次大小
        learning_rate=1e-4  # 学习率
    )

    # 评估模型
    print("\n开始模型评估...")
    texts, labels = trainer.prepare_training_data()
    accuracy = trainer.evaluate(texts, labels)

    # 保存最终模型
    print("\n保存最终模型...")
    save_path = trainer.save_peft_model('final_model')

    print("\n" + "=" * 60)
    print("✅ PEFT LoRA训练完成！")
    print(f"📊 最终准确率: {accuracy:.4f}")
    print(f"💾 模型保存路径: {save_path}")
    print("=" * 60)

    # 测试推理
    print("\n🧪 测试PEFT LoRA推理...")
    try:
        # 创建推理器
        classifier = PEFTLoRAQuestionClassifier(
            base_model_path=bert_model_path,
            peft_model_path=save_path
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


def test_peft_lora_classifier():
    """测试已训练的PEFT LoRA分类器"""
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    peft_model_path = r"E:\project\llm\lora\peft_lora_question_classifier\final_model"

    if not os.path.exists(peft_model_path):
        print(f"PEFT LoRA模型不存在: {peft_model_path}")
        print("请先运行训练程序")
        return

    print("=" * 60)
    print("🧪 PEFT LoRA问题分类器测试")
    print("=" * 60)

    # 创建分类器
    classifier = PEFTLoRAQuestionClassifier(
        base_model_path=bert_model_path,
        peft_model_path=peft_model_path
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


def interactive_peft_lora_test():
    """交互式PEFT LoRA分类器测试"""
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    peft_model_path = r"E:\project\llm\lora\peft_lora_question_classifier\final_model"

    if not os.path.exists(peft_model_path):
        print(f"PEFT LoRA模型不存在: {peft_model_path}")
        print("请先运行训练程序")
        return

    print("=" * 60)
    print("🤖 交互式PEFT LoRA分类器测试")
    print("=" * 60)

    classifier = PEFTLoRAQuestionClassifier(
        base_model_path=bert_model_path,
        peft_model_path=peft_model_path
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
        print("🤖 PEFT LoRA问题分类器系统")
        print("=" * 60)
        print("1. 训练PEFT LoRA分类器")
        print("2. 测试PEFT LoRA分类器")
        print("3. 交互式测试")
        print("0. 退出")
        print("=" * 60)

        try:
            choice = input("请选择功能 (0-3): ").strip()

            if choice == "1":
                main()
            elif choice == "2":
                test_peft_lora_classifier()
            elif choice == "3":
                interactive_peft_lora_test()
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
