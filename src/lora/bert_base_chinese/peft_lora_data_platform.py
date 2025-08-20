"""
基于PEFT库的BERT-base-chinese指令微调训练器
一次性训练，使用所有数据集，生成单个LoRA模型
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import logging
from tqdm import tqdm
from dataset import question_classifier, question_type_classifier, question_answer, file_name_pick

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)


class InstructionDataset(Dataset):
    """指令微调数据集"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建指令格式的完整文本（输入+输出）
        instruction = item["instruction"]
        input_text = item["input"]
        output_text = item["output"]
        
        # 构建完整的训练文本：指令 + 输入 + 输出
        full_text = f"指令：{instruction}\n输入：{input_text}\n输出：{output_text}"
        
        # 编码文本
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # 对于语言建模，标签就是input_ids
        }


def prepare_all_data():
    """准备所有数据集并打乱"""
    # 合并所有数据集
    all_data = question_classifier + question_type_classifier + question_answer + file_name_pick
    
    # 打乱数据以获得更好的训练效果
    random.shuffle(all_data)
    
    logger.info(f"总共准备了 {len(all_data)} 个训练样本")
    logger.info(f"问题场景分类: {len(question_classifier)} 个")
    logger.info(f"问题类型分类: {len(question_type_classifier)} 个") 
    logger.info(f"问题回答: {len(question_answer)} 个")
    logger.info(f"文件名提取: {len(file_name_pick)} 个")
    logger.info("数据已打乱")
    
    return all_data


def train_model():
    """训练模型"""
    # 模型路径配置
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    lora_path = r"E:\project\llm\lora\peft_lora_data_platform"
    
    # 创建保存目录
    os.makedirs(lora_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    # 添加特殊token以支持指令微调
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = BertModel.from_pretrained(bert_model_path)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=16,                    # LoRA秩
        lora_alpha=32,           # LoRA缩放因子
        target_modules=["query", "key", "value", "dense"],  # 目标模块
        lora_dropout=0.1,        # LoRA dropout
        bias="none",             # 偏置设置
        task_type=TaskType.FEATURE_EXTRACTION,  # 特征提取任务
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.to(device)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    # 准备数据
    logger.info("准备训练数据...")
    train_data = prepare_all_data()
    
    # 创建数据集
    train_dataset = InstructionDataset(train_data, tokenizer, max_length=512)
    
    # 数据收集器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=lora_path,
        num_train_epochs=3,                    # 训练轮数
        per_device_train_batch_size=8,         # 批次大小
        per_device_eval_batch_size=8,
        warmup_steps=100,                      # 预热步数
        weight_decay=0.01,                     # 权重衰减
        logging_dir=os.path.join(lora_path, 'logs'),
        logging_steps=50,                      # 日志步数
        save_steps=500,                        # 保存步数
        save_strategy="steps",
        learning_rate=2e-4,                    # 学习率
        fp16=torch.cuda.is_available(),        # 混合精度
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,                        # 禁用wandb等报告
        load_best_model_at_end=False,          # 不需要加载最佳模型
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("开始训练...")
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    logger.info("保存训练好的模型...")
    trainer.save_model()
    tokenizer.save_pretrained(lora_path)
    
    # 保存配置信息
    config = {
        "bert_model_path": bert_model_path,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["query", "key", "value", "dense"]
        },
        "training_info": {
            "total_samples": len(train_data),
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-4
        }
    }
    
    config_path = os.path.join(lora_path, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 训练完成！模型已保存到: {lora_path}")
    logger.info(f"📊 训练样本总数: {len(train_data)}")
    
    return lora_path


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 PEFT LoRA 数据平台训练系统")
    print("基于BERT-base-chinese的指令微调")
    print("=" * 60)
    
    try:
        model_path = train_model()
        print("\n" + "=" * 60)
        print("✅ 训练成功完成！")
        print(f"💾 模型保存路径: {model_path}")
        print("🎉 可以开始使用训练好的模型了！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
