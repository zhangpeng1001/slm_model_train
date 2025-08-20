"""
基于PEFT库的BERT-base-chinese多模式指令微调训练器
支持4种数据集的不同训练模式，都支持指令微调
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
import torch.nn as nn
from typing import Dict, List, Optional, Union
import argparse

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


class BertForInstructionTuning(nn.Module):
    """自定义BERT模型，支持指令微调"""

    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.config = bert_model.config
        # 添加语言建模头
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 过滤掉BERT不支持的参数
        bert_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['num_items_in_batch']:  # 过滤掉训练器传递的额外参数
                bert_kwargs[key] = value

        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **bert_kwargs
        )

        # 计算logits
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # 如果有labels，计算损失（用于训练）
        if labels is not None:
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 创建一个可以被索引的输出对象
            class ModelOutput:
                def __init__(self, loss, logits, last_hidden_state):
                    self.loss = loss
                    self.logits = logits
                    self.last_hidden_state = last_hidden_state

                def __getitem__(self, key):
                    if key == 0:
                        return self.loss
                    elif key == 1:
                        return self.logits
                    else:
                        raise IndexError(f"Index {key} out of range")

                def __contains__(self, key):
                    return key in ['loss', 'logits', 'last_hidden_state']

                def __getattr__(self, name):
                    if name == 'loss':
                        return self.loss
                    elif name == 'logits':
                        return self.logits
                    elif name == 'last_hidden_state':
                        return self.last_hidden_state
                    else:
                        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

            return ModelOutput(loss, logits, hidden_states)

        # 创建一个可以被索引的输出对象（无损失）
        class ModelOutput:
            def __init__(self, logits, last_hidden_state):
                self.logits = logits
                self.last_hidden_state = last_hidden_state

            def __getitem__(self, key):
                if key == 0:
                    return self.logits
                else:
                    raise IndexError(f"Index {key} out of range")

            def __contains__(self, key):
                return key in ['logits', 'last_hidden_state']

        return ModelOutput(logits, hidden_states)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """为生成准备输入，PEFT需要这个方法"""
        return {"input_ids": input_ids}

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.bert.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        """获取输出嵌入层"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        self.lm_head = new_embeddings


class InstructionDataset(Dataset):
    """指令微调数据集"""

    def __init__(self, data, tokenizer, max_length=512, dataset_type="mixed"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建指令格式的完整文本（输入+输出）
        instruction = item["instruction"]
        input_text = item["input"]
        output_text = item["output"]

        # 根据数据集类型构建不同的指令格式
        if self.dataset_type == "question_classifier":
            full_text = f"任务：问题场景分类\n指令：{instruction}\n问题：{input_text}\n分类结果：{output_text}"
        elif self.dataset_type == "question_type_classifier":
            full_text = f"任务：问题类型分类\n指令：{instruction}\n问题：{input_text}\n类型：{output_text}"
        elif self.dataset_type == "question_answer":
            full_text = f"任务：问题回答\n指令：{instruction}\n问题：{input_text}\n回答：{output_text}"
        elif self.dataset_type == "file_name_pick":
            full_text = f"任务：文件名称提取\n指令：{instruction}\n文本：{input_text}\n提取结果：{output_text}"
        else:  # mixed mode
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


class DatasetManager:
    """数据集管理器，支持不同的训练模式"""
    
    def __init__(self):
        self.datasets = {
            "question_classifier": question_classifier,
            "question_type_classifier": question_type_classifier,
            "question_answer": question_answer,
            "file_name_pick": file_name_pick
        }
    
    def get_dataset(self, dataset_type: str) -> List[Dict]:
        """获取指定类型的数据集"""
        if dataset_type == "all" or dataset_type == "mixed":
            # 合并所有数据集
            all_data = []
            for data in self.datasets.values():
                all_data.extend(data)
            random.shuffle(all_data)
            return all_data
        elif dataset_type in self.datasets:
            data = self.datasets[dataset_type].copy()
            random.shuffle(data)
            return data
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    def get_dataset_info(self, dataset_type: str) -> Dict:
        """获取数据集信息"""
        if dataset_type == "all" or dataset_type == "mixed":
            total_samples = sum(len(data) for data in self.datasets.values())
            info = {
                "total_samples": total_samples,
                "datasets": {name: len(data) for name, data in self.datasets.items()}
            }
        elif dataset_type in self.datasets:
            info = {
                "total_samples": len(self.datasets[dataset_type]),
                "dataset_type": dataset_type
            }
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        return info


class MultiModeTrainer:
    """多模式训练器"""
    
    def __init__(self, 
                 bert_model_path: str = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea",
                 base_output_dir: str = r"E:\project\llm\lora"):
        self.bert_model_path = bert_model_path
        self.base_output_dir = base_output_dir
        self.dataset_manager = DatasetManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练配置
        self.training_configs = {
            "question_classifier": {
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 32,
                "description": "问题场景分类任务"
            },
            "question_type_classifier": {
                "epochs": 4,
                "batch_size": 8,
                "learning_rate": 1.5e-4,
                "lora_r": 12,
                "lora_alpha": 24,
                "description": "问题类型分类任务"
            },
            "question_answer": {
                "epochs": 6,
                "batch_size": 6,
                "learning_rate": 2.5e-4,
                "lora_r": 20,
                "lora_alpha": 40,
                "description": "问题回答任务"
            },
            "file_name_pick": {
                "epochs": 4,
                "batch_size": 8,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 32,
                "description": "文件名称提取任务"
            },
            "mixed": {
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 32,
                "description": "混合多任务训练"
            }
        }
    
    def prepare_model_and_tokenizer(self, config: Dict):
        """准备模型和分词器"""
        logger.info(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        
        # 加载基础BERT模型
        bert_model = BertModel.from_pretrained(self.bert_model_path)
        
        # 使用自定义包装器
        model = BertForInstructionTuning(bert_model)
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.to(self.device)
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
        
        return model, tokenizer, lora_config
    
    def train_single_dataset(self, dataset_type: str, custom_output_dir: Optional[str] = None):
        """训练单个数据集"""
        if dataset_type not in self.training_configs:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        config = self.training_configs[dataset_type]
        logger.info(f"开始训练 {config['description']} ({dataset_type})")
        
        # 准备输出目录
        if custom_output_dir:
            output_dir = custom_output_dir
        else:
            output_dir = os.path.join(self.base_output_dir, f"peft_lora_{dataset_type}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备模型和分词器
        model, tokenizer, lora_config = self.prepare_model_and_tokenizer(config)
        
        # 准备数据
        logger.info("准备训练数据...")
        train_data = self.dataset_manager.get_dataset(dataset_type)
        dataset_info = self.dataset_manager.get_dataset_info(dataset_type)
        
        logger.info(f"数据集信息: {dataset_info}")
        
        # 创建数据集
        train_dataset = InstructionDataset(train_data, tokenizer, max_length=512, dataset_type=dataset_type)
        
        # 数据收集器
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=50,
            save_steps=500,
            save_strategy="steps",
            learning_rate=config["learning_rate"],
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            load_best_model_at_end=False,
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
        tokenizer.save_pretrained(output_dir)
        
        # 保存配置信息
        training_config = {
            "bert_model_path": self.bert_model_path,
            "dataset_type": dataset_type,
            "dataset_info": dataset_info,
            "lora_config": {
                "r": config["lora_r"],
                "lora_alpha": config["lora_alpha"],
                "lora_dropout": 0.1,
                "target_modules": ["query", "key", "value", "dense"]
            },
            "training_info": {
                "total_samples": dataset_info["total_samples"],
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "learning_rate": config["learning_rate"],
                "description": config["description"]
            }
        }
        
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ {config['description']} 训练完成！")
        logger.info(f"💾 模型保存路径: {output_dir}")
        logger.info(f"📊 训练样本数: {dataset_info['total_samples']}")
        
        return output_dir
    
    def train_all_datasets_separately(self):
        """分别训练所有数据集"""
        results = {}
        
        logger.info("=" * 60)
        logger.info("🚀 开始分别训练所有数据集")
        logger.info("=" * 60)
        
        for dataset_type in ["question_classifier", "question_type_classifier", "question_answer", "file_name_pick"]:
            try:
                logger.info(f"\n{'='*20} {dataset_type} {'='*20}")
                output_dir = self.train_single_dataset(dataset_type)
                results[dataset_type] = {
                    "status": "success",
                    "output_dir": output_dir
                }
            except Exception as e:
                logger.error(f"❌ {dataset_type} 训练失败: {e}")
                results[dataset_type] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return results
    
    def train_mixed_dataset(self):
        """训练混合数据集"""
        logger.info("=" * 60)
        logger.info("🚀 开始训练混合数据集")
        logger.info("=" * 60)
        
        return self.train_single_dataset("mixed")
    
    def get_available_datasets(self):
        """获取可用的数据集类型"""
        return list(self.training_configs.keys())
    
    def print_dataset_summary(self):
        """打印数据集摘要"""
        logger.info("=" * 60)
        logger.info("📊 数据集摘要")
        logger.info("=" * 60)
        
        for dataset_type in self.dataset_manager.datasets.keys():
            info = self.dataset_manager.get_dataset_info(dataset_type)
            config = self.training_configs[dataset_type]
            logger.info(f"{config['description']}: {info['total_samples']} 个样本")
        
        # 混合数据集信息
        mixed_info = self.dataset_manager.get_dataset_info("mixed")
        logger.info(f"混合数据集: {mixed_info['total_samples']} 个样本")
        logger.info(f"详细分布: {mixed_info['datasets']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PEFT LoRA 多模式指令微调训练器")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "all", "mixed", "summary"], 
        default="mixed",
        help="训练模式: single(单个数据集), all(所有数据集分别训练), mixed(混合训练), summary(显示数据集摘要)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["question_classifier", "question_type_classifier", "question_answer", "file_name_pick"],
        help="当mode为single时，指定要训练的数据集类型"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="自定义输出目录"
    )
    parser.add_argument(
        "--bert_model_path", 
        type=str, 
        default=r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea",
        help="BERT模型路径"
    )
    parser.add_argument(
        "--base_output_dir", 
        type=str, 
        default=r"E:\project\llm\lora",
        help="基础输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = MultiModeTrainer(
        bert_model_path=args.bert_model_path,
        base_output_dir=args.base_output_dir
    )
    
    print("=" * 60)
    print("🚀 PEFT LoRA 多模式指令微调训练系统")
    print("基于BERT-base-chinese的多任务指令微调")
    print("=" * 60)
    
    try:
        if args.mode == "summary":
            # 显示数据集摘要
            trainer.print_dataset_summary()
            
        elif args.mode == "single":
            # 训练单个数据集
            if not args.dataset:
                print("❌ 错误: single模式需要指定--dataset参数")
                print(f"可用的数据集类型: {trainer.get_available_datasets()}")
                return
            
            print(f"🎯 开始训练单个数据集: {args.dataset}")
            output_dir = trainer.train_single_dataset(args.dataset, args.output_dir)
            print(f"\n✅ 单个数据集训练完成！")
            print(f"💾 模型保存路径: {output_dir}")
            
        elif args.mode == "all":
            # 分别训练所有数据集
            print("🎯 开始分别训练所有数据集")
            results = trainer.train_all_datasets_separately()
            
            print("\n" + "=" * 60)
            print("📊 训练结果汇总")
            print("=" * 60)
            
            success_count = 0
            for dataset_type, result in results.items():
                if result["status"] == "success":
                    print(f"✅ {dataset_type}: 成功 - {result['output_dir']}")
                    success_count += 1
                else:
                    print(f"❌ {dataset_type}: 失败 - {result['error']}")
            
            print(f"\n🎉 训练完成！成功: {success_count}/{len(results)}")
            
        elif args.mode == "mixed":
            # 训练混合数据集
            print("🎯 开始训练混合数据集")
            output_dir = trainer.train_mixed_dataset()
            print(f"\n✅ 混合数据集训练完成！")
            print(f"💾 模型保存路径: {output_dir}")
        
        print("\n" + "=" * 60)
        print("🎉 所有训练任务完成！")
        print("💡 提示：可以使用推理脚本测试训练好的模型")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


# 兼容性函数，保持与原版本的兼容性
def prepare_all_data():
    """准备所有数据集并打乱（兼容性函数）"""
    manager = DatasetManager()
    return manager.get_dataset("mixed")


def train_model():
    """训练模型（兼容性函数）"""
    trainer = MultiModeTrainer()
    return trainer.train_mixed_dataset()


if __name__ == "__main__":
    # 检查是否有命令行参数
    import sys
    if len(sys.argv) == 1:
        # 没有命令行参数，使用默认的混合训练模式
        print("=" * 60)
        print("🚀 PEFT LoRA 数据平台训练系统")
        print("基于BERT-base-chinese的指令微调")
        print("=" * 60)
        
        try:
            trainer = MultiModeTrainer()
            trainer.print_dataset_summary()
            
            print("\n🎯 开始混合数据集训练...")
            model_path = trainer.train_mixed_dataset()
            
            print("\n" + "=" * 60)
            print("✅ 训练成功完成！")
            print(f"💾 模型保存路径: {model_path}")
            print("🎉 可以开始使用训练好的模型了！")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 有命令行参数，使用参数解析
        main()
