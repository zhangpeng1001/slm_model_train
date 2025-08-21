"""
基于LoRA的PEFT库对Qwen3-0.6B模型进行指令微调
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset.dataset import question_classifier, question_type_classifier, question_answer, file_name_pick


class QwenLoRAFineTuner:
    def __init__(self,
                 base_model_path: str,
                 output_model_path: str,
                 max_length: int = 512):
        """
        初始化Qwen LoRA微调器
        
        Args:
            base_model_path: 基础模型路径
            output_model_path: 输出模型路径
            max_length: 最大序列长度
        """
        self.base_model_path = base_model_path
        self.output_model_path = output_model_path
        self.max_length = max_length

        # 确保输出目录存在
        os.makedirs(output_model_path, exist_ok=True)

        print(f"基础模型路径: {base_model_path}")
        print(f"输出模型路径: {output_model_path}")

    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("正在加载模型和分词器...")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # 准备模型进行量化训练
        self.model = prepare_model_for_kbit_training(self.model)

        print("模型和分词器加载完成!")

    def setup_lora_config(self):
        """设置LoRA配置"""
        print("设置LoRA配置...")

        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # LoRA rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,  # LoRA dropout
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],  # Qwen3模型的注意力层
            bias="none",
        )

        # 应用LoRA到模型
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        print("LoRA配置完成!")

    def prepare_dataset(self):
        """准备训练数据集"""
        print("准备训练数据集...")

        # 合并所有数据集
        all_data = []
        all_data.extend(question_classifier)
        all_data.extend(question_type_classifier)
        all_data.extend(question_answer)
        all_data.extend(file_name_pick)

        print(f"总训练样本数: {len(all_data)}")

        # 格式化数据为对话格式
        formatted_data = []
        for item in all_data:
            # 构建对话格式
            conversation = [
                {"role": "user", "content": f"任务: {item['instruction']}\n输入: {item['input']}"},
                {"role": "assistant", "content": item['output']}
            ]
            formatted_data.append({"conversation": conversation})

        # 创建Dataset对象
        dataset = Dataset.from_list(formatted_data)

        # 数据预处理函数
        def preprocess_function(examples):
            inputs = []
            labels = []

            for conversation in examples["conversation"]:
                # 应用聊天模板
                text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False
                )

                # 分词
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )

                inputs.append(tokenized["input_ids"].squeeze())
                labels.append(tokenized["input_ids"].squeeze().clone())

            return {
                "input_ids": inputs,
                "labels": labels,
                "attention_mask": [torch.ones_like(input_id) for input_id in inputs]
            }

        # 应用预处理
        self.train_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        print("数据集准备完成!")

    def train(self):
        """开始训练"""
        print("开始训练...")

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_model_path,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            learning_rate=2e-4,
            fp16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # 不使用wandb等工具
        )

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_model_path)

        print(f"训练完成! 模型已保存到: {self.output_model_path}")

    def run_full_training(self):
        """执行完整的训练流程"""
        print("=" * 50)
        print("开始Qwen3-0.6B LoRA微调训练")
        print("=" * 50)

        # 1. 加载模型和分词器
        self.load_model_and_tokenizer()

        # 2. 设置LoRA配置
        self.setup_lora_config()

        # 3. 准备数据集
        self.prepare_dataset()

        # 4. 开始训练
        self.train()

        print("=" * 50)
        print("训练流程完成!")
        print("=" * 50)


def main():
    """主函数"""
    # 配置路径
    base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
    output_model_path = r"E:\project\llm\model-data\train-models\Qwen3-0.6B"

    # 检查基础模型路径是否存在
    if not os.path.exists(base_model_path):
        print(f"错误: 基础模型路径不存在: {base_model_path}")
        return

    # 创建微调器并开始训练
    fine_tuner = QwenLoRAFineTuner(
        base_model_path=base_model_path,
        output_model_path=output_model_path,
        max_length=512
    )

    # 执行完整训练流程
    fine_tuner.run_full_training()


if __name__ == "__main__":
    main()
