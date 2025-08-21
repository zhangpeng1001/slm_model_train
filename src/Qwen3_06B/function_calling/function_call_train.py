"""
基于LoRA的PEFT库对Qwen3-0.6B模型进行Function Calling微调
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

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.dataset.function_calling import function_calling_dataset


class QwenFunctionCallingFineTuner:
    def __init__(self,
                 base_model_path: str,
                 output_model_path: str,
                 max_length: int = 1024):
        """
        初始化Qwen Function Calling微调器
        
        Args:
            base_model_path: 基础模型路径
            output_model_path: 输出模型路径
            max_length: 最大序列长度（Function Calling需要更长的序列）
        """
        self.base_model_path = base_model_path
        self.output_model_path = output_model_path
        self.max_length = max_length

        # 确保输出目录存在
        os.makedirs(output_model_path, exist_ok=True)

        print(f"基础模型路径: {base_model_path}")
        print(f"输出模型路径: {output_model_path}")
        print(f"最大序列长度: {max_length}")

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

        # LoRA配置 - Function Calling需要更高的rank来处理复杂的工具调用
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # 提高rank以更好地学习function calling
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

    def format_function_calling_data(self, sample):
        """
        将Function Calling数据格式化为训练格式
        
        Args:
            sample: 单个训练样本
            
        Returns:
            formatted_text: 格式化后的文本
        """
        messages = sample["messages"]
        tools = sample.get("tools", [])
        
        # 构建系统消息，包含工具定义
        system_content = "You are a helpful assistant that can use tools to help users."
        if tools:
            system_content += "\n\nAvailable tools:\n"
            for tool in tools:
                tool_info = tool["function"]
                system_content += f"- {tool_info['name']}: {tool_info['description']}\n"
                system_content += f"  Parameters: {json.dumps(tool_info['parameters'], ensure_ascii=False)}\n"
        
        # 构建完整的对话
        formatted_messages = [{"role": "system", "content": system_content}]
        
        for msg in messages:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                # 处理包含工具调用的助手消息
                content = msg["content"]
                tool_calls = msg["tool_calls"]
                
                # 构建工具调用格式
                tool_calls_text = ""
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"]["arguments"]
                    tool_calls_text += f"\n<tool_call>\n{func_name}({func_args})\n</tool_call>"
                
                formatted_content = content + tool_calls_text
                formatted_messages.append({"role": "assistant", "content": formatted_content})
            else:
                formatted_messages.append(msg)
        
        return formatted_messages

    def prepare_dataset(self):
        """准备训练数据集"""
        print("准备Function Calling训练数据集...")

        # 使用导入的function calling数据集
        raw_data = function_calling_dataset
        print(f"原始数据样本数: {len(raw_data)}")

        # 格式化数据
        formatted_data = []
        for sample in raw_data:
            try:
                formatted_messages = self.format_function_calling_data(sample)
                formatted_data.append({"conversation": formatted_messages})
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue

        print(f"格式化后样本数: {len(formatted_data)}")

        # 创建Dataset对象
        dataset = Dataset.from_list(formatted_data)

        # 数据预处理函数
        def preprocess_function(examples):
            inputs = []
            labels = []

            for conversation in examples["conversation"]:
                try:
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
                except Exception as e:
                    print(f"分词时出错: {e}")
                    continue

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

        print("Function Calling数据集准备完成!")

    def train(self):
        """开始训练"""
        print("开始Function Calling微调训练...")

        # 训练参数 - 针对Function Calling优化
        training_args = TrainingArguments(
            output_dir=self.output_model_path,
            num_train_epochs=5,  # Function Calling需要更多轮次
            per_device_train_batch_size=1,  # 由于序列较长，减小batch size
            gradient_accumulation_steps=8,  # 增加梯度累积步数
            warmup_steps=200,
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
            learning_rate=1e-4,  # 稍微降低学习率
            fp16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # 不使用wandb等工具
            max_grad_norm=1.0,  # 梯度裁剪
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

        print(f"Function Calling微调训练完成! 模型已保存到: {self.output_model_path}")

    def run_full_training(self):
        """执行完整的训练流程"""
        print("=" * 60)
        print("开始Qwen3-0.6B Function Calling LoRA微调训练")
        print("=" * 60)

        # 1. 加载模型和分词器
        self.load_model_and_tokenizer()

        # 2. 设置LoRA配置
        self.setup_lora_config()

        # 3. 准备数据集
        self.prepare_dataset()

        # 4. 开始训练
        self.train()

        print("=" * 60)
        print("Function Calling微调训练流程完成!")
        print("=" * 60)

    def test_model(self, test_prompt: str = None):
        """测试微调后的模型"""
        if test_prompt is None:
            test_prompt = "今天北京的天气怎么样？"
        
        print(f"\n测试提示: {test_prompt}")
        
        # 构建测试消息
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant that can use tools to help users.\n\nAvailable tools:\n- get_weather: 获取指定地点的天气信息\n  Parameters: {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"城市名称\"}, \"date\": {\"type\": \"string\", \"description\": \"日期，可以是today、tomorrow或具体日期\"}}, \"required\": [\"location\"]}"},
            {"role": "user", "content": test_prompt}
        ]
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # 分词
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码回复
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"模型回复: {response}")


def main():
    """主函数"""
    # 配置路径
    base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
    output_model_path = r"E:\project\llm\model-data\train-models\Qwen3-function-calling"

    # 检查基础模型路径是否存在
    if not os.path.exists(base_model_path):
        print(f"错误: 基础模型路径不存在: {base_model_path}")
        print("请确保Qwen3-0.6B模型已下载到指定路径")
        return

    # 创建Function Calling微调器并开始训练
    fine_tuner = QwenFunctionCallingFineTuner(
        base_model_path=base_model_path,
        output_model_path=output_model_path,
        max_length=1024  # Function Calling需要更长的上下文
    )

    # 执行完整训练流程
    fine_tuner.run_full_training()
    
    # 训练完成后进行简单测试
    print("\n" + "=" * 60)
    print("进行Function Calling能力测试")
    print("=" * 60)
    
    # 测试几个不同的场景
    test_cases = [
        "今天北京的天气怎么样？",
        "帮我计算一下 25 * 4 + 10",
        "现在几点了？",
        "搜索一下人工智能的最新发展"
    ]
    
    for test_case in test_cases:
        fine_tuner.test_model(test_case)
        print("-" * 40)


if __name__ == "__main__":
    main()
