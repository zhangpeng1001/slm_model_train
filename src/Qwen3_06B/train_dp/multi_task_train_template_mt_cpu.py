"""
基于模型 tokenizer_config.json文件的内容，根据 chat_template 重构查询
"""
import json
import os
from typing import Dict, List, Tuple
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Trainer

# 配置路径
model_name = r"/mnt/workspace/base_model/Qwen3-0.6B"
output_model_path = r"/mnt/workspace/train_model/Qwen3-multi-task"

# 数据集配置
DATASET_CONFIG = {
    "data_extraction": {
        "file_path": r"/mnt/workspace/project/dataset/data_extraction.json",
        "system_prompt": "你是一个专业的数据提取助手。任务是分析用户输入的文本，提取用户描述的数据名称",
        "task_type": "extraction"
    },
    "dp_qa": {
        "file_path": r"/mnt/workspace/project/dataset/dp_qa.json",
        "system_prompt": "你是一个专业的数据平台问答助手。任务是分析用户输入的问题，并提供答案给用户",
        "task_type": "qa"
    },
    "question_classifier": {
        "file_path": r"/mnt/workspace/project/dataset/question_classifier.json",
        "system_prompt": "你是一个专业的问题分类助手。任务是分析用户输入的文本，判断用户的问题类型是（数据平台相关、通用对话、无关问题）中哪一个",
        "task_type": "classification"
    },
    "tool_data_platform": {
        "file_path": r"/mnt/workspace/project/dataset/tool_data_platform.json",
        "system_prompt": (
            "你是数据中台项目的工具调用助手，可以调用以下函数："
            "\n- get_data_collection(file_name: str)：用于数据采集工具;"
            "\n- query_data_by_filename(file_name: str)：用于文件信息查询工具;"
            "\n- data_warehousing(file_name: str,target_db_name:str)：用于数据入库工具;"
            "\n- data_service_publish(file_name: str)：用于数据发服务工具;"
            "\n- data_quality_check(file_name: str,check_type:str)：用于数据质检工具;"
            "\n请根据指令和输入,选择合适的函数并按指定格式调用。\n"
            "如果需要调用函数，请使用以下格式：\n<tool_call>\n{\"name\":\"函数名\",\"parameters\":{\"参数名\":参数值}}\n</tool_call>\n"
        ),
        "task_type": "function_calling"
    }
}


class MultiTaskTrainer:
    def __init__(self, model_name: str, output_path: str):
        self.model_name = model_name
        self.output_path = output_path
        self.tokenizer = None
        self.model = None
        # 从tokenizer获取chat_template的核心特殊token（确保与tokenizer_config一致）
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"  # 即tokenizer.eos_token
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"

    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("正在加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="cpu",  # 强制使用CPU
            torch_dtype="auto"  # 自动选择数据类型
        )

        # 配置LoRA - 针对CPU环境优化
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # 降低秩以减少计算量
            lora_alpha=16,  # 保持合理的alpha/r比例
            lora_dropout=0.1,  # 适度dropout防止过拟合
            target_modules=["q_proj", "v_proj"],  # 只针对关键模块
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def load_json_data(self, file_path: str) -> List[Dict]:
        """加载JSON数据"""
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过该数据集")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def prepare_dataset(self, task_name: str, config: Dict) -> Dataset:
        """为特定任务准备数据集"""
        print(f"正在准备 {task_name} 数据集...")

        # 加载原始数据
        raw_data = self.load_json_data(config["file_path"])
        if not raw_data:
            return None

        # 为每个样本添加任务标识和系统提示
        processed_data = []
        for item in raw_data:
            processed_item = {
                "task_name": task_name,
                "task_type": config["task_type"],
                "system_prompt": config["system_prompt"],
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "instruction": item.get("instruction", "")  # 保留原有instruction字段
            }
            processed_data.append(processed_item)

        return Dataset.from_list(processed_data)

    def preprocess_function(self, examples):
        inputs = []  # 对应“system + user”的完整输入（带角色标记）
        targets = []  # 对应“assistant”的输出（带角色标记和工具标签）

        for i in range(len(examples["input"])):
            # 提取单条样本的字段
            task_name = examples["task_name"][i]
            task_type = examples["task_type"][i]
            system_prompt = examples["system_prompt"][i].strip()  # 清理空行
            user_input = examples["input"][i].strip()
            expected_output = examples["output"][i].strip()

            # -------------------------- 1. 构建带角色标记的System + User输入 --------------------------
            # 遵循chat_template：<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>

            # 拼接system和user（chat_template的核心角色结构）
            prompt_with_role = (
                f"{self.im_start}system\n{system_prompt}{self.im_end}\n"
                f"{self.im_start}user\n{user_input}{self.im_end}\n"
                f"{self.im_start}assistant\n"  # 标记“接下来是模型需要学习的回复”
            )

            # -------------------------- 2. 构建带格式的Assistant目标输出 --------------------------
            # 根据任务类型，适配chat_template的输出格式（尤其是工具调用）
            target_with_format = f"{expected_output}{self.im_end}"
            # 加入列表
            inputs.append(prompt_with_role)
            targets.append(target_with_format)

        # -------------------------- 3. 构建完整训练文本（输入+目标） --------------------------
        # 格式：prompt_with_role（system+user+assistant开头） + target_with_format（assistant的回复）
        full_texts = [inp + tgt for inp, tgt in zip(inputs, targets)]
        # 无需额外加eos_token：因为target_with_format已包含<|im_end|>（而tokenizer.eos_token就是<|im_end|>）

        # -------------------------- 4. 编码（与原逻辑一致，但输入格式已对齐chat_template） --------------------------
        model_inputs = self.tokenizer(
            full_texts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False,
            return_length=False
        )

        # -------------------------- 5. 构建标签（修正输入长度计算逻辑） --------------------------
        # 关键：input_only需与prompt_with_role完全一致（带角色标记），才能准确分割输入/目标
        input_only_encodings = self.tokenizer(
            inputs,  # 这里的inputs是“system+user+assistant开头”的部分
            max_length=512,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False
        )

        labels = []
        for input_ids, full_input_ids in zip(input_only_encodings["input_ids"], model_inputs["input_ids"]):
            # 计算输入部分的有效长度（排除pad_token）
            input_len = 0
            for id in input_ids:
                if id == self.tokenizer.pad_token_id:
                    break
                input_len += 1
            # 输入部分（input_len之前）标记为-100（模型不学习），目标部分保留原id
            label = [-100] * input_len + full_input_ids[input_len:]
            # 确保标签长度与max_length一致
            label = label[:512] + [-100] * max(0, 512 - len(label))
            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs

    def prepare_multi_task_dataset(self) -> Dataset:
        """准备多任务数据集"""
        print("正在准备多任务数据集...")

        all_datasets = []
        for task_name, config in DATASET_CONFIG.items():
            dataset = self.prepare_dataset(task_name, config)
            if dataset is not None:
                print(f"{task_name} 数据集包含 {len(dataset)} 个样本")
                all_datasets.append(dataset)

        if not all_datasets:
            raise ValueError("没有成功加载任何数据集！")

        # 合并所有数据集
        combined_dataset = concatenate_datasets(all_datasets)
        print(f"合并后的数据集总共包含 {len(combined_dataset)} 个样本")

        # 打乱数据
        combined_dataset = combined_dataset.shuffle(seed=42)

        return combined_dataset

    def train(self):
        """执行多任务训练"""
        print("开始多任务训练...")

        # 加载模型和分词器
        self.load_model_and_tokenizer()

        # 准备数据集
        dataset = self.prepare_multi_task_dataset()

        # 预处理数据
        print("正在预处理数据...")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names  # 移除原始列
        )

        # 训练参数 - 针对CPU环境优化
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,  # CPU 友好
            num_train_epochs=10,  # 减少训练轮数
            logging_dir=None,  # 禁用日志
            report_to="none",  # 禁用日志
            save_strategy="no",  # 不保存中间结果
            learning_rate=2e-4,  # 更低学习率

            # gradient_accumulation_steps=8,  # 通过梯度累积增加有效批次大小
            # logging_steps=50,
            # save_total_limit=2,  # 只保留最近2个检查点
            # dataloader_num_workers=0,  # CPU环境不使用多进程
            # fp16=False,  # CPU不支持fp16
            # warmup_steps=100,  # 预热步数
            # weight_decay=0.01,  # 权重衰减
        )

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )

        # 开始训练
        print("开始训练...")
        trainer.train()

        # 保存模型
        print(f"保存模型到 {self.output_path}")
        self.model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)

        print("训练完成！")


def main():
    """主函数"""
    # 创建训练器
    trainer = MultiTaskTrainer(model_name, output_model_path)

    # 执行训练
    trainer.train()


if __name__ == '__main__':
    # 对于Windows系统，添加这个函数调用
    from multiprocessing import freeze_support

    freeze_support()

    # 调用主函数
    main()
