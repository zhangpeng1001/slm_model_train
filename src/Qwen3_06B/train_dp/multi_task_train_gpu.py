import json
import os
import torch
from typing import Dict, List, Tuple
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Trainer
import logging
import multiprocessing as mp
import gc

# 设置环境变量优化GPU内存
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置路径
model_name = r"/content/drive/MyDrive/models/Qwen3-0.6B"
output_model_path = r"/content/drive/MyDrive/train-models/Qwen3-multi-task-GPU"

# 数据集配置
DATASET_CONFIG = {
    "data_extraction": {
        "file_path": r"/content/drive/MyDrive/dp_dataset/data_extraction.json",
        "system_prompt": "你是一个专业的数据提取助手。任务是分析用户输入的文本，提取用户描述的数据名称",
        "task_type": "extraction"
    },
    "dp_qa": {
        "file_path": r"/content/drive/MyDrive/dp_dataset/dp_qa.json",
        "system_prompt": "你是一个专业的数据平台问答助手。任务是分析用户输入的问题，并提供答案给用户",
        "task_type": "qa"
    },
    "question_classifier": {
        "file_path": r"/content/drive/MyDrive/dp_dataset/question_classifier.json",
        "system_prompt": "你是一个专业的问题分类助手。任务是分析用户输入的文本，判断用户的问题类型是（数据平台相关、通用对话、无关问题）中哪一个",
        "task_type": "classification"
    },
    "question_type_classifier": {
        "file_path": r"/content/drive/MyDrive/dp_dataset/question_type_classifier.json",
        "system_prompt": "你是一个专业的问题类型分类助手。任务是分析用户输入的文本，判断用户的问题类型是（问题回答、任务处理）中哪一个",
        "task_type": "classification"
    },
    "tool_data_platform": {
        "file_path": r"/content/drive/MyDrive/dp_dataset/tool_data_platform.json",
        "system_prompt": (
            "你是数据中台项目的工具调用助手，可以调用以下函数："
            "\n- get_data_collection(data_source: str,data_type: str,time_range: str,business_platform: str)：用于数据采集工具;"
            "\n- query_data_by_filename(filename: str,query_content: str)：用于文件名查数据工具;"
            "\n- data_warehousing(source_data_path: str,target_db_type: str,target_db: str,target_table: str,order_detail:str)：用于数据入库工具;"
            "\n- data_service_publish(source_db_type: str,source_db: str,dw_sales: str,sales_summary: str,data_filter:str,service_type:str,authorization:str)：用于数据发服务工具;"
            "\n- data_quality_check(source_db_type: str,source_db: str,source_table: str,check_dimensions: str)：用于数据质检工具;"
            "\n- data_cleaning(source_data_path: str,source_data_type: str,clean_rules: str,target_save_path: str)：用于数据清洗工具;"
            "\n请根据指令和输入,选择合适的函数并按指定格式调用。\n"
            "如果需要调用函数，请使用以下格式：\n<FunctionCall>\n{\"name\":\"函数名\",\"parameters\":{\"参数名\":参数值}}\n</FunctionCall>\n"
        ),
        "task_type": "function_calling"
    }
}


# ---------------------- 1. 把preprocess_function改为独立函数 ----------------------
def preprocess_function(examples, tokenizer, device, max_length):
    """独立的预处理函数，不依赖Trainer实例属性"""
    inputs = []
    targets = []

    for i in range(len(examples["input"])):
        # 逻辑与之前一致，但不再使用self.xxx，而是从参数获取
        task_name = examples["task_name"][i]
        task_type = examples["task_type"][i]
        system_prompt = examples["system_prompt"][i]
        user_input = examples["input"][i]
        expected_output = examples["output"][i]
        instruction = examples.get("instruction", [""] * len(examples["input"]))[i]

        # 构建prompt的逻辑完全不变
        if task_type == "function_calling":
            prompt = system_prompt + f"\nInput: {user_input}\nOutput: "
        elif task_type == "classification":
            prompt = system_prompt + f"\n问题: {user_input}\n分类结果: "
        elif task_type == "extraction":
            prompt = system_prompt + f"\n输入文本: {user_input}\n提取的数据名称: "
        elif task_type == "qa":
            if instruction:
                prompt = system_prompt + f"\nInstruction: {instruction}\nInput: {user_input}\nOutput: "
            else:
                prompt = system_prompt + f"\n问题: {user_input}\n回答: "
        else:
            prompt = system_prompt + f"\nInput: {user_input}\nOutput: "

        inputs.append(prompt)
        targets.append(expected_output)

    # 拼接输入和目标输出
    full_texts = [f"{inp}{tgt}{tokenizer.eos_token}" for inp, tgt in zip(inputs, targets)]

    # 编码（使用传递的tokenizer和max_length）
    model_inputs = tokenizer(
        full_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=False,
        return_length=False
    )

    # 构建标签（逻辑不变）
    input_only = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    labels = []
    for input_ids, full_input_ids in zip(input_only["input_ids"], model_inputs["input_ids"]):
        input_len = len([id for id in input_ids if id != tokenizer.pad_token_id])
        label = [-100] * input_len + full_input_ids[input_len:]
        label = label[:max_length] + [-100] * max(0, max_length - len(label))
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs


class MultiTaskTrainer:
    def __init__(self, model_name: str, output_path: str):
        self.model_name = model_name
        self.output_path = output_path
        self.tokenizer = None
        self.model = None
        self.device = self._setup_device()

    def _setup_device(self):
        """设置GPU设备"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

            # 清理GPU缓存
            torch.cuda.empty_cache()
            gc.collect()

            # 设置GPU内存分配策略 - 更保守的内存使用
            torch.cuda.set_per_process_memory_fraction(0.85)  # 使用85%的GPU内存，留更多余量

        else:
            device = torch.device("cpu")
            logger.warning("CUDA不可用，使用CPU训练")

        return device

    def load_model_and_tokenizer(self):
        """加载模型和分词器 - GPU优化版本"""
        logger.info("正在加载模型和分词器...")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"  # 确保padding在右侧
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型 - 内存优化版本
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,  # GPU使用fp16
            device_map="auto" if self.device.type == "cuda" else None,  # 自动设备映射
            low_cpu_mem_usage=True,  # 减少CPU内存使用
            use_cache=False,  # 禁用缓存以节省内存
        )

        # 配置LoRA - 内存优化版本
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # 降低秩以减少内存使用
            lora_alpha=16,  # 相应调整alpha
            lora_dropout=0.1,  # 增加dropout
            target_modules=["q_proj", "v_proj"],  # 减少目标模块
            bias="none",
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # 确保模型在正确的设备上
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        
        # 确保模型参数需要梯度
        self.model.train()
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad_(True)
        
        # 启用梯度检查点
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def load_json_data(self, file_path: str) -> List[Dict]:
        """加载JSON数据"""
        if not os.path.exists(file_path):
            logger.warning(f"文件 {file_path} 不存在，跳过该数据集")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"成功加载 {len(data)} 条数据从 {file_path}")
            return data
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {e}")
            return []

    def prepare_dataset(self, task_name: str, config: Dict) -> Dataset:
        """为特定任务准备数据集"""
        logger.info(f"正在准备 {task_name} 数据集...")

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
        """统一的预处理函数，支持多任务 - GPU优化版本"""
        inputs = []
        targets = []

        for i in range(len(examples["input"])):
            task_name = examples["task_name"][i]
            task_type = examples["task_type"][i]
            system_prompt = examples["system_prompt"][i]
            user_input = examples["input"][i]
            expected_output = examples["output"][i]
            instruction = examples.get("instruction", [""] * len(examples["input"]))[i]

            # 根据任务类型构建不同的输入格式
            if task_type == "function_calling":
                # 工具调用任务
                prompt = system_prompt + f"\nInput: {user_input}\nOutput: "
            elif task_type == "classification":
                # 分类任务
                prompt = system_prompt + f"\n问题: {user_input}\n分类结果: "
            elif task_type == "extraction":
                # 数据提取任务
                prompt = system_prompt + f"\n输入文本: {user_input}\n提取的数据名称: "
            elif task_type == "qa":
                # 问答任务
                if instruction:
                    prompt = system_prompt + f"\nInstruction: {instruction}\nInput: {user_input}\nOutput: "
                else:
                    prompt = system_prompt + f"\n问题: {user_input}\n回答: "
            else:
                # 默认格式
                prompt = system_prompt + f"\nInput: {user_input}\nOutput: "

            inputs.append(prompt)
            targets.append(expected_output)

        # 拼接输入和目标输出
        full_texts = [f"{inp}{tgt}{self.tokenizer.eos_token}" for inp, tgt in zip(inputs, targets)]

        # 编码 - GPU优化：增加最大长度
        max_length = 1024 if self.device.type == "cuda" else 512  # GPU环境使用更长序列
        model_inputs = self.tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False,
            return_length=False
        )

        # 构建标签：输入部分标记为-100
        input_only = self.tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        labels = []
        for input_ids, full_input_ids in zip(input_only["input_ids"], model_inputs["input_ids"]):
            # 找到输入部分的结束位置
            input_len = len([id for id in input_ids if id != self.tokenizer.pad_token_id])
            # 输入部分标签设为-100，目标部分保留原id
            label = [-100] * input_len + full_input_ids[input_len:]
            # 确保长度正确
            label = label[:max_length] + [-100] * max(0, max_length - len(label))
            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs

    def prepare_multi_task_dataset(self) -> Dataset:
        """准备多任务数据集"""
        logger.info("正在准备多任务数据集...")

        all_datasets = []
        for task_name, config in DATASET_CONFIG.items():
            dataset = self.prepare_dataset(task_name, config)
            if dataset is not None:
                logger.info(f"{task_name} 数据集包含 {len(dataset)} 个样本")
                all_datasets.append(dataset)

        if not all_datasets:
            raise ValueError("没有成功加载任何数据集！")

        # 合并所有数据集
        combined_dataset = concatenate_datasets(all_datasets)
        logger.info(f"合并后的数据集总共包含 {len(combined_dataset)} 个样本")

        # 打乱数据
        combined_dataset = combined_dataset.shuffle(seed=42)

        return combined_dataset

    def get_training_args(self) -> TrainingArguments:
        """获取训练参数 - GPU优化版本"""

        # 定义worker初始化函数
        def worker_init_fn(worker_id):
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            # 可以添加其他初始化逻辑，如设置随机种子
            import random
            random.seed(42 + worker_id)

        if self.device.type == "cuda":
            # 内存优化参数 - 大幅减少内存使用
            return TrainingArguments(
                output_dir="./results",
                per_device_train_batch_size=2,  # 大幅减少batch_size
                gradient_accumulation_steps=8,  # 增加梯度累积步数保持有效batch_size
                num_train_epochs=3,
                learning_rate=5e-5,
                warmup_steps=200,  # 减少预热步数
                logging_steps=50,
                save_steps=500,
                save_total_limit=2,
                eval_strategy="no",
                save_strategy="steps",
                fp16=True,  # 启用混合精度训练
                dataloader_num_workers=0,
                dataloader_pin_memory=False,  # 禁用pin_memory节省内存
                remove_unused_columns=False,
                load_best_model_at_end=False,
                metric_for_best_model="loss",
                greater_is_better=False,
                report_to="none",
                run_name=f"qwen3-multi-task-memory-optimized",
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                max_grad_norm=1.0,
                lr_scheduler_type="cosine",
                optim="adamw_torch",
                ddp_find_unused_parameters=False,  # 禁用查找未使用参数
            )
        else:
            # CPU环境参数
            return TrainingArguments(
                output_dir="./results",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                num_train_epochs=5,
                learning_rate=2e-4,
                warmup_steps=100,
                logging_steps=50,
                save_strategy="no",
                fp16=False,
                dataloader_num_workers=0,
                # dataloader_worker_init_fn=worker_init_fn,  # CPU版本也添加
                report_to="none",
                weight_decay=0.01,
            )

    def train(self):
        """执行多任务训练 - GPU优化版本"""
        logger.info("开始多任务训练...")

        # 加载模型和分词器（主进程中执行，正常）
        self.load_model_and_tokenizer()

        # 准备数据集（无CUDA操作，正常）
        dataset = self.prepare_multi_task_dataset()

        # 预处理数据：调用独立函数，通过fn_kwargs传递参数
        logger.info("正在预处理数据...")
        # 大幅减少序列长度以节省内存
        max_length = 512 if self.device.type == "cuda" else 256
        tokenized_dataset = dataset.map(
            preprocess_function,  # 独立函数
            batched=True,
            batch_size=100 if self.device.type == "cuda" else 50,  # 减少批处理大小
            num_proc=1,  # 单进程避免内存冲突
            remove_columns=dataset.column_names,
            desc="预处理数据集",
            # 传递必要参数，避免序列化Trainer实例
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "device": self.device,
                "max_length": max_length
            }
        )

        # 获取训练参数
        training_args = self.get_training_args()

        # 创建数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if self.device.type == "cuda" else None,  # GPU优化
        )

        # 创建Trainer时添加 worker_init_fn
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # GPU内存监控
        if self.device.type == "cuda":
            logger.info(f"训练前GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            logger.info(f"训练前GPU内存缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

        # 开始训练
        logger.info("开始训练...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU内存不足！尝试减少batch_size或使用梯度检查点")
                # 可以在这里实现自动降低batch_size的逻辑
                raise
            else:
                raise

        # 训练后的GPU内存信息
        if self.device.type == "cuda":
            logger.info(f"训练后GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            logger.info(f"训练后GPU内存缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

        # 保存模型
        logger.info(f"保存模型到 {self.output_path}")
        os.makedirs(self.output_path, exist_ok=True)
        self.model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)

        # 清理GPU缓存
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("训练完成！")

    def monitor_gpu_memory(self):
        """监控GPU内存使用情况"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB, 总计: {total:.2f}GB")


def main():
    """主函数"""
    logger.info("=== Qwen3 多任务训练 (Linux GPU优化版) ===")

    # 检查CUDA可用性
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA不可用，将使用CPU训练")

    # 创建训练器
    trainer = MultiTaskTrainer(model_name, output_model_path)

    # 执行训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    # 关键：添加 force=True，强制设置启动方式（覆盖已有设置）
    try:
        mp.set_start_method('spawn', force=True)
        logger.info("多进程启动方式已设置为 'spawn'")
    except RuntimeError as e:
        logger.warning(f"设置多进程启动方式时警告: {e}")
    main()
