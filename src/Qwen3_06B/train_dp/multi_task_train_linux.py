import json
import os
from typing import Dict, List, Tuple
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from transformers import Trainer

# 配置路径
model_name = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
output_model_path = r"E:\project\llm\model-data\train-models\Qwen3-multi-task"

# 数据集配置
DATASET_CONFIG = {
    "data_extraction": {
        "file_path": r"E:\project\python\slm_model_train\src\dataset\data_platform\data_extraction.json",
        "system_prompt": "你是一个专业的数据提取助手。任务是分析用户输入的文本，提取用户描述的数据名称",
        "task_type": "extraction"
    },
    "dp_qa": {
        "file_path": r"E:\project\python\slm_model_train\src\dataset\data_platform\dp_qa.json",
        "system_prompt": "你是一个专业的数据平台问答助手。任务是分析用户输入的问题，并提供答案给用户",
        "task_type": "qa"
    },
    "question_classifier": {
        "file_path": r"E:\project\python\slm_model_train\src\dataset\data_platform\question_classifier.json",
        "system_prompt": "你是一个专业的问题分类助手。任务是分析用户输入的文本，判断用户的问题类型是（数据平台相关、通用对话、无关问题）中哪一个",
        "task_type": "classification"
    },
    "question_type_classifier": {
        "file_path": r"E:\project\python\slm_model_train\src\dataset\data_platform\question_type_classifier.json",
        "system_prompt": "你是一个专业的问题类型分类助手。任务是分析用户输入的文本，判断用户的问题类型是（问题回答、任务处理）中哪一个",
        "task_type": "classification"
    },
    "tool_data_platform": {
        "file_path": r"E:\project\python\slm_model_train\src\dataset\data_platform\tool_data_platform.json",
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


class MultiTaskTrainer:
    def __init__(self, model_name: str, output_path: str):
        self.model_name = model_name
        self.output_path = output_path
        self.tokenizer = None
        self.model = None
        
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
        """统一的预处理函数，支持多任务"""
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
        # print(f"full_texts:{full_texts}")
        # 编码
        model_inputs = self.tokenizer(
            full_texts,
            max_length=512,  # 增加最大长度以适应复杂任务
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False,
            return_length=False
        )
        
        # 构建标签：输入部分标记为-100
        input_only = self.tokenizer(
            inputs,
            max_length=512,
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
