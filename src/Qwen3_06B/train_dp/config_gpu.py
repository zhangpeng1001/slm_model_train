"""
GPU训练配置文件
适用于T4 GPU环境的优化配置
"""

import os

# ============= 基础配置 =============
# 模型路径配置 (请根据实际情况修改)
MODEL_NAME = "/home/user/models/Qwen3-0.6B"
OUTPUT_MODEL_PATH = "/home/user/train-models/Qwen3-multi-task"

# 数据集路径配置 (请根据实际情况修改)
DATASET_BASE_PATH = "/home/user/dp_dataset"

# ============= GPU优化配置 =============
# T4 GPU内存约16GB，以下配置针对T4优化
GPU_CONFIG = {
    "memory_fraction": 0.9,  # 使用90%的GPU内存
    "batch_size": 8,         # T4适合的批处理大小
    "gradient_accumulation_steps": 4,  # 梯度累积步数
    "max_length": 1024,      # 最大序列长度
    "fp16": True,           # 使用混合精度训练
    "dataloader_workers": 4, # 数据加载进程数
}

# ============= LoRA配置 =============
LORA_CONFIG = {
    "r": 16,                # LoRA秩，GPU环境可以使用更高值
    "lora_alpha": 32,       # LoRA alpha参数
    "lora_dropout": 0.05,   # LoRA dropout
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],  # 目标模块
    "bias": "none",
}

# ============= 训练参数配置 =============
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_torch",
    "logging_steps": 100,
    "save_steps": 1000,
    "save_total_limit": 3,
}

# ============= 数据集配置 =============
DATASET_CONFIG = {
    "data_extraction": {
        "file_path": os.path.join(DATASET_BASE_PATH, "data_extraction.json"),
        "system_prompt": "你是一个专业的数据提取助手。任务是分析用户输入的文本，提取用户描述的数据名称",
        "task_type": "extraction"
    },
    "dp_qa": {
        "file_path": os.path.join(DATASET_BASE_PATH, "dp_qa.json"),
        "system_prompt": "你是一个专业的数据平台问答助手。任务是分析用户输入的问题，并提供答案给用户",
        "task_type": "qa"
    },
    "question_classifier": {
        "file_path": os.path.join(DATASET_BASE_PATH, "question_classifier.json"),
        "system_prompt": "你是一个专业的问题分类助手。任务是分析用户输入的文本，判断用户的问题类型是（数据平台相关、通用对话、无关问题）中哪一个",
        "task_type": "classification"
    },
    "question_type_classifier": {
        "file_path": os.path.join(DATASET_BASE_PATH, "question_type_classifier.json"),
        "system_prompt": "你是一个专业的问题类型分类助手。任务是分析用户输入的文本，判断用户的问题类型是（问题回答、任务处理）中哪一个",
        "task_type": "classification"
    },
    "tool_data_platform": {
        "file_path": os.path.join(DATASET_BASE_PATH, "tool_data_platform.json"),
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

# ============= 环境检查函数 =============
def check_environment():
    """检查训练环境"""
    import torch
    
    print("=== 环境检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 检查路径
    print("\n=== 路径检查 ===")
    print(f"模型路径: {MODEL_NAME}")
    print(f"模型存在: {os.path.exists(MODEL_NAME)}")
    print(f"输出路径: {OUTPUT_MODEL_PATH}")
    print(f"数据集路径: {DATASET_BASE_PATH}")
    print(f"数据集路径存在: {os.path.exists(DATASET_BASE_PATH)}")
    
    # 检查数据集文件
    print("\n=== 数据集文件检查 ===")
    for task_name, config in DATASET_CONFIG.items():
        file_path = config["file_path"]
        exists = os.path.exists(file_path)
        print(f"{task_name}: {file_path} - {'存在' if exists else '不存在'}")


if __name__ == "__main__":
    check_environment()
