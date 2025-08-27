"""
推理配置文件
用于配置模型推理的各种参数
"""

import os

# ============= 模型路径配置 =============
# 基础模型路径 (请根据实际情况修改)
BASE_MODEL_PATH = "/home/user/models/Qwen3-0.6B"

# 训练后的LoRA模型路径 (请根据实际情况修改)
PEFT_MODEL_PATH = "/home/user/train-models/Qwen3-multi-task"

# ============= 推理设备配置 =============
# 推理设备选择: "auto", "cuda", "cpu"
INFERENCE_DEVICE = "auto"

# ============= 生成参数配置 =============
# 不同任务的生成参数
GENERATION_CONFIGS = {
    "data_extraction": {
        "max_length": 256,
        "temperature": 0.3,
        "top_p": 0.8,
        "do_sample": False,
        "repetition_penalty": 1.1,
    },
    "dp_qa": {
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1,
    },
    "question_classifier": {
        "max_length": 64,
        "temperature": 0.1,
        "top_p": 0.5,
        "do_sample": False,
        "repetition_penalty": 1.0,
    },
    "question_type_classifier": {
        "max_length": 64,
        "temperature": 0.1,
        "top_p": 0.5,
        "do_sample": False,
        "repetition_penalty": 1.0,
    },
    "tool_data_platform": {
        "max_length": 256,
        "temperature": 0.3,
        "top_p": 0.8,
        "do_sample": False,
        "repetition_penalty": 1.1,
    }
}

# ============= 默认生成参数 =============
DEFAULT_GENERATION_CONFIG = {
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
}

# ============= 任务映射配置 =============
TASK_MAPPING = {
    # 任务别名映射到标准任务名
    "extraction": "data_extraction",
    "qa": "dp_qa",
    "classify": "question_classifier",
    "type_classify": "question_type_classifier",
    "tool": "tool_data_platform",
    "function_calling": "tool_data_platform",
}

# ============= 批处理配置 =============
BATCH_CONFIG = {
    "batch_size": 4,  # 批处理大小
    "max_batch_size": 8,  # 最大批处理大小
}

# ============= 缓存配置 =============
CACHE_CONFIG = {
    "enable_cache": True,  # 是否启用缓存
    "cache_size": 1000,    # 缓存大小
    "cache_ttl": 3600,     # 缓存过期时间(秒)
}

# ============= 日志配置 =============
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None,  # 日志文件路径，None表示只输出到控制台
}

# ============= 安全配置 =============
SECURITY_CONFIG = {
    "max_input_length": 2048,  # 最大输入长度
    "max_output_length": 1024, # 最大输出长度
    "filter_sensitive": False, # 是否过滤敏感内容
}

# ============= 性能配置 =============
PERFORMANCE_CONFIG = {
    "use_cache": True,          # 使用KV缓存
    "torch_compile": False,     # 是否使用torch.compile (需要PyTorch 2.0+)
    "attention_implementation": "eager",  # 注意力实现方式
}

def get_generation_config(task_type: str) -> dict:
    """
    获取指定任务的生成配置
    
    Args:
        task_type: 任务类型
        
    Returns:
        生成配置字典
    """
    # 处理任务别名
    task_type = TASK_MAPPING.get(task_type, task_type)
    
    # 获取任务特定配置，如果不存在则使用默认配置
    config = GENERATION_CONFIGS.get(task_type, DEFAULT_GENERATION_CONFIG.copy())
    
    return config

def validate_paths():
    """验证路径配置"""
    issues = []
    
    if not os.path.exists(BASE_MODEL_PATH):
        issues.append(f"基础模型路径不存在: {BASE_MODEL_PATH}")
    
    if not os.path.exists(PEFT_MODEL_PATH):
        issues.append(f"LoRA模型路径不存在: {PEFT_MODEL_PATH}")
    
    return issues

def check_config():
    """检查配置"""
    print("=== 推理配置检查 ===")
    print(f"基础模型路径: {BASE_MODEL_PATH}")
    print(f"LoRA模型路径: {PEFT_MODEL_PATH}")
    print(f"推理设备: {INFERENCE_DEVICE}")
    
    # 验证路径
    issues = validate_paths()
    if issues:
        print("\n⚠️  发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ 配置检查通过")
        return True

if __name__ == "__main__":
    check_config()
