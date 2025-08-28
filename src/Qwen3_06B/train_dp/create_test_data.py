import json
import os
from typing import Dict, List

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


def load_json_data(file_path: str) -> List[Dict]:
    """加载JSON数据"""
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，跳过该数据集")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def prepare_dataset(task_name: str, config: Dict):
    """为特定任务准备数据集"""
    print(f"正在准备 {task_name} 数据集...")

    # 加载原始数据
    raw_data = load_json_data(config["file_path"])
    if not raw_data:
        return None

    # 为每个样本添加任务标识和系统提示
    processed_data = []
    for item in raw_data:
        # 根据任务类型构建不同的输入格式
        if config["task_type"] == "function_calling":
            # 工具调用任务
            prompt = config["system_prompt"] + f"\nInput: {item.get("input", "")}\nOutput: "
        elif config["task_type"] == "classification":
            # 分类任务
            prompt = config["system_prompt"] + f"\n问题: {item.get("input", "")}\n分类结果: "
        elif config["task_type"] == "extraction":
            # 数据提取任务
            prompt = config["system_prompt"] + f"\n输入文本: {item.get("input", "")}\n提取的数据名称: "
        elif config["task_type"] == "qa":
            # 问答任务
            if item.get("instruction", ""):
                prompt = config[
                             "system_prompt"] + f"\nInstruction: {item.get("instruction", "")}\nInput: {item.get("input", "")}\nOutput: "
            else:
                prompt = config["system_prompt"] + f"\n问题: {item.get("input", "")}\n回答: "
        else:
            # 默认格式
            prompt = config["system_prompt"] + f"\nInput: {item.get("input", "")}\nOutput: "
        processed_data.append(prompt)

    return processed_data


def prepare_multi_task_dataset():
    """准备多任务数据集"""
    for task_name, config in DATASET_CONFIG.items():
        dataset = prepare_dataset(task_name, config)
        print(f"数据集包含 {len(dataset)} 个样本")
        print(f"{json.dumps(dataset[5], ensure_ascii=False, indent=2)} ")
        print(f"{json.dumps(dataset[6], ensure_ascii=False, indent=2)} ")
        print(f"{json.dumps(dataset[7], ensure_ascii=False, indent=2)} ")
        print(f"{json.dumps(dataset[8], ensure_ascii=False, indent=2)} ")
        print(f"{json.dumps(dataset[9], ensure_ascii=False, indent=2)} ")


prepare_multi_task_dataset()
