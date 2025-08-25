from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import List, Dict, Any

from peft import PeftModel

# 模型路径
base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
fine_tuned_model_path = r"E:\project\llm\model-data\train-models\Qwen3-tool-dp"


def load_model_and_tokenizer():
    """加载模型和tokenizer"""

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        trust_remote_code=True
    )

    # 2. 加载微调后的适配器参数
    model = PeftModel.from_pretrained(model, fine_tuned_model_path)

    # 3. （可选）合并基础模型与适配器参数，提升推理速度
    model = model.merge_and_unload()

    # 启用评估模式
    model.eval()
    return model, tokenizer


def test_function_calling(model, tokenizer, query: str) -> str:
    """测试模型的function calling能力"""
    # 构建提示
    # tools_str = "你是数据中台项目的工具调用助手,请根据用户的问题，判断是否需要调用工具，如果需要，请返回合适的工具。如果不需要，可以直接回答。"
    # prompt = f"{tools_str}\n\n用户问: {query}\n"
    system_prompt = "你是数据中台项目的工具调用助手，可以调用以下函数："
    system_prompt += "\n- get_data_collection(data_source: str,data_type: str,time_range: str,business_platform: str)：用于数据采集工具;"
    system_prompt += "\n- query_data_by_filename(filename: str,query_content: st)：用于文件名查数据工具;"
    system_prompt += "\n- data_warehousing(source_data_path: str,target_db_type: str,target_db: str,target_table: str,order_detail:str)：用于数据入库工具;"
    system_prompt += "\n- data_service_publish(source_db_type: str,source_db: str,dw_sales: str,sales_summary: str,data_filter:str,service_type:str,authorization:str)：用于数据发服务工具;"
    system_prompt += "\n- data_quality_check(source_db_type: str,source_db: str,source_table: str,check_dimensions: str)：用于数据质检工具;"
    system_prompt += "\n- data_cleaning(source_data_path: str,source_data_type: str,clean_rules: str,target_save_path: str)：用于数据清洗工具;"
    system_prompt += "\n请根据指令和输入,选择合适的函数并按指定格式调用。\n"

    function_format = "如果需要调用函数，请使用以下格式：\n<FunctionCall>\n{\"name\":\"函数名\",\"parameters\":{\"参数名\":参数值}}\n</FunctionCall>\n"
    # 修正5：添加function calling格式引导（根据实际需求调整）
    prompt = system_prompt  # 加入系统提示
    prompt += function_format
    # prompt += f"Instruction: {instr}\n"
    prompt += f"Input: {query}\n"
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取模型回答部分（去除提示部分）
    response = response[len(prompt):].strip()

    return response


def main():
    # 加载微调后的模型
    print("加载微调后的模型...")
    fine_tuned_model, tokenizer = load_model_and_tokenizer()

    # 也可以加载原始模型进行对比
    # print("加载原始模型...")
    # base_model, _ = load_model_and_tokenizer(model_name)

    # 测试用例
    test_cases = [
        "需要采集近 7 天的线下门店销售数据，数据源类型是 MySQL 数据库，存储地址在 192.168.1.100:3306",
        "查询文件名包含 “用户画像_v2” 的所有 CSV 文件，需要知道它们的创建时间和数据大小",
        "将本地文件 “C:/data/user_202405.csv” 的用户数据增量入库到 MySQL 的 user_db 库的 user_info 表，按 “user_id” 字段去重",
        "将 MySQL user_db 库 user_portrait 表的用户画像数据（排除手机号字段），发布为文件下载服务，支持 CSV 格式，有效期 7 天",
        "质检本地文件 “user_data_202405.csv”，检查 “用户年龄” 字段是否在 18-65 之间、“手机号” 格式是否符合 11 位数字规则"
    ]

    # 运行测试
    print("\n开始测试...\n")
    for i, query in enumerate(test_cases, 1):
        print(f"测试用例 {i}: {query}")
        response = test_function_calling(fine_tuned_model, tokenizer, query)
        print(f"模型回答: {response}\n")


if __name__ == "__main__":
    main()
