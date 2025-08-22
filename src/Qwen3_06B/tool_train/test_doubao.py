from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import List, Dict, Any

from peft import PeftModel

# 模型路径
base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
fine_tuned_model_path = r"E:\project\llm\model-data\train-models\Qwen3-tool"

# 定义一些测试用的工具函数描述
TOOL_DESCRIPTIONS = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式为YYYY-MM-DD，可选参数，默认是当天"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "进行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如：2+3*4"
                }
            },
            "required": ["expression"]
        }
    }
]


def format_tools(tools: List[Dict[str, Any]]) -> str:
    """将工具描述格式化为模型可以理解的字符串"""
    formatted = "可用工具:\n"
    for tool in tools:
        formatted += f"- {tool['name']}: {tool['description']}\n"
        formatted += "  参数: " + str(tool['parameters']) + "\n"
    formatted += "如果需要调用工具，请使用以下格式包裹:\n"
    formatted += "<function_call>\n{\"name\": \"工具名\", \"parameters\": {\"参数名\": \"参数值\"}}\n</function_call>"
    return formatted


def load_model_and_tokenizer(model_path: str):
    """加载模型和tokenizer"""
    # 配置4位量化以节省内存
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. 加载微调后的适配器参数（20M的那个文件）
    model = PeftModel.from_pretrained(model, fine_tuned_model_path)

    # 3. （可选）合并基础模型与适配器参数，提升推理速度
    model = model.merge_and_unload()

    # 启用评估模式
    model.eval()
    return model, tokenizer


def test_function_calling(model, tokenizer, query: str, tools: List[Dict[str, Any]]) -> str:
    """测试模型的function calling能力"""
    # 构建提示
    tools_str = format_tools(tools)
    prompt = f"{tools_str}\n\n用户问: {query}\n请根据用户的问题，判断是否需要调用工具，如果需要，请按照指定格式调用合适的工具。如果不需要，可以直接回答。"

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
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
    fine_tuned_model, tokenizer = load_model_and_tokenizer(fine_tuned_model_path)

    # 也可以加载原始模型进行对比
    # print("加载原始模型...")
    # base_model, _ = load_model_and_tokenizer(model_name)

    # 测试用例
    test_cases = [
        "北京今天的天气怎么样？",
        "帮我计算一下35乘以24加上187等于多少",
        "什么是人工智能？",  # 这个问题不需要调用工具
        "上海明天会下雨吗？",
        "100的阶乘是多少？"
    ]

    # 运行测试
    print("\n开始测试...\n")
    for i, query in enumerate(test_cases, 1):
        print(f"测试用例 {i}: {query}")
        response = test_function_calling(fine_tuned_model, tokenizer, query, TOOL_DESCRIPTIONS)
        print(f"模型回答: {response}\n")


if __name__ == "__main__":
    main()
