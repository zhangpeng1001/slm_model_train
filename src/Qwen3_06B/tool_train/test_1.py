import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置路径
base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
finetuned_model_path = r"E:\project\llm\model-data\train-models\Qwen3-tool"

# 系统提示词（微调后仍需添加）
system_msg = {
    "role": "system",
    "content": "你只能返回<tool_call>JSON</tool_call>格式"
}


def load_model():
    """加载基础模型和微调适配器"""
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print("加载微调适配器...")
    model = PeftModel.from_pretrained(base_model, finetuned_model_path)
    model = model.merge_and_unload()  # 合并LoRA权重
    model.eval()
    return model


def test_function_calling(model, tokenizer):
    """测试function calling能力"""
    test_cases = [
        {
            "system": "你是一个专业助手，请根据用户需求调用合适的工具",
            "user": "查询北京明天的天气",
            "expected": "<|FunctionCallBegin|>思考过程<|FunctionCallEnd|>"
        },
        {
            "system": "你是一个数学助手，请调用计算器解决问题",
            "user": "计算385乘以27等于多少？",
            "expected": "<|FunctionCallBegin|>思考过程<|FunctionCallEnd|>"
        },
        {
            "system": "你是一个医学助手，请调用医学数据库查询",
            "user": "阿司匹林的禁忌症有哪些？",
            "expected": "<|FunctionCallBegin|>思考过程<|FunctionCallEnd|>"
        }
    ]

    print("\n" + "=" * 50)
    print("功能调用测试结果")
    print("=" * 50)

    for i, case in enumerate(test_cases):
        messages = [
            {"role": "system", "content": case["system"]},
            {"role": "user", "content": case["user"]}
        ]

        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # 生成响应
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,  # 降低随机性
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # 输出清洗
        # tool_call = response.split("<tool_call>")[1].split("</tool_call>")[0]
        # args = json.loads(tool_call)["arguments"]
        # print(f"调用{args['name']} 参数:{args}")

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 提取助手回复
        assistant_start = response.find("<|im_start|>assistant")
        if assistant_start != -1:
            assistant_response = response[assistant_start:]
            assistant_response = assistant_response.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "")
        else:
            assistant_response = response

        # 验证结果
        success = case["expected"] in assistant_response
        status = "✓" if success else "✗"

        print(f"\n测试案例 {i + 1}: {status}")
        print(f"用户输入: {case['user']}")
        print(f"生成响应: {assistant_response[:200]}...")
        print(f"预期标记: {case['expected']}")
        print("-" * 50)


def interactive_test(model, tokenizer):
    """交互式功能测试"""
    print("\n" + "=" * 50)
    print("交互测试模式 (输入'exit'退出)")
    print("=" * 50)

    messages = [
        {
            "role": "system",
            "content": "你是一个多功能助手，请根据用户需求调用合适的工具并生成结构化响应"
        }
    ]

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'exit':
            break

        messages.append({"role": "user", "content": user_input})

        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 提取助手回复并更新对话历史
        assistant_start = response.find("<|im_start|>assistant")
        if assistant_start != -1:
            assistant_response = response[assistant_start:]
            assistant_response = assistant_response.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "")
        else:
            assistant_response = response

        print(f"\n助手: {assistant_response}")
        messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True
    )
    model = load_model()

    # 执行测试
    test_function_calling(model, tokenizer)
    interactive_test(model, tokenizer)
