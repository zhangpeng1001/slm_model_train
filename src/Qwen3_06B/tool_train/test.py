from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

output_model_path = r"E:\project\llm\model-data\train-models\Qwen3-tool"
# 加载微调后的模型
infer_tokenizer = AutoTokenizer.from_pretrained(output_model_path)
infer_model = AutoModelForCausalLM.from_pretrained(output_model_path)

# 创建推理管道
generator = pipeline(
    "text-generation",
    model=infer_model,
    tokenizer=infer_tokenizer,
    max_new_tokens=256,
    device="cpu",  # 显式指定 CPU
)


# 输入指令
# instruction = "治理河流数据"
# response = generator(instruction)[0]["generated_text"]
# print(response)
# # 提取 JSON 输出
# start_idx = response.find("{")
# end_idx = response.rfind("}") + 1
# json_output = response[start_idx:end_idx]
# print(json_output)


# 测试工具调用生成
def generate_tool_call(prompt):
    messages = [
        {"role": "system", "content": "你是一个工具调用助手"},
        {"role": "user", "content": prompt}
    ]
    inputs = infer_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cpu")
    outputs = infer_model.generate(inputs, max_new_tokens=256, temperature=0.1)
    return infer_tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


# 示例测试
# print(generate_tool_call("查询上海明天湿度"))
# 期望输出：<tool_call>{"name": "get_weather", "arguments": {"location": "Shanghai", "date": "tomorrow"}}</tool_call>

if __name__ == '__main__':
    """交互式测试模式"""
    print(f"\n{'=' * 80}")
    print("进入交互式测试模式")
    print("输入 'quit' 或 'exit' 退出")
    print(f"{'=' * 80}")

    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("退出交互式测试模式")
                break

            if not user_input:
                continue

            print(f"\n处理中...")

            # 生成响应
            # result = generator(user_input)[0]["generated_text"]
            result = generate_tool_call(user_input)

            print(f"\n{'=' * 60}")
            print(f"用户: {user_input}")
            print(f"助手1: {result}")
            # print(f"助手: {result['content']}")
            #
            # if result["tool_calls"]:
            #     print(f"\n工具调用:")
            #     for i, tool_call in enumerate(result["tool_calls"]):
            #         func_name = tool_call["function"]["name"]
            #         func_args = tool_call["function"]["arguments"]
            #         print(f"  {i + 1}. {func_name}({func_args})")
            #
            # print(f"\n生成时间: {result['generation_time']:.2f}秒")
            # print(f"{'=' * 60}")

        except KeyboardInterrupt:
            print("\n\n用户中断，退出交互式测试模式")
            break
        except Exception as e:
            print(f"处理时出错: {e}")
            continue
