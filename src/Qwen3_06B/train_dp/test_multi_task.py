from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from peft import PeftModel

# 配置路径
base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
fine_tuned_model_path = r"E:\project\llm\model-data\train-models\Qwen3-multi-task"


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
    # 编码输入
    inputs = tokenizer(query, return_tensors="pt").to(model.device)

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
    # response = response[len(prompt):].strip()

    return response

# user 用户、assistant 助手、system 系统指令
def test_query_pipeline(model, tokenizer, query: str) -> str:
    """基于模型 tokenizer_config.json文件的内容，根据 chat_template 重构查询：Hugging Face Pipeline（快速调用）"""
    # Pipeline自动用chat_template格式化messages
    chatbot = pipeline("conversational", model=model, tokenizer=tokenizer)  # type: ignore

    chatbot = pipeline(
        task="conversational",
        tools=[
            {"name": "search_tool", "desc": "获取最新行业数据、统计报告"},
            {"name": "calculation_tool", "desc": "执行数值计算（增长率、占比等）"}
        ],
        system_prompt="你是智能助手，对话中若需要数据支持，需先调用search_tool获取准确数据，再用calculation_tool计算，最后用自然语言回复用户"
    )

    # 2. 手动指定 pipeline_class，无需传 task
    # chatbot = pipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     pipeline_class=ConversationalPipeline  # 直接指定对话专用 Pipeline 类
    # )
    messages = [{"role": "user", "content": query}]
    response = chatbot(messages)

    return response


def test_query_generate(model, tokenizer, query: str) -> str:
    """基于模型 tokenizer_config.json文件的内容，根据 chat_template 重构查询：手动构建模型输入（精细化控制）"""
    # 1. 定义对话历史
    messages = [
        {"role": "system", "content": "你是一个AI助手，简洁回答问题"},
        {"role": "user", "content": query}
    ]

    # 2. 用chat_template格式化（add_generation_prompt=True表示要生成回复）
    formatted_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"  # 返回PyTorch张量（模型输入格式）
    )

    # 3. 模型生成回复
    response = model.generate(formatted_text, max_new_tokens=100)
    # 4. 解码输出（自动忽略特殊Token）
    response = tokenizer.decode(response[0], skip_special_tokens=False)  # 不跳过特殊Token可看完整格式
    return response


def main():
    # 加载微调后的模型
    print("加载微调后的模型...")
    fine_tuned_model, tokenizer = load_model_and_tokenizer()

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

            # response = test_function_calling(fine_tuned_model, tokenizer, user_input)
            #
            # response = test_query_pipeline(fine_tuned_model, tokenizer, user_input)

            response = test_query_generate(fine_tuned_model, tokenizer, user_input)

            print(f"\n{'=' * 60}")
            print(f"用户: {user_input}")
            print(f"助手1: {response}")

        except KeyboardInterrupt:
            print("\n\n用户中断，退出交互式测试模式")
            break
        except Exception as e:
            print(f"处理时出错: {e}")
            continue


if __name__ == "__main__":
    main()
