from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 模型本地路径
model_path = r"E:\project\llm\model-data\base-models\TinyLlama-1.1B-Chat-v1.0"

# 配置量化参数（显存不足时启用，可选）
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # 4bit量化（也可设为load_in_8bit=True）
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )


def interactive_peft_lora_test():
    print("=" * 60)
    print("🤖 交互式PEFT LoRA分类器测试")
    print("=" * 60)

    print("输入问题进行分类测试，输入 'quit' 退出")
    print("-" * 60)

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=bnb_config,  # 若不量化可删除此行
        device_map="auto",  # 自动分配设备（优先GPU）
        trust_remote_code=True
    )

    while True:
        try:
            question = input("请输入问题: ").strip()

            if question.lower() in ['quit', 'exit', '退出', 'q']:
                break

            if not question:
                continue

            # 构造TinyLlama要求的对话格式（参考官方说明）
            formatted_prompt = f"<|user|>\n{question}\n<|assistant|>\n"

            # 编码输入
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            # 生成回复（可调整参数控制生成效果）
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # 最大生成长度
                temperature=0.7,  # 随机性（0-1，值越小越确定）
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # 解码并打印结果
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("模型回复：\n", response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"处理错误: {e}")


if __name__ == "__main__":
    interactive_peft_lora_test()
