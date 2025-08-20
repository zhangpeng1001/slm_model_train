import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, GenerationConfig
from peft import PeftModel, PeftConfig

from src.lora.bert_base_chinese.peft_lora_data_platform import BertForInstructionTuning

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def load_model():
    # 模型路径
    peft_model_path = r"E:\project\llm\lora\peft_lora_mixed"
    base_model_name = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型和指令微调模型
    base_model = BertModel.from_pretrained(base_model_name)
    model = BertForInstructionTuning(base_model)

    # 配置生成参数
    model.generation_config = GenerationConfig(
        max_length=128,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.cls_token_id,
        repetition_penalty=1.2
    )

    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, peft_model_path)
    model = model.to(device)
    model.eval()  # 切换到评估模式

    print("✓ 模型加载成功")
    return model, tokenizer


def get_response(model, tokenizer, instruction, input_text):
    """根据指令类型选择合适的处理方式（分类或生成）"""
    # 构建提示词
    prompt = f"指令: {instruction}\n问题: {input_text}\n回答:"

    # 编码输入
    encoding = tokenizer(
        prompt,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        # 判断任务类型：分类任务还是生成任务
        if "分类" in instruction or "提取" in instruction:
            # 分类任务：使用[CLS] token的输出
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 假设模型输出包含分类头，取[CLS]位置的logits (batch_size, num_labels)
            # 如果是序列输出，取第一个token的输出作为分类结果
            if outputs.logits.dim() == 3:  # (batch, seq_len, num_labels)
                cls_logits = outputs.logits[:, 0, :]  # 取[CLS] token
            else:
                cls_logits = outputs.logits  # 已经是(batch, num_labels)

            probabilities = torch.softmax(cls_logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_label].item()

            return {
                'instruction': instruction,
                'input': input_text,
                'output_type': 'classification',
                'predicted_label': predicted_label,
                'confidence': round(confidence, 4),
                'top_probability': round(torch.max(probabilities).item(), 4)
            }

        else:
            # 生成任务：使用generate方法
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=model.generation_config
            )

            # 解码生成的文本，去除输入部分
            generated_text = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # 提取生成的回答部分（去除提示词）
            if "回答:" in generated_text:
                answer = generated_text.split("回答:")[-1].strip()
            else:
                answer = generated_text.strip()

            return {
                'instruction': instruction,
                'input': input_text,
                'output_type': 'generation',
                'generated_text': answer
            }


def print_result(result):
    """格式化输出结果"""
    print(f"指令: {result['instruction']}")
    print(f"输入: {result['input']}")

    if result['output_type'] == 'classification':
        print(f"预测标签: {result['predicted_label']}")
        print(f"置信度: {result['confidence']}")
    else:
        print(f"生成回答: {result['generated_text']}")
    print("-" * 50)


if __name__ == "__main__":
    # 加载模型
    model, tokenizer = load_model()
    print("模型加载完成，开始测试...\n")

    # 测试案例
    test_cases = [
        ("问题场景分类", "数据平台怎么使用？"),
        ("问题类型分类", "帮我处理一下这些数据"),
        ("问题回答", "数据清洗有哪些步骤？"),
        ("文件名称提取", "请处理北京市建筑物矢量数据")
    ]

    # 执行测试
    for instruction, text in test_cases:
        result = get_response(model, tokenizer, instruction, text)
        print_result(result)
