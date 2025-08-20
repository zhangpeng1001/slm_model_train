import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, GenerationConfig
from peft import PeftModel, PeftConfig

from src.lora.bert_base_chinese.peft_lora_data_platform import BertForInstructionTuning

device = torch.device('cpu')  # 强制使用CPU


# 加载模型和分词器
def load_model():
    # 替换为你的模型保存路径
    peft_model_path = r"E:\project\llm\lora\peft_lora_mixed"
    base_model_name = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 加载配置和基础模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = BertModel.from_pretrained(base_model_name)
    model = BertForInstructionTuning(base_model)

    # 添加生成配置
    model.generation_config = GenerationConfig(
        max_length=128,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id,
        eos_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.cls_token_id
    )

    print("✓ 基础模型加载成功")

    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, peft_model_path)
    model.eval()  # 切换到评估模式
    return model, tokenizer


# 简单的预测函数（使用分类方式而非生成）
def get_response(model, tokenizer, instruction, input_text):
    # 构建输入文本
    prompt = f"指令: {instruction}\n问题: {input_text}"
    # 编码文本
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
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(logits, dim=1)
        confidence = probabilities[0][predicted_label]

    return {
        'text': text,
        'predicted_type': predicted_label,
        'confidence': confidence,
        'probabilities': probabilities
    }


# 测试
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
        print(f"指令: {instruction}")
        print(f"输入: {text}")
        print(f"输出: {result}\n")
