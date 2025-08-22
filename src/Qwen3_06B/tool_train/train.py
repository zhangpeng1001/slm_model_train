from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer
import json

# 配置路径
model_name = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
output_model_path = r"E:\project\llm\model-data\train-models\Qwen3-tool"


# === Step 1: 加载数据集 ===
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


dataset = load_json_data(r"E:\project\python\slm_model_train\src\dataset\tool_train.json")  # 替换为你的数据集路径

# === Step 2: 加载模型和分词器 ===
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# === Step 3: 配置 LoRA ===
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # 低秩矩阵秩（CPU 友好）
    lora_alpha=8,  # 缩放因子
    lora_dropout=0.05,  # Dropout 概率
    target_modules=["q_proj", "v_proj"],  # 根据模型结构调整
)

# 应用 LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# === Step 4: 数据预处理 ===
def preprocess_function(examples):
    # 当使用batched=True时，examples是一个包含所有样本的字典
    # 每个键对应的值是一个包含该字段所有样本的列表
    inputs = [instr + ex_input for instr, ex_input in
              zip(examples["instruction"], examples.get("input", [""] * len(examples["instruction"])))]
    targets = examples["output"]

    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length").input_ids

    # 初始化labels列表
    model_inputs["labels"] = []

    # 处理标签，将padding token设为-100（在计算损失时会被忽略）
    for label in labels:
        processed_label = [-100 if token == tokenizer.pad_token_id else token for token in label]
        model_inputs["labels"].append(processed_label)

    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# === Step 5: 训练参数 ===
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # CPU 友好
    num_train_epochs=1,  # 减少训练轮数
    logging_dir=None,  # 禁用日志
    report_to="none",  # 禁用日志
    save_strategy="no",  # 不保存中间结果
    learning_rate=1e-4,  # 更低学习率
)

# === Step 6: 创建 Trainer ===

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# === Step 7: 开始训练 ===
trainer.train()

# === Step 8: 保存模型 ===
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
