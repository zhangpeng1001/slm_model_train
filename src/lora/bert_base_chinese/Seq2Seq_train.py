from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType

bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
save_path = r"E:\project\llm\lora\bert_base_chinese_seq2seq"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(bert_model_path)

# 配置LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # 选择序列到序列任务类型
    inference_mode=False,
    r=8,  # LoRA秩
    lora_alpha=32,  # 缩放因子
    lora_dropout=0.1  # Dropout概率
)

model = get_peft_model(model, peft_config)


# 数据处理函数
def preprocess_function(examples):
    inputs = [f"{example['instruction']}: {example['input']}" for example in examples]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # 处理标签
    labels = tokenizer(text_target=examples["output"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 创建数据集
from datasets import Dataset

dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=save_path,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir=os.path.join(save_path, 'logs'),
    load_best_model_at_end=True,
)

# 创建Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)



# 推理代码
def generate_response(instruction, input_text):
    prompt = f"{instruction}: {input_text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # 生成输出
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    # 开始训练
    trainer.train()

    # 示例调用
    # print(generate_response("问题场景分类", "数据清洗流程是什么？"))  # 输出: 数据平台相关问题
    # print(generate_response("问题回答", "数据清洗流程"))  # 输出: 数据清洗流程需要先配置清洗规则...
    # print(generate_response("文件名称提取", "请帮我把实景三维模型成果数据进行治理"))  # 输出: 实景三维模型成果
