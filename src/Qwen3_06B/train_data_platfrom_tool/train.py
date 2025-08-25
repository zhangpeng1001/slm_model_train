from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, \
    BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer
import json
import torch

# 配置路径
model_name = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
output_model_path = r"E:\project\llm\model-data\train-models\Qwen3-tool-dp"


# 在你的train.py文件中，确保主程序逻辑被包裹在这个条件判断中
def main():
    # === Step 1: 加载数据集 ===
    def load_json_data(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Dataset.from_list(data)

    dataset = load_json_data(r"E:\project\python\slm_model_train\src\dataset\tool_data_platform.json")

    # 配置4/8位量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 或 load_in_8bit=True
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # === Step 2: 加载模型和分词器 ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 关键优化：启用8-bit量化加载模型，减少CPU内存占用（需安装bitsandbytes库）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # quantization_config=bnb_config,
        # device_map="cpu",  # 明确指定在CPU上运行
        # torch_dtype=torch.bfloat16  # CPU通常不支持fp16，使用float32
        # torch_dtype=torch.float32  # CPU通常不支持fp16，使用float32
    )
    # === Step 3: 配置 LoRA ===
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # 适度降低秩（从16→12），减少计算量同时保持效果
        lora_alpha=8,  # 增大alpha值补偿r的降低（alpha/r比例保持合理）
        lora_dropout=0.05,  # 保持dropout防止过拟合
        # lora_dropout=0,  # CPU环境缺乏GPU的并行计算优势，Dropout会显著拖慢速度
        # 针对因果LM，增加关键注意力模块（不显著增加计算量但提升效果）
        target_modules=["q_proj", "v_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # bias="none",  # 不训练偏置参数，减少计算
        # inference_mode=False
    )
    # 选择性参数冻结:仅训练适配器避免全模型梯度计算
    # for param in model.parameters():
    #     param.requires_grad = False
    # 应用 LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # === Step 4: 数据预处理 ===
    def preprocess_function(examples):
        """优化的预处理函数，更适合function calling场景"""
        # 1. 处理输入：拼接instruction和input，添加function calling格式提示
        print("==" * 60)
        inputs = []
        for instr, ex_input in zip(
                examples["instruction"],
                examples.get("input", [""] * len(examples["instruction"]))
        ):
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
            prompt += f"Input: {ex_input}\n"
            prompt += f"Output: "  # 明确输出起始标记
            inputs.append(prompt)
        print(f"inputs:{inputs}")
        # 2. 处理目标输出（function call内容）
        targets = examples["output"]

        # 3. 联合编码输入和目标（因果LM的标准做法）
        # 修正6：将输入和目标拼接，仅对目标部分计算损失
        full_texts = [f"{inpt}{tgt}{tokenizer.eos_token}" for inpt, tgt in zip(inputs, targets)]
        print(f"full_texts:{full_texts}")
        model_inputs = tokenizer(
            full_texts,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False,  # 移除不必要的返回值
            return_length=False
        )

        # 4. 构建标签：输入部分标记为-100（不参与损失计算）
        # 先单独编码输入部分以获取掩码边界
        input_only = tokenizer(
            inputs,
            max_length=256,
            truncation=True,
            padding="max_length"
        )

        # 修正7：用向量化操作替代循环，提升效率
        labels = []
        for input_ids, full_input_ids in zip(input_only["input_ids"], model_inputs["input_ids"]):
            # 找到输入部分的结束位置（排除padding）
            input_len = len([id for id in input_ids if id != tokenizer.pad_token_id])
            # 输入部分标签设为-100，目标部分保留原id
            label = [-100] * input_len + full_input_ids[input_len:]
            # 确保长度正确
            label = label[:256] + [-100] * max(0, 256 - len(label))
            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs

    # tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # 修正8：使用多进程加速预处理，移除不需要的列
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        # num_proc=4,  # 根据CPU核心数调整
        # remove_columns=dataset.column_names,  # 移除原始列节省内存
    )

    # === Step 5: 训练参数 ===
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     per_device_train_batch_size=8,  # CPU环境下保持最小batch
    #     gradient_accumulation_steps=2,  # 梯度累积（模拟batch_size=4），提升效果
    #     num_train_epochs=3,  # 保持较少轮次，减少总计算量
    #     learning_rate=2e-4,  # 适度提高LoRA学习率（LoRA参数少，可承受更高学习率）
    #     logging_dir=None,  # 禁用日志节省IO
    #     report_to="none",
    #     save_strategy="no",  # 不保存中间模型
    #     max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸
    #     # optim="adamw_torch_fused",  # 使用融合优化器，提升CPU训练速度
    #     optim="adamw_torch",  # 融合优化器在 GPU 上效率高，但在 CPU 上可能因缺少硬件支持反而变慢
    #     lr_scheduler_type="cosine",  # 余弦调度器，后期学习率衰减更平滑
    #     warmup_ratio=0.05,  # 小比例预热，稳定训练初期
    #     fp16=False,  # 禁用半精度（CPU 不支持）
    # )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,  # CPU 友好
        num_train_epochs=10,  # 减少训练轮数
        logging_dir=None,  # 禁用日志
        report_to="none",  # 禁用日志
        save_strategy="no",  # 不保存中间结果
        learning_rate=2e-4,  # 更低学习率
    )
    # === Step 6: 创建 Trainer ===

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
        # 按8的倍数padding，提升计算效率
    )

    # === Step 7: 开始训练 ===
    trainer.train()

    # === Step 8: 保存模型 ===
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)


if __name__ == '__main__':
    # 对于Windows系统，添加这个函数调用
    from multiprocessing import freeze_support

    freeze_support()

    # 调用主函数
    main()
