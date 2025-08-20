# PEFT LoRA 数据平台训练系统

基于PEFT库的BERT-base-chinese指令微调训练器，使用LoRA技术对BERT模型进行高效微调，支持多任务训练。

## 功能特性

- ✅ 基于PEFT库的LoRA实现，参数高效微调
- ✅ 支持指令微调格式的数据训练
- ✅ 数据集自动打乱，提升训练效果
- ✅ 多任务支持：问题场景分类、问题类型分类等
- ✅ 完整的训练、评估和推理流程
- ✅ 交互式测试功能

## 文件结构

```
src/lora/bert_base_chinese/
├── peft_lora_data_platform.py    # 主训练脚本
├── dataset.py                    # 数据集定义
└── README.md                     # 使用说明
```

## 环境要求

```bash
pip install torch transformers peft scikit-learn tqdm
```

## 模型配置

### 基础模型路径
```python
bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
```

### LoRA模型保存路径
```python
lora_path = r"E:\project\llm\lora\peft_lora_data_platform"
```

## 数据集说明

数据集采用指令微调格式，包含以下字段：
- `instruction`: 任务指令
- `input`: 输入文本
- `output`: 期望输出

### 支持的任务类型

1. **问题场景分类** (`question_classifier`)
   - 数据平台相关问题
   - 通用问题
   - 无关问题

2. **问题类型分类** (`question_type_classifier`)
   - 问题回答
   - 任务处理

3. **问题回答** (`question_answer`)
   - 基于知识库的问答

4. **文件名称提取** (`file_name_pick`)
   - 从描述中提取文件名

## 使用方法

### 1. 直接运行主程序

```bash
python peft_lora_data_platform.py
```

程序会显示菜单：
```
🤖 PEFT LoRA 数据平台训练系统
基于BERT-base-chinese的指令微调
============================================================
1. 训练问题场景分类器
2. 训练问题类型分类器
3. 测试推理功能
4. 交互式测试
0. 退出
============================================================
```

### 2. 编程方式调用

```python
from peft_lora_data_platform import PeftLoraTrainer, PeftLoraInference

# 训练模型
trainer = PeftLoraTrainer(
    bert_model_path=bert_model_path,
    lora_path=lora_path,
    task_type="question_classifier",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# 开始训练
trainer.train(
    epochs=5,
    batch_size=8,
    learning_rate=2e-4
)

# 推理使用
inference = PeftLoraInference(bert_model_path, lora_path)
result = inference.predict("问题场景分类", "数据清洗流程是什么？")
print(result)
```

## 训练参数说明

### LoRA配置
- `lora_r`: LoRA的秩，控制参数量 (默认: 16)
- `lora_alpha`: LoRA的缩放因子 (默认: 32)
- `lora_dropout`: LoRA的dropout率 (默认: 0.1)

### 训练配置
- `epochs`: 训练轮数 (默认: 5)
- `batch_size`: 批次大小 (默认: 8)
- `learning_rate`: 学习率 (默认: 2e-4)
- `warmup_steps`: 预热步数 (默认: 50)

## 数据集特点

### 数据打乱
- 所有数据集在训练前会自动打乱 (`shuffle_data=True`)
- 提升模型的泛化能力和训练效果

### 指令格式
```python
{
    "instruction": "问题场景分类",
    "input": "数据清洗流程是什么？",
    "output": "数据平台相关问题"
}
```

## 输出结果

### 训练输出
- 模型文件保存在指定的 `lora_path` 目录
- 包含LoRA权重、tokenizer、配置文件等
- 训练日志记录在 `logs/` 子目录

### 推理输出
```python
{
    'predicted_label': '数据平台相关问题',
    'confidence': 0.95,
    'probabilities': {
        '数据平台相关问题': 0.95,
        '通用问题': 0.03,
        '无关问题': 0.02
    }
}
```

## 性能特点

### 参数效率
- 使用LoRA技术，只训练少量参数
- 大幅减少显存占用和训练时间
- 保持原模型能力的同时获得任务特化能力

### 训练效果
- 支持多任务联合训练
- 数据打乱提升训练效果
- 完整的评估指标（准确率、F1分数等）

## 注意事项

1. **模型路径**: 确保BERT基础模型路径正确
2. **依赖库**: 安装所需的Python包
3. **显存**: 根据显存大小调整batch_size
4. **数据格式**: 确保数据集格式符合要求

## 扩展功能

### 添加新任务
1. 在 `dataset.py` 中添加新的数据集
2. 在 `PeftLoraTrainer` 中添加对应的任务类型
3. 更新标签映射和数量配置

### 自定义数据
1. 按照指令微调格式准备数据
2. 修改 `prepare_data` 方法加载自定义数据
3. 调整相应的标签映射

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减小batch_size
2. **模型加载失败**: 检查模型路径
3. **依赖库缺失**: 安装所需包

### 调试建议
1. 使用小数据集测试
2. 检查日志输出
3. 验证数据格式

---

## 更
