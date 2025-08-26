# Qwen3-0.6B 多任务训练系统

## 概述

本项目实现了基于Qwen3-0.6B模型的多任务微调训练系统，支持5个不同的数据平台相关任务：

1. **问题分类** (Question Classification) - 判断问题类型（数据平台相关、通用对话、无关问题）
2. **问题类型分类** (Question Type Classification) - 判断问题是问题回答还是任务处理
3. **数据平台问答** (Data Platform QA) - 回答数据平台相关问题
4. **数据提取** (Data Extraction) - 从文本中提取数据名称
5. **工具调用** (Tool Calling) - 调用数据平台相关工具函数

## 文件结构

```
src/Qwen3_06B/train_dp/
├── multi_task_train.py      # 多任务训练主程序
├── test_multi_task.py       # 测试脚本
├── train_all.py            # 原始单任务训练代码（已废弃）
└── README.md               # 本文档
```

## 环境要求

- Python 3.8+
- PyTorch (CPU版本)
- Transformers
- PEFT
- Datasets

## 特性

### CPU优化配置
- 使用较小的LoRA参数（r=8）减少计算量
- 小批次大小（batch_size=1）适应CPU内存限制
- 梯度累积增加有效批次大小
- 禁用fp16（CPU不支持）
- 不使用多进程数据加载

### 多任务支持
- 统一的数据预处理流程
- 任务特定的系统提示词
- 不同任务类型的输入格式化
- 数据集自动合并和打乱

## 数据集配置

每个任务的数据集配置包含：
- `file_path`: 数据集文件路径
- `system_prompt`: 任务特定的系统提示词
- `task_type`: 任务类型标识

### 支持的数据集

1. **data_extraction.json** - 数据提取任务
   - 输入：包含数据名称的文本
   - 输出：提取的数据名称

2. **dp_qa.json** - 数据平台问答任务
   - 输入：数据平台相关问题
   - 输出：详细答案

3. **question_classifier.json** - 问题分类任务
   - 输入：用户问题
   - 输出：分类结果（数据平台相关/通用对话/无关问题）

4. **question_type_classifier.json** - 问题类型分类任务
   - 输入：用户问题
   - 输出：问题类型（问题回答/任务处理）

5. **tool_data_platform.json** - 工具调用任务
   - 输入：用户请求
   - 输出：函数调用格式

## 使用方法

### 1. 运行测试
```bash
cd src/Qwen3_06B/train_dp
python test_multi_task.py
```

### 2. 开始训练
```bash
cd src/Qwen3_06B/train_dp
python multi_task_train.py
```

### 3. 自定义配置

修改 `multi_task_train.py` 中的配置：

```python
# 模型路径配置
model_name = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
output_model_path = r"E:\project\llm\model-data\train-models\Qwen3-multi-task"

# 训练参数
per_device_train_batch_size=1  # 批次大小
gradient_accumulation_steps=4  # 梯度累积步数
num_train_epochs=3            # 训练轮数
learning_rate=1e-4           # 学习率
```

## 训练参数说明

### LoRA配置
- `r=8`: 低秩适应的秩，控制参数量
- `lora_alpha=16`: LoRA的缩放参数
- `lora_dropout=0.1`: Dropout率
- `target_modules=["q_proj", "v_proj"]`: 目标模块

### 训练配置
- `per_device_train_batch_size=1`: 每设备批次大小
- `gradient_accumulation_steps=4`: 梯度累积步数
- `num_train_epochs=3`: 训练轮数
- `learning_rate=1e-4`: 学习率
- `max_length=512`: 最大序列长度

## 输出格式

训练完成后，模型将保存到指定路径，包含：
- 模型权重文件
- 分词器配置
- LoRA适配器权重

## 系统提示词

每个任务都有专门的系统提示词：

### 数据提取
```
你是一个专业的数据提取助手。任务是分析用户输入的文本，提取用户描述的数据名称
```

### 数据平台问答
```
你是一个专业的数据平台问答助手。任务是分析用户输入的问题，并提供答案给用户
```

### 问题分类
```
你是一个专业的问题分类助手。任务是分析用户输入的文本，判断用户的问题类型是（数据平台相关、通用对话、无关问题）中哪一个
```

### 问题类型分类
```
你是一个专业的问题类型分类助手。任务是分析用户输入的文本，判断用户的问题类型是（问题回答、任务处理）中哪一个
```

### 工具调用
```
你是数据中台项目的工具调用助手，可以调用以下函数：
- get_data_collection(...)：用于数据采集工具
- query_data_by_filename(...)：用于文件名查数据工具
- data_warehousing(...)：用于数据入库工具
- data_service_publish(...)：用于数据发服务工具
- data_quality_check(...)：用于数据质检工具
- data_cleaning(...)：用于数据清洗工具
```

## 注意事项

1. **CPU环境优化**: 代码已针对CPU环境进行优化，训练时间可能较长
2. **内存使用**: 建议至少8GB内存，大数据集可能需要更多内存
3. **数据路径**: 确保数据集文件路径正确，文件格式为JSON
4. **模型路径**: 确保Qwen3-0.6B基础模型路径正确

## 故障排除

### 常见问题

1. **内存不足**: 减少batch_size或max_length
2. **数据加载失败**: 检查数据集文件路径和格式
3. **模型加
