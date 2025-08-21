# Qwen3-0.6B Function Calling 微调

基于LoRA的PEFT库对Qwen3-0.6B模型进行Function Calling能力微调。

## 项目结构

```
src/Qwen3_06B/function_calling/
├── function_call_train.py          # 主要的微调训练代码
├── run_function_calling_train.py   # 训练启动脚本
└── README.md                       # 本文档
```

## 功能特性

- **基于LoRA微调**: 使用PEFT库进行高效的参数微调
- **Function Calling支持**: 训练模型学习工具调用能力
- **多种工具类型**: 支持天气查询、计算、搜索、文件操作等多种工具
- **完整训练流程**: 包含数据预处理、模型训练、保存和测试

## 数据集

使用 `src/dataset/function_calling.py` 中定义的Function Calling数据集，包含以下类型的训练样本：

- 天气查询 (2个样本)
- 时间查询 (2个样本)  
- 数学计算 (2个样本)
- 网络搜索 (1个样本)
- 文件操作 (1个样本)
- 邮件发送 (1个样本)
- 文本翻译 (1个样本)
- 数据库操作 (1个样本)
- 图像处理 (1个样本)
- 系统命令 (1个样本)
- 数据分析 (1个样本)
- 多函数调用 (1个样本)

总计：**15个训练样本**

## 模型配置

### 基础模型
- **模型**: Qwen3-0.6B
- **路径**: `E:\project\llm\model-data\base-models\Qwen3-0.6B`

### 输出模型
- **路径**: `E:\project\llm\model-data\train-models\Qwen3-function-calling`

### LoRA配置
- **Rank (r)**: 16 (比普通微调更高，以更好学习function calling)
- **Alpha**: 32
- **Dropout**: 0.1
- **目标模块**: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

### 训练参数
- **训练轮数**: 5 epochs
- **批次大小**: 1 (per device)
- **梯度累积步数**: 8
- **学习率**: 1e-4
- **最大序列长度**: 1024 (Function Calling需要更长的上下文)
- **优化器**: AdamW
- **学习率调度**: Cosine

## 使用方法

### 1. 环境准备

确保安装了以下依赖：
```bash
pip install torch transformers datasets peft accelerate
```

### 2. 模型准备

确保Qwen3-0.6B基础模型已下载到指定路径：
```
E:\project\llm\model-data\base-models\Qwen3-0.6B\
```

### 3. 开始训练

#### 方法一：使用启动脚本（推荐）
```bash
cd src/Qwen3_06B/function_calling
python run_function_calling_train.py
```

#### 方法二：直接运行主脚本
```bash
cd src/Qwen3_06B/function_calling
python function_call_train.py
```

### 4. 训练过程

训练过程包含以下步骤：
1. 加载基础模型和分词器
2. 设置LoRA配置
3. 准备Function Calling数据集
4. 执行微调训练
5. 保存微调后的模型
6. 进行简单的功能测试

### 5. 输出文件

训练完成后，微调后的模型将保存到：
```
E:\project\llm\model-data\train-models\Qwen3-function-calling\
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

## Function Calling格式

训练后的模型将学会以下格式进行工具调用：

### 输入格式
```
用户: 今天北京的天气怎么样？
```

### 期望输出格式
```
我来帮您查询北京今天的天气情况。
<tool_call>
get_weather({"location": "北京", "date": "today"})
</tool_call>
```

## 测试用例

训练完成后会自动运行以下测试用例：
1. "今天北京的天气怎么样？"
2. "帮我计算一下 25 * 4 + 10"
3. "现在几点了？"
4. "搜索一下人工智能的最新发展"

## 注意事项

1. **硬件要求**: 建议使用GPU进行训练，至少需要8GB显存
2. **训练时间**: 根据硬件配置，完整训练可能需要30分钟到2小时
3. **内存使用**: Function Calling需要较长的序列长度，注意内存使用情况
4. **数据质量**: 可以根据需要扩展 `function_calling_dataset` 以提高模型性能

## 扩展功能

### 添加新的工具类型
1. 在 `src/dataset/function_calling.py` 中添加新的训练样本
2. 重新运行训练脚本

### 调整训练参数
修改 `function_call_train.py` 中的以下参数：
- `num_train_epochs`: 训练轮数
- `learning_rate`: 学习率
- `per_device_train_batch_size`: 批次大小
- `max_length`: 最大序列长度

## 故障排除

### 常见问题

1. **模型路径不存在**
   - 确保Qwen3-0.6B模型已正确下载
   - 检查路径配置是否正确

2. **显存不足**
   - 减小 `per_device_train_batch_size`
   - 减小 `max_length`
   - 使用更小的LoRA rank

3. **训练过程中断**
   - 检查 `save_steps` 配置
   - 从最近的checkpoint恢复训练

## 相关文件

- `src/dataset/function_calling.py`: Function Calling训练数据集
- `src/Qwen3_06B/lora_fine_tuning.py`: 参考的LoRA微调代码
- `src/dataset/validate_function_calling.py`: 数据集验证脚本
