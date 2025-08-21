# Qwen3-0.6B LoRA微调项目

基于LoRA的PEFT库对Qwen3-0.6B模型进行指令微调的完整解决方案。

## 项目概述

本项目实现了对Qwen3-0.6B模型的LoRA（Low-Rank Adaptation）微调，专门针对以下四个任务：

1. **问题场景分类** - 将用户问题分类为"数据平台相关问题"、"通用问题"、"无关问题"
2. **问题类型分类** - 区分问题是"问题回答"类型还是"任务处理"类型
3. **问题回答** - 针对数据平台相关问题提供专业回答
4. **文件名称提取** - 从用户描述中提取关键的文件名称信息

## 文件结构

```
src/Qwen3_06B/
├── README.md                 # 项目说明文档
├── lora_fine_tuning.py      # LoRA微调训练核心代码
├── run_training.py          # 训练启动脚本
├── inference_test.py        # 推理测试工具
├── base_test.py            # 基础模型测试
└── base_test2.py           # 基础模型测试2
```

## 环境依赖

```bash
pip install torch transformers datasets peft accelerate bitsandbytes
pip install modelscope  # 用于加载Qwen模型
```

## 使用方法

### 1. 训练模型

#### 方法一：使用启动脚本（推荐）

```bash
cd src/Qwen3_06B
python run_training.py
```

#### 方法二：自定义参数训练

```bash
python run_training.py \
    --base_model "E:\project\llm\model-data\base-models\Qwen3-0.6B" \
    --output_dir "E:\project\llm\model-data\train-models\Qwen3-0.6B" \
    --max_length 512
```

#### 方法三：直接运行训练代码

```bash
python lora_fine_tuning.py
```

### 2. 测试微调结果

```bash
python inference_test.py
```

测试工具提供四种测试模式：
1. **综合测试** - 对所有任务类型进行全面测试
2. **交互式测试** - 手动输入测试用例
3. **性能评估** - 基于预期答案评估准确率
4. **全部测试** - 依次执行上述所有测试

## 训练配置

### LoRA配置参数
- **r**: 8 (LoRA rank)
- **lora_alpha**: 32 (LoRA scaling parameter)
- **lora_dropout**: 0.1 (LoRA dropout)
- **target_modules**: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

### 训练参数
- **num_train_epochs**: 3
- **per_device_train_batch_size**: 2
- **gradient_accumulation_steps**: 4
- **learning_rate**: 2e-4
- **warmup_steps**: 100
- **max_length**: 512

## 数据集说明

训练数据来源于 `src/dataset/dataset.py`，包含四个任务的训练样本：

- **question_classifier**: 问题场景分类数据（20个样本）
- **question_type_classifier**: 问题类型分类数据（23个样本）
- **question_answer**: 问题回答数据（28个样本）
- **file_name_pick**: 文件名称提取数据（50个样本）

总计：**121个训练样本**

## 模型路径配置

### 默认路径配置
- **基础模型路径**: `E:\project\llm\model-data\base-models\Qwen3-0.6B`
- **微调模型保存路径**: `E:\project\llm\model-data\train-models\Qwen3-0.6B`

### 自定义路径
如需修改路径，可以：
1. 修改 `run_training.py` 中的默认参数
2. 使用命令行参数指定
3. 直接修改 `lora_fine_tuning.py` 中的路径配置

## 测试示例

### 问题场景分类测试
```
输入: 任务: 问题场景分类\n输入: 数据清洗流程是什么？
预期输出: 数据平台相关问题
```

### 问题类型分类测试
```
输入: 任务: 问题类型分类\n输入: 我有一批西安市地类图斑的数据，怎么进行发服务
预期输出: 问题回答
```

### 问题回答测试
```
输入: 任务: 问题回答\n输入: 数据清洗流程
预期输出: 数据清洗流程需要先配置清洗规则，再配置清洗方案，最后针对要清洗的数据启动任务即可...
```

### 文件名称提取测试
```
输入: 任务: 文件名称提取\n输入: 我有一批西安市地类图斑的数据，怎么进行发服务
预期输出: 西安市地类图斑
```

## 性能优化建议

1. **GPU内存优化**
   - 使用 `fp16=True` 启用半精度训练
   - 调整 `per_device_train_batch_size` 和 `gradient_accumulation_steps`
   - 使用 `device_map="auto"` 自动分配GPU

2. **训练效率优化**
   - 根据数据集大小调整 `num_train_epochs`
   - 使用 `cosine` 学习率调度策略
   - 设置合适的 `warmup_steps`

3. **模型质量优化**
   - 增加训练数据的多样性
   - 调整 `max_length` 适应数据特点
   - 实验不同的 `lora_alpha` 和 `r` 值

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   解决方案：
   - 减小 batch_size
   - 启用 gradient_checkpointing
   - 使用更小的 max_length
   ```

2. **模型路径不存在**
   ```
   解决方案：
   - 检查基础模型是否正确下载到指定路径
   - 确认路径格式正确（使用原始字符串 r""）
   ```

3. **依赖包版本冲突**
   ```
   解决方案：
   - 使用虚拟环境
   - 更新到最新版本的transformers和peft
   ```

## 扩展功能

### 添加新的训练任务
1. 在 `src/dataset/dataset.py` 中添加新的数据集
2. 修改 `lora_fine_tuning.py` 中的数据加载部分
3. 更新测试用例

### 模型部署
1. 使用 `PeftModel.from_pretrained()` 加载微调模型
2. 结合基础模型进行推理
3. 可以转换为标准格式用于生产环境

## 技术特点

- ✅ **内存高效**: 使用LoRA技术，只训练少量参数
- ✅ **快速训练**: 相比全参数微调，训练速度显著提升
- ✅ **易于部署**: 微调权重文件小，便于分发和部署
- ✅ **任务专用**: 针对特定任务优化，效果显著
- ✅ **可扩展性**: 支持添加新任务和数据集

## 许可证

本项目采用 MIT 许可证。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 更新日志

### v1.0.0
- 实现基础的LoRA微调功能
- 支持四种任务类型的训练
- 提供完整的测试工具
- 添加详细的使用文档

---

**注意**: 请确保在运行前已正确安装所有依赖包，并且基础模型已下载到指定路径。
