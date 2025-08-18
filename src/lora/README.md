# LoRA问题分类器

基于LoRA（Low-Rank Adaptation）技术的BERT问题分类器实现，用于对问题进行三分类：数据平台相关、通用对话、无关问题。

## 🚀 特性

- **高效微调**: 使用LoRA技术，只训练少量参数（通常<5%），大幅减少计算资源需求
- **保留原有能力**: 冻结预训练模型参数，保持BERT的原有语言理解能力
- **灵活配置**: 支持自定义LoRA参数（rank、alpha、dropout）
- **完整流程**: 包含训练、评估、推理和交互测试功能
- **易于使用**: 提供命令行和交互式两种使用方式

## 📋 环境要求

```bash
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
tqdm>=4.64.0
numpy>=1.21.0
```

## 🛠️ 安装依赖

```bash
pip install torch transformers scikit-learn tqdm numpy
```

## 📁 文件结构

```
src/lora/
├── lora_question_classifier.py    # 主实现文件
├── test_lora_implementation.py    # 测试脚本
└── README.md                      # 说明文档
```

## 🎯 核心概念

### LoRA技术原理

LoRA（Low-Rank Adaptation）通过在预训练模型的线性层中添加低秩矩阵来实现高效微调：

```
h = W₀x + ΔWx = W₀x + BAx
```

其中：
- `W₀`: 冻结的预训练权重
- `B`, `A`: 可训练的低秩矩阵
- `rank`: 低秩矩阵的秩，控制参数量
- `alpha`: 缩放因子，控制LoRA的影响强度

### 参数配置

- **lora_rank**: LoRA矩阵的秩（默认8）
  - 较小值：参数更少，训练更快，但表达能力有限
  - 较大值：表达能力更强，但参数增加

- **lora_alpha**: 缩放因子（默认16）
  - 控制LoRA适应的强度
  - 通常设为rank的2倍

- **lora_dropout**: Dropout率（默认0.1）
  - 防止过拟合
  - 提高泛化能力

## 🚀 快速开始

### 1. 训练模型

```bash
# 使用默认参数训练
python lora_question_classifier.py train

# 或者直接运行进入菜单
python lora_question_classifier.py
```

### 2. 测试模型

```bash
# 测试已训练的模型
python lora_question_classifier.py test
```

### 3. 交互式测试

```bash
# 交互式问题分类测试
python lora_question_classifier.py interactive
```

### 4. 运行测试脚本

```bash
# 验证实现正确性
python test_lora_implementation.py
```

## 📊 使用示例

### 训练自定义LoRA模型

```python
from lora_question_classifier import LoRAQuestionClassifierTrainer

# 创建训练器
trainer = LoRAQuestionClassifierTrainer(
    model_path="path/to/bert-base-chinese",
    save_dir="./my_lora_model",
    lora_rank=16,      # 更大的rank
    lora_alpha=32,     # 对应的alpha
    lora_dropout=0.1
)

# 训练模型
trainer.train(
    epochs=5,
    batch_size=8,
    learning_rate=1e-4
)

# 保存模型
trainer.save_lora_weights('final_model')
```

### 使用训练好的模型进行推理

```python
from lora_question_classifier import LoRAQuestionClassifier

# 加载模型
classifier = LoRAQuestionClassifier(
    base_model_path="path/to/bert-base-chinese",
    lora_path="./my_lora_model/final_model"
)

# 分类问题
result = classifier.classify_question("数据清洗流程是什么？")
print(f"分类: {result['category']}")
print(f"置信度: {result['confidence']:.3f}")
```

## 🎛️ 配置说明

### 模型路径配置

在代码中修改BERT模型路径：

```python
bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
```

### LoRA保存路径

默认保存到：`E:/project/llm/lora/lora_question_classifier`

可以通过`save_dir`参数自定义：

```python
trainer = LoRAQuestionClassifierTrainer(
    model_path=bert_model_path,
    save_dir="./custom_save_path"
)
```

## 📈 性能对比

| 方法 | 可训练参数 | 训练时间 | 内存占用 | 准确率 |
|------|------------|----------|----------|--------|
| 全量微调 | ~110M | 长 | 高 | 高 |
| LoRA微调 | ~2M | 短 | 低 | 接近全量 |

## 🔧 高级用法

### 自定义数据集

```python
def prepare_custom_data():
    texts = ["你的问题1", "你的问题2", ...]
    labels = [0, 1, 2, ...]  # 0:数据平台, 1:通用对话, 2:无关
    return texts, labels

# 在训练器中使用
trainer.prepare_training_data = prepare_custom_data
```

### 调整LoRA应用层

默认应用到attention层和前馈网络，可以在`_apply_lora`方法中自定义。

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 减小lora_rank
   - 使用CPU训练

2. **模型路径错误**
   - 检查BERT模型路径是否正确
   - 确保模型文件完整

3. **依赖版本冲突**
   - 使用虚拟环境
   - 按照要求安装指定版本

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 扩展开发

### 添加新的分类类别

1. 修改`label_map`
2. 更新训练数据
3. 调整模型输出维度

### 支持其他预训练模型

1. 修改模型加载代码
2. 调整LoRA应用层
3. 适配tokenizer
