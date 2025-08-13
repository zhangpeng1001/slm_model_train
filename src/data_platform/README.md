# 数据平台智能问答系统

## 项目简介

这是一个基于BERT模型和知识库的数据平台事实性问答系统，专门用于回答数据清洗、数据入库、数据处理等数据平台相关的问题。

## 功能特性

- 🤖 **智能问答**: 基于BERT中文模型进行语义理解
- 📚 **知识库**: 包含数据平台相关的专业知识
- 🔍 **语义匹配**: 支持模糊问题匹配和语义相似度计算
- 💬 **交互式界面**: 提供友好的命令行交互体验
- 🛡️ **容错机制**: 支持模型加载失败时的基础模式回退

## 支持的问答主题

### 数据清洗相关
- 数据清洗流程
- 数据清洗规则
- 数据质量检查

### 数据入库相关
- 数据入库流程
- 数据入库策略
- 数据库连接配置

### 数据处理相关
- 数据预处理
- 数据转换
- 数据标准化

### 数据监控相关
- 数据监控
- 任务调度
- 异常处理

### 数据安全相关
- 数据安全
- 权限管理
- 数据备份

## 安装依赖

```bash
pip install torch transformers numpy
```

## 使用方法

### 1. 启动交互式问答系统

```bash
# 方法1：直接运行启动脚本
cd src/data_platform
python run_qa.py

# 方法2：作为模块运行
python -m src.data_platform.qa_system
```

### 2. 运行测试

```bash
cd src/data_platform
python test_qa.py
```

### 3. 程序化调用

```python
from src.data_platform.qa_system import DataPlatformQASystem

# 创建问答系统
qa_system = DataPlatformQASystem()

# 提问
result = qa_system.ask_question("数据清洗流程是什么？")
print(result["answer"])

# 批量提问
questions = ["数据入库流程", "数据质量检查", "数据监控"]
results = qa_system.batch_ask(questions)
```

## 系统架构

```
src/data_platform/
├── knowledge_base.py      # 知识库模块
├── simple_qa_model.py     # 简化版BERT问答模型
├── qa_model.py           # 完整版BERT问答模型（需sklearn）
├── qa_system.py          # 问答系统主接口
├── run_qa.py             # 启动脚本
├── test_qa.py            # 测试脚本
└── README.md             # 说明文档
```

## 工作原理

1. **知识库匹配**: 首先尝试直接关键词匹配
2. **语义理解**: 使用BERT模型计算问题的向量表示
3. **相似度计算**: 计算用户问题与知识库问题的余弦相似度
4. **答案返回**: 返回最匹配的答案及置信度

## 示例对话

```
❓ 请输入您的问题：数据清洗流程是什么？

🔍 正在分析您的问题...

==============================
📋 回答：
数据清洗流程需要先配置清洗规则，再配置清洗方案，最后针对要清洗的数据启动任务即可。具体步骤：1.定义数据质量规则 2.配置清洗策略 3.执行清洗任务 4.验证清洗结果

✅ 直接匹配成功
==============================
```

## 扩展功能

### 添加新知识

```python
qa_system = DataPlatformQASystem()
qa_system.qa_model.add_knowledge("新问题", "新答案")
```

### 自定义模型路径

```python
from src.data_platform.simple_qa_model import SimpleDataPlatformQAModel

# 使用自定义模型路径
model = SimpleDataPlatformQAModel(model_path="your/model/path")
```

## 技术特点

- **轻量级**: 不依赖复杂的外部库（除了基础的PyTorch和Transformers）
- **高效**: 预计算知识库问题的向量表示，提高查询速度
- **可扩展**: 支持动态添加新的知识条目
- **容错性**: 多层级的错误处理和回退机制

## 注意事项

1. 首次运行时需要加载BERT模型，可能需要一些时间
2. 如果BERT模型加载失败，系统会自动回退到基础知识库模式
3. 问答效果依赖于知识库的完整性，可根据需要扩展知识库内容
4. 建议在有GPU的环境下运行以获得更好的性能

## 故障排除

### 模型加载失败
- 检查模型路径是否正确
- 确保有足够的内存空间
- 系统会自动回退到基础模式

### 导入错误
- 确保在正确的目录下运行脚本
- 检查Python路径配置
- 安装必要的依赖包

## 更新日志

- v1.0.0: 初始版本，支持基础问答功能
- 支持BERT语义匹配
- 提供交互式界面
- 包含数据平台相关知识库
