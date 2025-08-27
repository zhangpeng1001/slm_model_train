# Qwen3 多任务模型推理使用指南

## 概述

本指南介绍如何使用训练完成的Qwen3多任务模型进行推理。提供了多种推理方式，包括程序化调用、交互式推理和批量处理。

## 文件说明

### 推理相关文件
- **`inference_multi_task.py`** - 核心推理类，提供多任务推理功能
- **`interactive_inference.py`** - 交互式推理脚本，支持命令行交互
- **`inference_config.py`** - 推理配置文件，包含各种参数设置
- **`Inference_Guide.md`** - 本使用指南

## 支持的任务类型

### 1. 数据提取 (data_extraction)
- **功能**: 从文本中提取数据名称
- **输入示例**: "我需要获取销售数据和用户行为数据"
- **输出示例**: "销售数据, 用户行为数据"

### 2. 问答 (dp_qa)
- **功能**: 回答数据平台相关问题
- **输入示例**: "数据平台如何进行数据质量检查？"
- **输出示例**: "数据平台通过以下方式进行数据质量检查..."

### 3. 问题分类 (question_classifier)
- **功能**: 判断问题类型（数据平台相关、通用对话、无关问题）
- **输入示例**: "今天天气怎么样？"
- **输出示例**: "无关问题"

### 4. 问题类型分类 (question_type_classifier)
- **功能**: 判断问题类型（问题回答、任务处理）
- **输入示例**: "帮我查询一下用户数据"
- **输出示例**: "任务处理"

### 5. 工具调用 (tool_data_platform)
- **功能**: 根据需求调用相应的数据平台工具
- **输入示例**: "我需要采集电商平台的销售数据，时间范围是最近一个月"
- **输出示例**: 
```json
<FunctionCall>
{"name":"get_data_collection","parameters":{"data_source":"电商平台","data_type":"销售数据","time_range":"最近一个月","business_platform":"电商"}}
</FunctionCall>
```

## 使用方法

### 1. 环境准备

确保已安装必要的依赖：
```bash
pip install torch transformers datasets peft
```

### 2. 配置路径

编辑 `inference_config.py` 文件，修改模型路径：
```python
# 基础模型路径
BASE_MODEL_PATH = "/path/to/your/Qwen3-0.6B"

# 训练后的LoRA模型路径
PEFT_MODEL_PATH = "/path/to/your/trained/model"
```

### 3. 程序化调用

```python
from inference_multi_task import MultiTaskInference

# 初始化推理器
inferencer = MultiTaskInference(
    base_model_path="/path/to/Qwen3-0.6B",
    peft_model_path="/path/to/trained/model",
    device="auto"  # 自动选择设备
)

# 数据提取
result = inferencer.data_extraction("我需要获取销售数据和用户行为数据")
print(result)

# 问答
result = inferencer.question_answer("数据平台如何进行数据质量检查？")
print(result)

# 问题分类
result = inferencer.question_classify("今天天气怎么样？")
print(result)

# 工具调用
result = inferencer.tool_calling("我需要采集电商平台的销售数据")
print(result)
```

### 4. 交互式推理

```bash
# Linux/Mac
python3 interactive_inference.py

# Windows
python interactive_inference.py

# 指定参数
python interactive_inference.py --base_model /path/to/model --device cuda
```

交互式推理提供友好的菜单界面：
```
==================================================
Qwen3 多任务推理系统
==================================================
请选择任务类型:
1. 数据提取
2. 问答
3. 问题分类
4. 问题类型分类
5. 工具调用
6. 自定义任务
0. 退出
==================================================
```

### 5. 批量推理

```python
from inference_multi_task import MultiTaskInference

inferencer = MultiTaskInference(
    base_model_path="/path/to/Qwen3-0.6B",
    peft_model_path="/path/to/trained/model"
)

# 批量输入
inputs = [
    {"input": "我需要销售数据", "task_type": "data_extraction"},
    {"input": "如何清洗数据？", "task_type": "dp_qa"},
    {"input": "今天天气如何？", "task_type": "question_classifier"}
]

# 批量推理
results = inferencer.batch_inference(inputs)
for i, result in enumerate(results):
    print(f"输入 {i+1}: {inputs[i]['input']}")
    print(f"输出 {i+1}: {result}")
    print("-" * 50)
```

## 高级配置

### 1. 生成参数调整

可以在 `inference_config.py` 中调整不同任务的生成参数：

```python
GENERATION_CONFIGS = {
    "data_extraction": {
        "max_length": 256,      # 最大生成长度
        "temperature": 0.3,     # 温度参数（越小越确定）
        "top_p": 0.8,          # Top-p采样
        "do_sample": False,     # 是否采样
        "repetition_penalty": 1.1,  # 重复惩罚
    },
    # ... 其他任务配置
}
```

### 2. 自定义任务

```python
# 使用通用生成方法
response = inferencer.generate_response(
    user_input="你的问题",
    task_type="custom_task_type",
    max_length=512,
    temperature=0.7,
    top_p=0.9
)
```

### 3. 设备选择

```python
# 自动选择设备
inferencer = MultiTaskInference(..., device="auto")

# 强制使用GPU
inferencer = MultiTaskInference(..., device="cuda")

# 强制使用CPU
inferencer = MultiTaskInference(..., device="cpu")
```

## 性能优化建议

### 1. GPU推理优化
- 使用GPU可以显著提升推理速度
- 启用混合精度推理（FP16）减少内存占用
- 批量处理多个请求提高吞吐量

### 2. 内存优化
- 对于长文本，适当调整max_length参数
- 使用较小的batch_size避免内存溢出
- 及时清理不需要的变量释放内存

### 3. 推理速度优化
- 使用较低的temperature值可以加快生成速度
- 关闭采样（do_sample=False）可以提高确定性任务的速度
- 预加载模型避免重复加载开销

## 故障排除

### 1. 模型加载错误
**错误**: `Model not found` 或 `Path does not exist`
**解决**: 检查 `inference_config.py` 中的路径配置是否正确

### 2. 内存不足错误
**错误**: `CUDA out of memory` 或 `RuntimeError: out of memory`
**解决**: 
- 减少max_length参数
- 使用CPU推理
- 减少batch_size

### 3. 设备相关错误
**错误**: `CUDA device not available`
**解决**: 
- 检查CUDA安装
- 使用device="cpu"强制CPU推理
- 检查GPU驱动

### 4. 依赖包错误
**错误**: `ModuleNotFoundError`
**解决**: 
```bash
pip install torch transformers datasets peft
```

## 示例脚本

### 简单推理示例
```python
# simple_inference_example.py
from inference_multi_task import MultiTaskInference

def main():
    # 初始化推理器
    inferencer = MultiTaskInference(
        base_model_path="/path/to/Qwen3-0.6B",
        peft_model_path="/path/to/trained/model",
        device="auto"
    )
    
    # 测试各种任务
    test_cases = [
        ("数据提取", "我需要用户数据和订单数据", "data_extraction"),
        ("问答", "如何进行数据清洗？", "dp_qa"),
        ("分类", "今天天气不错", "question_classifier"),
        ("工具调用", "帮我采集销售数据", "tool_data_platform")
    ]
    
    for task_name, question, task_type in test_cases:
        print(f"\n=== {task_name} ===")
        print(f"输入: {question}")
        
        response = inferencer.generate_response(
            user_input=question,
            task_type=task_type
        )
        
        print(f"输出: {response}")

if __name__ == "__main__":
    main()
```

### 批量处理示例
```python
# batch_inference_example.py
from inference_multi_task import MultiTaskInference
import json

def process_batch_file(input_file, output_file):
    """处理批量推理文件"""
    
    # 初始化推理器
    inferencer = MultiTaskInference(
        base_model_path="/path/to/Qwen3-0.6B",
        peft_model_path="/path/to/trained/model"
    )
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        inputs = json.load(f)
    
    # 批量推理
    results = []
    for item in inputs:
        response = inferencer.generate_response(
            user_input=item['input'],
            task_type=item.get('task_type', 'dp_qa')
        )
        
        results.append({
            'input': item['input'],
            'task_type': item.get('task_type', 'dp_qa'),
            'output': response
        })
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"批量推理完成，结果保存到: {output_file}")

if __name__ == "__main__":
    process_batch_file("input.json", "output.json")
```

## API参考

### MultiTaskInference类

#### 初始化参数
- `base_model_path`: 基础模型路径
- `peft_model_path`: LoRA模型路径  
- `device`: 推理设备 ("auto", "cuda", "cpu")

#### 主要方法

**generate_response(user_input, task_type, instruction, max_length, temperature, top_p, do_sample)**
- 通用生成方法
- 返回: 生成的响应文本

**data_extraction(text)**
- 数据提取任务
- 返回: 提取的数据名称

**question_answer(question, instruction)**
- 问答任务
- 返回: 问题答案

**question_classify(question)**
- 问题分类任务
- 返回: 分类结果

**question_type_classify(question)**
- 问题类型分类任务
- 返回: 类型分类结果

**tool_calling(instruction)**
- 工具调用任务
- 返回: 函数调用格式

**batch_inference(inputs)**
- 批量推理
- 参数: 输入列表
- 返回: 响应列表

## 最佳实践

### 1. 任务选择
- 根据具体需求选择合适的任务类型
- 对于不确定的任务，可以先使用问题分类确定类型
- 工具调用任务需要明确的指令描述

### 2. 参数调优
- 分类任务使用低temperature (0.1-0.3)
- 生成任务使用中等temperature (0.5-0.8)
- 创作任务可以使用高temperature (0.8-1.0)

### 3. 性能监控
- 记录推理时间和内存使用
- 监控模型输出质量
- 定期评估不同任务的表现

### 4. 错误处理
- 实现重试机制
- 记录失败案例
- 提供降级方案

---

**更新时间**: 2024年8月26日  
**版本**: v1.0  
**适用模型**: Qwen3-0.6B + LoRA微调  
**维护状态**: 活跃维护
