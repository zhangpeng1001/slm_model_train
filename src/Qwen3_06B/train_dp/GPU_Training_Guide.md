# Qwen3 多任务训练 GPU优化版使用指南

## 概述

本指南介绍如何在Linux环境下使用T4 GPU进行Qwen3-0.6B模型的多任务微调训练。相比原始的CPU版本，GPU版本在性能、内存管理和训练效率方面都进行了大幅优化。

## 主要优化内容

### 1. GPU环境适配
- **设备自动检测**: 自动检测CUDA可用性并选择最佳设备
- **内存管理**: 智能GPU内存分配和缓存管理
- **混合精度训练**: 使用FP16减少内存占用并加速训练
- **设备映射**: 自动模型设备映射优化

### 2. 训练参数优化
- **批处理大小**: T4 GPU优化的batch size (8)
- **梯度累积**: 4步梯度累积增加有效批处理大小
- **序列长度**: GPU环境支持更长序列 (1024 tokens)
- **学习率调度**: 余弦学习率调度策略
- **优化器**: AdamW优化器配置

### 3. LoRA配置优化
- **更高秩**: r=16 (相比CPU版本的r=8)
- **更多目标模块**: 包含q_proj, v_proj, k_proj, o_proj
- **优化dropout**: 降低至0.05以提高训练稳定性

### 4. 数据处理优化
- **多进程加载**: 4个数据加载进程
- **批量预处理**: GPU环境使用更大的预处理批次
- **内存对齐**: 8的倍数填充优化GPU计算

## 文件结构

```
src/Qwen3_06B/train_dp/
├── multi_task_train_linux_gpu.py  # GPU优化的主训练脚本
├── config_gpu.py                  # GPU训练配置文件
├── run_gpu_training.sh            # 启动脚本
├── GPU_Training_Guide.md          # 本使用指南
├── logs/                          # 训练日志目录
└── results/                       # 训练结果目录
```

## 环境要求

### 硬件要求
- **GPU**: NVIDIA T4 (16GB显存) 或同等性能GPU
- **内存**: 建议32GB以上系统内存
- **存储**: 至少50GB可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+推荐)
- **Python**: 3.8+
- **CUDA**: 11.0+
- **驱动**: NVIDIA驱动版本450+

### Python依赖
```bash
pip install torch>=1.12.0 transformers>=4.21.0 datasets>=2.0.0 peft>=0.3.0
```

## 配置说明

### 1. 路径配置 (config_gpu.py)

在使用前，请修改以下路径配置：

```python
# 模型路径 - 修改为你的Qwen3-0.6B模型路径
MODEL_NAME = "/home/user/models/Qwen3-0.6B"

# 输出路径 - 修改为你希望保存训练结果的路径
OUTPUT_MODEL_PATH = "/home/user/train-models/Qwen3-multi-task"

# 数据集路径 - 修改为你的数据集存放路径
DATASET_BASE_PATH = "/home/user/dp_dataset"
```

### 2. GPU配置调整

根据你的GPU内存大小调整以下参数：

```python
GPU_CONFIG = {
    "memory_fraction": 0.9,     # GPU内存使用比例
    "batch_size": 8,            # 批处理大小 (T4: 8, V100: 16)
    "gradient_accumulation_steps": 4,  # 梯度累积步数
    "max_length": 1024,         # 最大序列长度
    "fp16": True,              # 混合精度训练
    "dataloader_workers": 4,    # 数据加载进程数
}
```

### 3. 内存不足时的调整建议

如果遇到GPU内存不足 (OOM) 错误，可以尝试以下调整：

1. **减少批处理大小**:
   ```python
   "batch_size": 4,  # 从8减少到4
   ```

2. **增加梯度累积步数**:
   ```python
   "gradient_accumulation_steps": 8,  # 从4增加到8
   ```

3. **减少序列长度**:
   ```python
   "max_length": 512,  # 从1024减少到512
   ```

4. **降低LoRA秩**:
   ```python
   LORA_CONFIG = {
       "r": 8,  # 从16减少到8
   }
   ```

## 使用方法

### 1. 环境检查

首先运行环境检查确保所有配置正确：

```bash
cd src/Qwen3_06B/train_dp/
python3 config_gpu.py
```

### 2. 启动训练

#### 方法一：使用启动脚本 (推荐)
```bash
chmod +x run_gpu_training.sh
./run_gpu_training.sh
```

#### 方法二：直接运行Python脚本
```bash
python3 multi_task_train_linux_gpu.py
```

### 3. 监控训练过程

训练过程中可以通过以下命令监控GPU使用情况：

```bash
# 实时监控GPU状态
watch -n 1 nvidia-smi

# 查看训练日志
tail -f logs/training_*.log
```

## 性能对比

| 指标 | CPU版本 | GPU版本 (T4) | 提升倍数 |
|------|---------|--------------|----------|
| 批处理大小 | 2 | 8 | 4x |
| 序列长度 | 512 | 1024 | 2x |
| 训练轮数 | 10 | 3 | - |
| 预计训练时间 | 8-12小时 | 1-2小时 | 4-6x |
| 内存使用 | 8-16GB (系统) | 12-14GB (GPU) | - |

## 故障排除

### 1. CUDA相关错误

**错误**: `CUDA out of memory`
**解决**: 
- 减少batch_size
- 增加gradient_accumulation_steps
- 减少max_length

**错误**: `CUDA driver version is insufficient`
**解决**: 更新NVIDIA驱动程序

### 2. 模型加载错误

**错误**: `Model not found`
**解决**: 检查config_gpu.py中的MODEL_NAME路径

**错误**: `Permission denied`
**解决**: 确保有读取模型文件的权限

### 3. 数据集错误

**错误**: `Dataset file not found`
**解决**: 
- 检查config_gpu.py中的DATASET_BASE_PATH路径
- 确保所有数据集文件存在于指定路径

### 4. 权限错误

**错误**: `Permission denied when writing to output directory`
**解决**: 
```bash
sudo chown -R $USER:$USER /home/user/train-models/
chmod -R 755 /home/user/train-models/
```

### 5. 训练过程中断

**错误**: 训练意外中断
**解决**: 
- 检查GPU温度是否过高
- 检查系统内存是否充足
- 查看训练日志获取详细错误信息

## 高级配置

### 1. 多GPU训练 (如果有多个GPU)

修改启动脚本中的环境变量：
```bash
export CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和1
```

并在训练参数中启用数据并行：
```python
# 在TrainingArguments中添加
dataloader_pin_memory=True,
dataloader_persistent_workers=True,
```

### 2. 梯度检查点 (节省内存)

如果内存仍然不足，可以启用梯度检查点：
```python
# 在TrainingArguments中添加
gradient_checkpointing=True,
```

### 3. 自定义学习率调度

```python
TRAINING_CONFIG = {
    "lr_scheduler_type": "polynomial",  # 或 "linear", "constant"
    "polynomial_decay_power": 1.0,
    # ... 其他参数
}
```

## 最佳实践

### 1. 训练前准备
1. 确保GPU驱动和CUDA版本兼容
2. 运行环境检查脚本验证配置
3. 备份重要数据和配置文件
4. 预估训练时间和资源需求

### 2. 训练过程中
1. 定期监控GPU使用率和温度
2. 观察训练损失变化趋势
3. 保存训练日志便于问题排查
4. 适时调整学习率和其他超参数

### 3. 训练后处理
1. 验证模型输出质量
2. 保存最佳检查点
3. 清理临时文件和缓存
4. 记录训练配置和结果

## 常见问题FAQ

### Q1: 为什么GPU版本训练轮数更少？
A1: GPU版本使用了更优的训练参数和更大的有效批处理大小，可以在更少的轮数内达到相同或更好的效果。

### Q2: 如何判断训练是否收敛？
A2: 观察训练损失是否稳定下降并趋于平缓，同时可以通过验证集评估模型性能。

### Q3: 可以在训练过程中修改参数吗？
A3: 不建议在训练过程中修改参数，这可能导致训练不稳定。如需调整，建议停止训练后重新开始。

### Q4: 如何选择合适的LoRA参数？
A4: 一般来说，r值越大模型容量越大但训练越慢；alpha/r比例建议保持在2左右；dropout可以根据数据集大小调整。

### Q5: 训练完成后如何使用模型？
A5: 训练完成的模型会保存在OUTPUT_MODEL_PATH指定的路径，可以使用transformers库直接加载使用。

## 技术支持

如果遇到问题，请按以下顺序排查：
1. 查看训练日志文件
2. 运行环境检查脚本
3. 检查GPU状态和驱动
4. 验证配置文件设置
5. 查阅本文档的故障排除部分

## 更新日志

- **v1.0** (2024-01): 初始GPU优化版本
- 支持T4 GPU训练
- 混合精度训练
- 智能内存管理
- 多进程数据加载

---

**注意**: 本指南基于T4 GPU环境编写，其他GPU型号可能需要调整相应参数。建议在正式训练前先进行小规模测试。
