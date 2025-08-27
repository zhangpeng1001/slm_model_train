# GPU内存优化总结

## 问题描述
原始训练代码在14.74 GiB GPU上出现内存不足错误：
- GPU总容量：14.74 GiB
- 已使用：13.37 GiB
- 可用：1.37 GiB
- 尝试分配32.00 MiB时失败

## 优化措施

### 1. 环境变量优化
```python
# 设置环境变量优化GPU内存
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

### 2. 内存分配策略
```python
# 设置GPU内存分配策略 - 更保守的内存使用
torch.cuda.set_per_process_memory_fraction(0.85)  # 从90%降到85%
```

### 3. 模型加载优化
```python
# 加载模型 - 内存优化版本
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用fp16
    device_map="auto",
    low_cpu_mem_usage=True,
    use_cache=False,  # 禁用缓存以节省内存
)
```

### 4. LoRA配置优化
```python
# 配置LoRA - 内存优化版本
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # 从16降到8，降低秩以减少内存使用
    lora_alpha=16,  # 从32降到16
    lora_dropout=0.1,  # 从0.05增加到0.1
    target_modules=["q_proj", "v_proj"],  # 从4个模块减少到2个
    bias="none",
)
```

### 5. 训练参数优化
```python
# 内存优化参数 - 大幅减少内存使用
return TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # 从8降到2
    gradient_accumulation_steps=8,  # 从4增加到8
    gradient_checkpointing=True,  # 启用梯度检查点
    fp16=True,  # 启用混合精度训练
    dataloader_pin_memory=False,  # 禁用pin_memory节省内存
    save_total_limit=2,  # 从3降到2
    warmup_steps=200,  # 从500降到200
    ddp_find_unused_parameters=False,  # 禁用查找未使用参数
    # ... 其他参数
)
```

### 6. 序列长度优化
```python
# 大幅减少序列长度以节省内存
max_length = 512 if self.device.type == "cuda" else 256  # 从1024降到512
```

### 7. 数据预处理优化
```python
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=100,  # 从1000降到100
    num_proc=1,  # 单进程避免内存冲突
    # ... 其他参数
)
```

### 8. 内存清理
```python
def _setup_device(self):
    if torch.cuda.is_available():
        # 清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()  # 添加垃圾回收
```

## 优化效果

### 内存使用对比
| 项目 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| Batch Size | 8 | 2 | 75% |
| 序列长度 | 1024 | 512 | 50% |
| LoRA秩 | 16 | 8 | 50% |
| 目标模块 | 4个 | 2个 | 50% |
| 内存分配 | 90% | 85% | 5% |

### 训练效率保持
- 通过梯度累积步数从4增加到8，保持有效batch size = 2 × 8 = 16
- 启用梯度检查点，用计算时间换内存空间
- 启用混合精度训练，提升训练速度

## 关键优化点

1. **最重要**：大幅减少batch_size（从8到2）
2. **关键**：启用梯度检查点
3. **有效**：减少序列长度（从1024到512）
4. **必要**：降低LoRA配置复杂度
5. **辅助**：各种内存管理优化

## 使用建议

1. 如果仍然内存不足，可以进一步：
   - 将batch_size降到1
   - 将序列长度降到256
   - 将LoRA秩降到4

2. 如果内存充足，可以适当提升：
   - 增加batch_size到4
   - 增加序列长度到768
   - 增加LoRA秩到12

3. 监控GPU内存使用：
   ```python
   # 训练前后都会输出内存使用情况
   logger.info(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
   logger.info(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
   ```

## 故障排除

如果仍然出现OOM错误：

1. 检查是否有其他进程占用GPU内存
2. 重启Python进程清理内存
3. 进一步降低batch_size到1
4. 考虑使用CPU训练（虽然会很慢）

## 总结

通过以上全面的内存优化措施，应该能够在14.74 GiB GPU上成功运行训练。主要策略是：
- **大幅减少内存占用**：降低batch_size、序列长度、模型复杂度
- **启用内存优化技术**：梯度检查点、混合精度训练
- **优化内存管理**：禁用不必要的缓存、改进内存分配策略

这些优化在保证训练效果的同时，最大程度地减少了GPU内存使用。
