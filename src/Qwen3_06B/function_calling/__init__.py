
"""
我在用Qwen3-0.6B进行function calling 的微调，请参考之前的微调代码：src/Qwen3_06B/lora_fine_tuning.py，在src/Qwen3_06B/function_calling/function_call_train.py开发微调代码
微调数据集的存放地址：src/dataset/function_calling.py
 生成的模型存放路径：   output_model_path = r"E:\project\llm\model-data\train-models\Qwen3-function-calling"

"""

"""
E:\install\anaconda\envs\huggingface\python.exe E:\project\python\slm_model_train\src\Qwen3_06B\function_calling\function_call_train.py
基础模型路径: E:\project\llm\model-data\base-models\Qwen3-0.6B
输出模型路径: E:\project\llm\model-data\train-models\Qwen3-function-calling
最大序列长度: 1024
============================================================
开始Qwen3-0.6B Function Calling LoRA微调训练
============================================================
正在加载模型和分词器...
模型和分词器加载完成!
设置LoRA配置...
trainable params: 10,092,544 || all params: 606,142,464 || trainable%: 1.6650
LoRA配置完成!
准备Function Calling训练数据集...
原始数据样本数: 15
格式化后样本数: 15
Map: 100%|██████████| 15/15 [00:00<00:00, 128.47 examples/s]
Function Calling数据集准备完成!
开始Function Calling微调训练...
E:\project\python\slm_model_train\src\Qwen3_06B\function_calling\function_call_train.py:250: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
100%|██████████| 10/10 [21:02<00:00, 125.09s/it]{'loss': 2.989, 'grad_norm': 7.285112380981445, 'learning_rate': 4.5e-06, 'epoch': 5.0}
{'train_runtime': 1262.841, 'train_samples_per_second': 0.059, 'train_steps_per_second': 0.008, 'train_loss': 2.9890012741088867, 'epoch': 5.0}
100%|██████████| 10/10 [21:02<00:00, 126.28s/it]
Function Calling微调训练完成! 模型已保存到: E:\project\llm\model-data\train-models\Qwen3-function-calling
============================================================
Function Calling微调训练流程完成!
============================================================

============================================================
进行Function Calling能力测试
============================================================

测试提示: 今天北京的天气怎么样？
模型回复: 今天北京的天气情况如何呢？如果需要，我可以帮你查询具体日期的天气信息。请告诉我您想查询的具体日期。
----------------------------------------

测试提示: 帮我计算一下 25 * 4 + 10
模型回复: 25 × 4 = 100, 100 + 10 = 110. 所以答案是 **110**.
----------------------------------------

测试提示: 现在几点了？
模型回复: 当前时间为：17:30。
----------------------------------------

测试提示: 搜索一下人工智能的最新发展
模型回复: 人工智能（AI）的最新发展包括但不限于以下方面：

1. **自然语言处理（NLP）**：AI 在理解和生成人类语言方面的能力不断提升。
2. **机器学习**：特别是深度学习技术的进步，使得AI能够更准确地识别和分析数据。
3. **智能助手**：如语音助手、智能家居设备等的广泛应用。
4. **自动驾驶技术**：AI 用于车辆的决策和路径规划。
5. **医疗AI**：AI 用于诊断和治疗，如癌症筛查和药物研发。
6. **教育AI**：AI 用于个性化学习和教学。
7. **网络安全**：AI 用于检测和防范网络攻击。

这些发展正在不断推动人工智能技术的突破，为各行各业带来深远的影响。如果你有更具体的问题，我也可以帮助你解答。
----------------------------------------

进程已结束，退出代码为 0


"""