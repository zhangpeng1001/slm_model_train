





"""
E:\install\anaconda\envs\huggingface\python.exe E:\project\python\slm_model_train\src\Qwen3_06B\lora_fine_tuning.py
基础模型路径: E:\project\llm\model-data\base-models\Qwen3-0.6B
输出模型路径: E:\project\llm\model-data\train-models\Qwen3-0.6B
==================================================
开始Qwen3-0.6B LoRA微调训练
==================================================
正在加载模型和分词器...
模型和分词器加载完成!
设置LoRA配置...
trainable params: 5,046,272 || all params: 601,096,192 || trainable%: 0.8395
LoRA配置完成!
准备训练数据集...
总训练样本数: 177
Map: 100%|██████████| 177/177 [00:00<00:00, 1165.37 examples/s]
数据集准备完成!
开始训练...
E:\project\python\slm_model_train\src\Qwen3_06B\lora_fine_tuning.py:201: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
 14%|█▍        | 10/69 [10:55<1:11:19, 72.53s/it]{'loss': 5.4448, 'grad_norm': 19.964406967163086, 'learning_rate': 1.8e-05, 'epoch': 0.45}
 29%|██▉       | 20/69 [21:21<47:21, 57.99s/it]{'loss': 3.7822, 'grad_norm': 8.013687133789062, 'learning_rate': 3.8e-05, 'epoch': 0.9}
 43%|████▎     | 30/69 [29:57<36:03, 55.47s/it]{'loss': 2.3764, 'grad_norm': 6.840965747833252, 'learning_rate': 5.8e-05, 'epoch': 1.31}
 58%|█████▊    | 40/69 [39:30<27:50, 57.61s/it]{'loss': 1.654, 'grad_norm': 5.4495744705200195, 'learning_rate': 7.800000000000001e-05, 'epoch': 1.76}
 72%|███████▏  | 50/69 [48:50<18:13, 57.57s/it]{'loss': 1.0789, 'grad_norm': 4.273277759552002, 'learning_rate': 9.8e-05, 'epoch': 2.18}
 87%|████████▋ | 60/69 [59:04<09:11, 61.33s/it]{'loss': 0.7682, 'grad_norm': 5.881645679473877, 'learning_rate': 0.000118, 'epoch': 2.63}
100%|██████████| 69/69 [1:07:25<00:00, 58.63s/it]
{'train_runtime': 4045.3368, 'train_samples_per_second': 0.131, 'train_steps_per_second': 0.017, 'train_loss': 2.271136608676634, 'epoch': 3.0}
训练完成! 模型已保存到: E:\project\llm\model-data\train-models\Qwen3-0.6B
==================================================
训练流程完成!
==================================================

进程已结束，退出代码为 0


"""