#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的BERT LoRA模型测试脚本
针对已训练好的模型进行推理测试
"""

import torch
from transformers import BertTokenizer, BertModel

from src.lora.bert_base_chinese.peft_lora_data_platform import BertForInstructionTuning
from peft import PeftModel
import os


def test_model():
    """测试训练好的模型"""
    # 模型路径
    model_path = r"E:\project\llm\lora\peft_lora_mixed"
    base_model_name = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    print("=" * 50)
    print("BERT LoRA 模型测试")
    print("=" * 50)
    print(f"模型路径: {model_path}")

    try:
        # 1. 加载tokenizer
        print("正在加载tokenizer...")
        if os.path.exists(model_path):
            tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        print("✓ tokenizer加载成功")

        # 2. 加载基础模型
        print("正在加载基础模型...")
        bert_model = BertModel.from_pretrained(base_model_name)
        print("✓ 基础模型加载成功")

        # 3. 创建自定义模型包装器
        print("正在创建模型包装器...")
        model = BertForInstructionTuning(bert_model)
        
        # 4. 加载LoRA适配器（如果存在）
        if os.path.exists(model_path):
            print("正在加载LoRA适配器...")
            try:
                model = PeftModel.from_pretrained(model, model_path)
                print("✓ LoRA适配器加载成功")
            except Exception as e:
                print(f"⚠️ LoRA适配器加载失败，使用基础模型: {e}")
        else:
            print("⚠️ 模型路径不存在，使用基础模型")
        
        model.eval()
        print("✓ 模型准备完成")

        # 4. 测试数据
        test_texts = [
            "这个数据平台怎么查询销售数据？",
            "请帮我统计一下上个月的营业额",
            "系统登录不了怎么办？",
            "如何导出客户信息表？",
            "数据库连接失败"
        ]

        print("\n" + "=" * 50)
        print("开始推理测试")
        print("=" * 50)

        # 5. 逐个测试
        for i, text in enumerate(test_texts, 1):
            print(f"\n测试 {i}: {text}")
            print("-" * 30)

            # 编码输入
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            # 推理
            with torch.no_grad():
                outputs = model(**inputs)
                
                # 处理模型输出
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # 检查logits的维度
                print(f"Logits shape: {logits.shape}")
                
                # 如果是语言建模输出 (batch_size, seq_len, vocab_size)
                if len(logits.shape) == 3:
                    # 取最后一个token的logits作为分类结果
                    # 或者可以取平均值
                    last_token_logits = logits[:, -1, :]  # 取最后一个token
                    
                    # 为了演示，我们假设只有2个类别，取前2个维度
                    classification_logits = last_token_logits[:, :2]
                    probabilities = torch.softmax(classification_logits, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    print(f"预测类别: {predicted_class}")
                    print(f"置信度: {confidence:.4f}")
                    print(f"概率分布: [类别0: {probabilities[0][0]:.4f}, 类别1: {probabilities[0][1]:.4f}]")
                    
                elif len(logits.shape) == 2:
                    # 如果是分类输出 (batch_size, num_classes)
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    print(f"预测类别: {predicted_class}")
                    print(f"置信度: {confidence:.4f}")
                    if probabilities.shape[1] >= 2:
                        print(f"概率分布: [类别0: {probabilities[0][0]:.4f}, 类别1: {probabilities[0][1]:.4f}]")
                    else:
                        print(f"单类别输出: {probabilities[0][0]:.4f}")
                else:
                    # 其他情况
                    print(f"未知的logits形状: {logits.shape}")
                    print("无法处理此输出格式")

        print("\n" + "=" * 50)
        print("测试完成！")
        print("=" * 50)

        # 6. 交互式测试
        print("\n进入交互式测试模式（输入 'quit' 退出）:")
        while True:
            user_input = input("\n请输入测试文本: ").strip()
            if user_input.lower() == 'quit':
                break

            if not user_input:
                continue

            try:
                inputs = tokenizer(
                    user_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                messages = [
                    {"role": "user", "content": inputs}
                ]
                with torch.no_grad():
                    outputs = model(messages)
                    
                    # 处理模型输出
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    # 检查logits的维度
                    print(f"Logits shape: {logits.shape}")
                    
                    # 如果是语言建模输出 (batch_size, seq_len, vocab_size)
                    if len(logits.shape) == 3:
                        # 取最后一个token的logits作为分类结果
                        last_token_logits = logits[:, -1, :]  # 取最后一个token
                        
                        # 为了演示，我们假设只有2个类别，取前2个维度
                        classification_logits = last_token_logits[:, :2]
                        probabilities = torch.softmax(classification_logits, dim=-1)
                        predicted_class = torch.argmax(probabilities, dim=-1).item()
                        confidence = probabilities[0][predicted_class].item()
                        
                        print(f"预测类别: {predicted_class}")
                        print(f"置信度: {confidence:.4f}")
                        print(f"概率分布: [类别0: {probabilities[0][0]:.4f}, 类别1: {probabilities[0][1]:.4f}]")
                        
                    elif len(logits.shape) == 2:
                        # 如果是分类输出 (batch_size, num_classes)
                        probabilities = torch.softmax(logits, dim=-1)
                        predicted_class = torch.argmax(probabilities, dim=-1).item()
                        confidence = probabilities[0][predicted_class].item()
                        
                        print(f"预测类别: {predicted_class}")
                        print(f"置信度: {confidence:.4f}")
                        if probabilities.shape[1] >= 2:
                            print(f"概率分布: [类别0: {probabilities[0][0]:.4f}, 类别1: {probabilities[0][1]:.4f}]")
                        else:
                            print(f"单类别输出: {probabilities[0][0]:.4f}")
                    else:
                        # 其他情况
                        print(f"未知的logits形状: {logits.shape}")
                        print("无法处理此输出格式")

            except Exception as e:
                print(f"推理错误: {e}")

        print("退出测试")

    except Exception as e:
        print(f"❌ 模型加载或测试失败: {e}")
        print("请检查:")
        print("1. 模型路径是否正确")
        print("2. 是否安装了必要的依赖包 (torch, transformers, peft)")
        print("3. 模型文件是否完整")


if __name__ == "__main__":
    test_model()
