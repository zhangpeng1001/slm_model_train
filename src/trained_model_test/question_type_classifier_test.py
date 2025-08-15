"""
问题类型分类器测试
用于训练和测试问题类型分类模型（问题回答 vs 任务处理）
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification


def test_question_classifier(model_path, test_cases, device=None):
    """
    测试训练好的问题类型分类模型

    参数:
    - model_path: 模型保存路径
    - test_cases: 测试用例列表
    - device: 运行设备，默认自动选择
    """
    # 自动选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()  # 设置为评估模式

    # 类别映射
    id2label = {
        0: "问题回答",
        1: "任务处理"
    }

    correct = 0
    total = len(test_cases)

    print(f"开始测试，共{total}个测试用例，使用设备: {device}\n")

    with torch.no_grad():  # 关闭梯度计算
        for i, (question, expected_label) in enumerate(test_cases, 1):
            # 文本编码
            inputs = tokenizer(
                question,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            # 模型预测
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            predicted_label = id2label.get(predicted_class_id, f"未知类别({predicted_class_id})")

            # 判断是否正确
            is_correct = predicted_label == expected_label
            if is_correct:
                correct += 1

            # 输出结果
            print(f"测试用例 {i}:")
            print(f"问题: {question}")
            print(f"预测类别: {predicted_label}")
            print(f"预期类别: {expected_label}")
            print(f"结果: {'正确' if is_correct else '错误'}\n")

    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    print(f"测试完成，总准确率: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


if __name__ == "__main__":
    # 模型路径（根据你的实际路径修改）
    model_path = r"E:\project\python\slm_model_train\src\train_data\trained_question_type_classifier\question_type_classifier"

    test_samples = [
        # 问题回答类型样本
        ("我有一批西安市地类图斑的数据，怎么进行发服务", "问题回答"),
        ("DEM高程数据怎么进行质量检查", "问题回答"),
        ("矢量地形图数据如何入库", "问题回答"),
        ("我的点云数据处理完了，如何发布服务", "问题回答"),
        ("地理信息数据库的数据怎么备份", "问题回答"),
        ("卫星遥感数据的预处理流程是什么", "问题回答"),
        ("无人机航拍数据如何进行几何纠正", "问题回答"),
        ("土地利用现状数据怎么更新", "问题回答"),
        ("基础地理信息数据如何管理", "问题回答"),

        # 任务处理类型样本
        ("请帮我把实景三维模型成果数据进行治理", "任务处理"),
        ("我已经上传了单波段浮点投影的数据，现在想进行入库", "任务处理"),
        ("有一些遥感影像数据需要处理", "任务处理"),
        ("正射影像数据需要进行坐标转换", "任务处理"),
        ("激光雷达数据进行格式转换", "任务处理"),
        ("地籍调查数据需要标准化处理", "任务处理"),
        ("处理地下管线探测数据", "任务处理"),
        ("倾斜摄影测量数据进行建模", "任务处理")
    ]

    # 运行测试
    test_results = test_question_classifier(model_path, test_samples)
