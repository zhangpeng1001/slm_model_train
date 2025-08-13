"""
分类器训练和测试脚本
提供完整的分类器训练、测试和使用流程
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train.question_classifier_trainer import QuestionClassifierTrainer
from src.controller.question_classifier import QuestionClassifier
from src.controller.intelligent_qa_system import IntelligentQASystem


def train_classifier():
    """训练问题分类器"""
    print("=" * 60)
    print("开始训练问题分类器")
    print("=" * 60)

    # 模型路径
    model_path = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = QuestionClassifierTrainer(model_path)

    # 开始训练
    trainer.train(
        epochs=3,  # 分类任务通常不需要太多轮次
        batch_size=4,  # CPU训练使用较小的batch size
        learning_rate=2e-5
    )

    # 评估模型
    texts, labels = trainer.prepare_training_data()
    accuracy = trainer.evaluate(texts, labels)

    # 保存模型
    save_path = trainer.save_model('question_classifier')

    print("=" * 60)
    print(f"分类器训练完成！")
    print(f"准确率: {accuracy:.4f}")
    print(f"模型保存路径: {save_path}")
    print("=" * 60)

    return save_path


def test_classifier():
    """测试分类器"""
    print("=" * 60)
    print("测试问题分类器")
    print("=" * 60)

    # 检查是否有训练好的分类器
    classifier_path = "./trained_classifiers/question_classifier"

    if not os.path.exists(classifier_path):
        print("未找到训练好的分类器，请先训练分类器")
        return

    # 创建分类器
    classifier = QuestionClassifier(classifier_path)

    # 测试问题
    test_questions = [
        # 数据平台相关
        "数据清洗流程是什么？",
        "如何进行数据入库？",
        "数据质量检查方法",
        "数据监控怎么做？",
        "数据安全措施",

        # 通用对话
        "你好",
        "谢谢",
        "再见",
        "不客气",
        "好的",

        # 无关问题
        "今天天气怎么样？",
        "北京有什么好吃的？",
        "如何学习英语？",
        "what is your name?",
        "asdfghjkl"
    ]

    print("测试结果：")
    print("-" * 60)

    for question in test_questions:
        result = classifier.classify_question(question)
        category_cn = {
            "data_platform": "数据平台相关",
            "general_chat": "通用对话",
            "irrelevant": "无关问题"
        }

        print(f"问题: {question}")
        print(f"分类: {category_cn[result['category']]} (置信度: {result['confidence']:.3f})")
        print(f"方法: {result['method']}")
        print("-" * 40)


def test_intelligent_system():
    """测试智能问答系统"""
    print("=" * 60)
    print("测试智能问答系统")
    print("=" * 60)

    # 检查模型路径
    classifier_path = "./trained_classifiers/question_classifier"
    qa_model_path = "./trained_models/final_model"

    if not os.path.exists(classifier_path):
        classifier_path = None
        print("未找到分类器模型，将使用规则分类")

    if not os.path.exists(qa_model_path):
        qa_model_path = None
        print("未找到问答模型，将使用基础模型")

    # 创建智能问答系统
    qa_system = IntelligentQASystem(
        classifier_model_path=classifier_path,
        qa_model_path=qa_model_path
    )

    # 测试问题
    test_questions = [
        "你好",  # 通用对话
        "数据清洗流程是什么？",  # 数据平台相关
        "今天天气怎么样？",  # 无关问题
        "谢谢",  # 通用对话
        "如何进行数据入库？",  # 数据平台相关
        "random english text",  # 无关问题
    ]

    print("测试结果：")
    print("-" * 60)

    for question in test_questions:
        result = qa_system.answer_question(question)

        print(f"问题: {question}")
        print(f"回答: {result['answer']}")
        print(f"路由: {result['system_route']}")
        print(f"分类: {result['classification']['category']} (置信度: {result['classification']['confidence']:.3f})")
        print("-" * 60)


def interactive_classifier_test():
    """交互式分类器测试"""
    print("=" * 60)
    print("交互式分类器测试")
    print("=" * 60)

    classifier_path = "./trained_classifiers/question_classifier"

    if not os.path.exists(classifier_path):
        print("未找到训练好的分类器，请先训练分类器")
        return

    classifier = QuestionClassifier(classifier_path)

    print("输入问题进行分类测试，输入 'quit' 退出")
    print("-" * 60)

    while True:
        try:
            question = input("请输入问题: ").strip()

            if question.lower() in ['quit', 'exit', '退出', 'q']:
                break

            if not question:
                continue

            result = classifier.classify_question(question)

            category_cn = {
                "data_platform": "数据平台相关",
                "general_chat": "通用对话",
                "irrelevant": "无关问题"
            }

            print(f"分类结果: {category_cn[result['category']]}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"分类方法: {result['method']}")
            print("-" * 40)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"处理错误: {e}")


def main():
    """主菜单"""
    while True:
        print("\n" + "=" * 60)
        print("🤖 问题分类器训练和测试系统")
        print("=" * 60)
        print("1. 训练问题分类器")
        print("2. 测试分类器效果")
        print("3. 测试智能问答系统")
        print("4. 交互式分类器测试")
        print("5. 启动智能问答系统")
        print("0. 退出")
        print("=" * 60)

        try:
            choice = input("请选择功能 (0-5): ").strip()

            if choice == "1":
                train_classifier()
            elif choice == "2":
                test_classifier()
            elif choice == "3":
                test_intelligent_system()
            elif choice == "4":
                interactive_classifier_test()
            elif choice == "5":
                # 启动智能问答系统
                classifier_path = "./trained_classifiers/question_classifier"
                qa_model_path = "./trained_models/final_model"

                if not os.path.exists(classifier_path):
                    classifier_path = None

                if not os.path.exists(qa_model_path):
                    qa_model_path = None

                qa_system = IntelligentQASystem(
                    classifier_model_path=classifier_path,
                    qa_model_path=qa_model_path
                )
                qa_system.interactive_mode()
            elif choice == "0":
                print("感谢使用！再见！")
                break
            else:
                print("无效选择，请重新输入")

        except KeyboardInterrupt:
            print("\n感谢使用！再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
