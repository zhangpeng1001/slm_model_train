"""
数据平台问答系统训练和测试脚本
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trainer.question_answer.qa_trainer import QATrainer
from src.data_platform.enhanced_qa_model import EnhancedDataPlatformQAModel


def train_model():
    """训练模型"""
    print("=" * 50)
    print("开始训练数据平台问答模型")
    print("=" * 50)

    # 模型路径
    model_path = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = QATrainer(model_path, save_dir='../trained_models')

    # 准备训练数据
    print("准备训练数据...")
    qa_pairs = trainer.prepare_training_data()

    # 开始训练
    print("开始训练...")
    trainer.train(
        qa_pairs=qa_pairs,
        epochs=5,  # 减少epoch数量以适应CPU训练
        batch_size=1,  # CPU训练使用更小的batch size
        learning_rate=5e-6  # 使用更小的学习率
    )

    # 评估模型
    print("评估模型...")
    trainer.evaluate(qa_pairs)

    # 保存最终模型
    trainer.save_model('final_model')

    print("训练完成！")
    return './trained_models/final_model'


def test_model(trained_model_path=None):
    """测试模型"""
    print("=" * 50)
    print("测试问答系统")
    print("=" * 50)

    # 创建增强版问答系统
    qa_system = EnhancedDataPlatformQAModel(
        trained_model_path=trained_model_path
    )

    # 显示模型信息
    model_info = qa_system.get_model_info()
    print("模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()

    # 测试问题
    test_questions = [
        "数据清洗流程是什么？",
        "如何进行数据入库？",
        "数据质量检查包括哪些内容？",
        "数据监控怎么做？",
        "数据安全措施有哪些？",
        "什么是数据预处理？",
        "数据备份策略",
        "任务调度系统的作用"
    ]

    print("开始测试问答...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        result = qa_system.answer_question(question)

        print(f"回答: {result['answer']}")
        print(f"方法: {result['method']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"来源: {result['source']}")

        if 'matched_question' in result:
            print(f"匹配问题: {result['matched_question']}")

        print("-" * 40)


def interactive_test(trained_model_path=None):
    """交互式测试"""
    print("=" * 50)
    print("交互式问答测试")
    print("=" * 50)

    # 创建增强版问答系统
    qa_system = EnhancedDataPlatformQAModel(
        trained_model_path=trained_model_path
    )

    print("增强版数据平台问答系统已启动！")
    print("输入 'quit' 退出，输入 'help' 查看可回答的主题")
    print("-" * 50)

    while True:
        try:
            question = input("\n请输入您的问题: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break

            if question.lower() in ['help', '帮助']:
                topics = qa_system.get_available_topics()
                print("\n可回答的主题:")
                for i, topic in enumerate(topics, 1):
                    print(f"  {i}. {topic}")
                continue

            # 回答问题
            result = qa_system.answer_question(question)

            print(f"\n回答: {result['answer']}")
            print(f"方法: {result['method']} | 置信度: {result['confidence']:.3f} | 来源: {result['source']}")

            if 'matched_question' in result:
                print(f"匹配问题: {result['matched_question']}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    """主函数"""
    print("数据平台问答系统训练和测试")
    print("1. 训练模型")
    print("2. 测试模型（使用基础BERT）")
    print("3. 测试模型（使用训练后的模型）")
    print("4. 交互式测试（基础BERT）")
    print("5. 交互式测试（训练后的模型）")

    choice = input("\n请选择操作 (1-5): ").strip()

    if choice == '1':
        # 训练模型
        trained_model_path = train_model()
        print(f"模型已保存到: {trained_model_path}")

        # 询问是否测试
        test_choice = input("\n是否测试训练后的模型？(y/n): ").strip().lower()
        if test_choice in ['y', 'yes', '是']:
            test_model(trained_model_path)

    elif choice == '2':
        # 测试基础模型
        test_model()

    elif choice == '3':
        # 测试训练后的模型
        model_path = input("请输入训练后的模型路径 (默认: ./trained_models/final_model): ").strip()
        if not model_path:
            model_path = '../../train_data/trained_question_answer/final_model'
        test_model(model_path)

    elif choice == '4':
        # 交互式测试基础模型
        interactive_test()

    elif choice == '5':
        # 交互式测试训练后的模型
        model_path = input("请输入训练后的模型路径 (默认: ./trained_models/final_model): ").strip()
        if not model_path:
            model_path = '../../train_data/trained_question_answer/final_model'
        interactive_test(model_path)

    else:
        print("无效选择")


if __name__ == "__main__":
    main()
