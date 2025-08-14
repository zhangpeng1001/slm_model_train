"""
数据提取训练器的训练和测试脚本
"""
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_extraction_trainer import DataExtractionTrainer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_data_extractor():
    """训练数据提取模型"""
    # 模型路径
    model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = DataExtractionTrainer(model_path)

    # 准备训练数据
    texts, labels = trainer.prepare_training_data()

    # 开始训练
    trainer.train(
        epochs=6,
        batch_size=2,
        learning_rate=1e-5
    )

    # 评估模型
    trainer.evaluate(texts, labels)

    # 保存最终模型
    model_save_path = trainer.save_model('final_data_extractor')

    return model_save_path


def test_data_extractor(model_path=None):
    """测试数据提取模型"""
    # 模型路径
    bert_model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = DataExtractionTrainer(bert_model_path)

    # 如果提供了训练好的模型路径，则加载
    if model_path and os.path.exists(model_path):
        trainer.load_model(model_path)
        logger.info(f"已加载训练好的模型: {model_path}")
    else:
        logger.info("使用未训练的模型进行测试")

    # 测试样例
    test_questions = [
        "我有一批西安市地类图斑的数据，怎么进行发服务",
        "请帮我把实景三维模型成果数据进行治理",
        "我已经上传了单波段浮点投影的数据，现在想进行入库",
        "有一些遥感影像数据需要处理",
        "DEM高程数据怎么进行质量检查",
        "矢量地形图数据如何入库",
        "我的点云数据处理完了，如何发布服务",
        "正射影像数据需要进行坐标转换",
        "地理信息数据库的数据怎么备份",
        "卫星遥感数据的预处理流程是什么"
    ]

    logger.info("\n=== 数据提取测试结果 ===")

    for i, question in enumerate(test_questions, 1):
        result = trainer.extract_data_from_question(question)

        logger.info(f"\n测试 {i}:")
        logger.info(f"问题: {question}")
        logger.info(f"提取的主要数据类型: {result['main_data_type']}")
        logger.info(f"所有提取的实体: {result['all_entities']}")
        logger.info("-" * 60)


def interactive_test():
    """交互式测试"""
    # 模型路径
    bert_model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 创建训练器
    trainer = DataExtractionTrainer(bert_model_path)

    # 尝试加载训练好的模型
    trained_model_path = r"/src/train/trained_data_extractors/final_data_extractor"
    if os.path.exists(trained_model_path):
        trainer.load_model(trained_model_path)
        logger.info("已加载训练好的模型")
    else:
        logger.info("未找到训练好的模型，使用未训练的模型")

    logger.info("\n=== 交互式数据提取测试 ===")
    logger.info("请输入问题，程序将提取其中的数据类型")
    logger.info("输入 'quit' 或 'exit' 退出")

    while True:
        try:
            question = input("\n请输入问题: ").strip()

            if question.lower() in ['quit', 'exit', '退出']:
                logger.info("退出交互式测试")
                break

            if not question:
                logger.info("请输入有效的问题")
                continue

            result = trainer.extract_data_from_question(question)

            print(f"\n提取结果:")
            print(f"主要数据类型: {result['main_data_type']}")
            print(f"所有实体: {result['all_entities']}")

        except KeyboardInterrupt:
            logger.info("\n用户中断，退出程序")
            break
        except Exception as e:
            logger.error(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    # logger.info("开始训练数据提取模型...")
    # model_path = train_data_extractor()
    # logger.info(f"训练完成，模型保存在: {model_path}")

    # logger.info("开始测试数据提取模型...")
    # test_data_extractor(r"E:\project\python\slm_model_train\src\train\trained_data_extractors\final_data_extractor")

    logger.info("交互式测试...")
    interactive_test()
