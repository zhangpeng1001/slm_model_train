"""
问题类型分类器训练和测试脚本
用于训练和测试问题类型分类模型（问题回答 vs 任务处理）
"""
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.trainer.question_type_classifier.question_type_classifier_trainer import QuestionTypeClassifierTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_question_type_classifier(model_path=None):
    """测试问题类型分类器"""
    logger.info("=== 测试问题类型分类器 ===")

    if model_path is None:
        # 使用默认的训练好的模型路径
        model_path = r"E:\project\python\slm_model_train\src\train_data\trained_question_type_classifier\question_type_classifier"

    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        logger.info("请先训练模型或检查模型路径")
        return

    try:
        # 创建训练器并加载模型
        bert_model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
        trainer = QuestionTypeClassifierTrainer(bert_model_path)
        trainer.load_model(model_path)

        # 测试样本
        test_samples = [
            # 问题回答类型样本
            "我有一批西安市地类图斑的数据，怎么进行发服务",
            "DEM高程数据怎么进行质量检查",
            "矢量地形图数据如何入库",
            "我的点云数据处理完了，如何发布服务",
            "地理信息数据库的数据怎么备份",
            "卫星遥感数据的预处理流程是什么",
            "无人机航拍数据如何进行几何纠正",
            "土地利用现状数据怎么更新",
            "基础地理信息数据如何管理",

            # 任务处理类型样本
            "请帮我把实景三维模型成果数据进行治理",
            "我已经上传了单波段浮点投影的数据，现在想进行入库",
            "有一些遥感影像数据需要处理",
            "正射影像数据需要进行坐标转换",
            "激光雷达数据进行格式转换",
            "地籍调查数据需要标准化处理",
            "处理地下管线探测数据",
            "倾斜摄影测量数据进行建模"
        ]

        logger.info("开始预测测试...")

        # 单个预测测试
        for i, sample in enumerate(test_samples, 1):
            result = trainer.predict(sample)
            logger.info(f"\n--- 测试样本 {i} ---")
            logger.info(f"文本: {result['text']}")
            logger.info(f"预测类型: {result['predicted_type']}")
            logger.info(f"置信度: {result['confidence']:.4f}")
            logger.info(f"概率分布: 问题回答={result['probabilities']['问题回答']:.4f}, "
                        f"任务处理={result['probabilities']['任务处理']:.4f}")

        # 批量预测测试
        logger.info("\n=== 批量预测测试 ===")
        batch_results = trainer.batch_predict(test_samples[:5])

        for i, result in enumerate(batch_results, 1):
            logger.info(f"批量测试 {i}: {result['predicted_type']} (置信度: {result['confidence']:.4f})")

        logger.info("测试完成！")

    except Exception as e:
        # logger.error(f"测试过程中出现错误: {str(e)}")
        logger.error(f"测试过程中出现错误: {str(e)}", e)


def interactive_test():
    """交互式测试"""
    logger.info("=== 交互式测试 ===")

    # 模型路径
    model_path = r"E:\project\python\slm_model_train\src\train_data\trained_question_type_classifier\question_type_classifier"

    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        logger.info("请先训练模型")
        return

    try:
        # 加载模型
        bert_model_path = r"E:\project\python\slm_model_train\src\model_data\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
        trainer = QuestionTypeClassifierTrainer(bert_model_path)
        trainer.load_model(model_path)

        logger.info("模型加载完成！输入文本进行分类预测（输入 'quit' 退出）")

        while True:
            try:
                text = input("\n请输入要分类的文本: ").strip()

                if text.lower() == 'quit':
                    logger.info("退出交互式测试")
                    break

                if not text:
                    logger.info("请输入有效的文本")
                    continue

                # 预测
                result = trainer.predict(text)

                print(f"\n--- 预测结果 ---")
                print(f"输入文本: {result['text']}")
                print(f"预测类型: {result['predicted_type']}")
                print(f"置信度: {result['confidence']:.4f}")
                print(f"概率分布:")
                print(f"  问题回答: {result['probabilities']['问题回答']:.4f}")
                print(f"  任务处理: {result['probabilities']['任务处理']:.4f}")

            except KeyboardInterrupt:
                logger.info("用户中断，退出交互式测试")
                break
            except Exception as e:
                logger.error(f"预测过程中出现错误: {str(e)}")

    except Exception as e:
        logger.error(f"交互式测试初始化失败: {str(e)}")


if __name__ == "__main__":
    # 测试模式
    test_question_type_classifier()

    # 交互式测试模式
    # interactive_test()
