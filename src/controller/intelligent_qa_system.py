"""
智能问答系统控制器
集成问题分类器和问答模型，提供智能路由功能
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.question_classifier import QuestionClassifier
from src.data_platform.enhanced_qa_model import EnhancedDataPlatformQAModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentQASystem:
    """智能问答系统"""

    def __init__(self, classifier_model_path=None, qa_model_path=None):
        """
        初始化智能问答系统
        
        Args:
            classifier_model_path: 分类器模型路径
            qa_model_path: 问答模型路径
        """
        # 初始化问题分类器
        self.classifier = QuestionClassifier(classifier_model_path)

        # 初始化问答模型
        self.qa_model = EnhancedDataPlatformQAModel(trained_model_path=qa_model_path)

        # 通用回复模板
        self.general_responses = {
            "greeting": [
                "您好！我是数据平台智能助手，很高兴为您服务！",
                "您好！有什么数据平台相关的问题需要帮助吗？",
                "您好！我可以帮您解答数据清洗、数据入库、数据监控等相关问题。"
            ],
            "thanks": [
                "不客气！很高兴能帮到您。",
                "您太客气了！还有其他问题吗？",
                "不用谢！如果还有数据平台相关的问题，随时问我。"
            ],
            "goodbye": [
                "再见！祝您工作顺利！",
                "再见！有问题随时找我。",
                "再见！期待下次为您服务。"
            ],
            "default": [
                "好的，我明白了。还有什么数据平台相关的问题吗？",
                "收到！如果有数据处理方面的问题，我很乐意帮助您。",
                "明白！我是专门处理数据平台问题的助手。"
            ]
        }

        logger.info("智能问答系统初始化完成")

    def answer_question(self, question):
        """
        智能回答问题
        
        Args:
            question: 用户问题
            
        Returns:
            dict: 包含答案、分类信息等的完整结果
        """
        # 第一步：问题分类
        classification = self.classifier.classify_question(question)

        # 第二步：根据分类结果路由到不同处理器
        if classification["category"] == "data_platform":
            # 数据平台相关问题 -> 专业问答模型
            qa_result = self.qa_model.answer_question(question)

            return {
                "answer": qa_result["answer"],
                "classification": classification,
                "qa_method": qa_result["method"],
                "qa_confidence": qa_result["confidence"],
                "qa_source": qa_result["source"],
                "system_route": "专业问答模型",
                "matched_question": qa_result.get("matched_question", None)
            }

        elif classification["category"] == "general_chat":
            # 通用对话 -> 友好回复
            response = self._generate_general_response(question)

            return {
                "answer": response,
                "classification": classification,
                "qa_method": "template_response",
                "qa_confidence": 1.0,
                "qa_source": "general_chat",
                "system_route": "通用对话处理"
            }

        else:
            # 无关问题 -> 礼貌拒绝
            response = self._generate_irrelevant_response()

            return {
                "answer": response,
                "classification": classification,
                "qa_method": "polite_decline",
                "qa_confidence": 1.0,
                "qa_source": "system_response",
                "system_route": "无关问题处理"
            }

    def _generate_general_response(self, question):
        """生成通用对话回复"""
        question_lower = question.lower().strip()

        # 问候语
        if any(word in question_lower for word in ["你好", "您好", "hi", "hello", "早上好", "下午好", "晚上好"]):
            import random
            return random.choice(self.general_responses["greeting"])

        # 感谢语
        elif any(word in question_lower for word in ["谢谢", "感谢"]):
            import random
            return random.choice(self.general_responses["thanks"])

        # 告别语
        elif any(word in question_lower for word in ["再见", "拜拜", "bye"]):
            import random
            return random.choice(self.general_responses["goodbye"])

        # 默认回复
        else:
            import random
            return random.choice(self.general_responses["default"])

    def _generate_irrelevant_response(self):
        """生成无关问题的礼貌拒绝回复"""
        responses = [
            "抱歉，我是数据平台专业助手，主要回答数据清洗、数据入库、数据监控、数据安全等相关问题。您的问题似乎不在我的专业范围内。",
            "很抱歉，我专注于数据平台相关的技术问题。如果您有数据处理、数据质量、数据安全等方面的问题，我很乐意帮助您。",
            "不好意思，我是专门处理数据平台业务的智能助手。请问您是否有数据清洗、数据入库或数据监控方面的问题需要咨询？",
            "抱歉，您的问题超出了我的专业领域。我主要协助解决数据平台相关的技术问题，比如数据处理流程、数据质量管理等。"
        ]

        import random
        return random.choice(responses)

    def get_system_info(self):
        """获取系统信息"""
        classifier_info = self.classifier.get_classification_info()
        qa_model_info = self.qa_model.get_model_info()

        return {
            "classifier": classifier_info,
            "qa_model": qa_model_info,
            "system_status": "运行正常",
            "supported_categories": ["数据平台相关", "通用对话", "无关问题"]
        }

    def batch_answer(self, questions):
        """批量回答问题"""
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append({
                "question": question,
                "result": result
            })
        return results

    def interactive_mode(self):
        """交互式问答模式"""
        print("=" * 60)
        print("🤖 智能数据平台问答系统")
        print("=" * 60)
        print("功能说明：")
        print("• 数据平台相关问题 -> 专业问答")
        print("• 通用对话 -> 友好回复")
        print("• 无关问题 -> 礼貌拒绝")
        print("输入 'quit' 或 '退出' 结束对话")
        print("=" * 60)

        while True:
            try:
                question = input("\n👤 您: ").strip()

                if question.lower() in ['quit', 'exit', '退出', 'q']:
                    print("🤖 助手: 再见！祝您工作顺利！")
                    break

                if not question:
                    continue

                # 获取回答
                result = self.answer_question(question)

                # 显示回答
                print(f"🤖 助手: {result['answer']}")

                # 显示详细信息（可选）
                if result['classification']['method'] != 'rule_based':
                    print(
                        f"   [分类: {result['classification']['category']}, 置信度: {result['classification']['confidence']:.2f}]")

            except KeyboardInterrupt:
                print("\n🤖 助手: 再见！祝您工作顺利！")
                break
            except Exception as e:
                print(f"🤖 助手: 抱歉，处理您的问题时出现了错误: {e}")


def main():
    """主函数 - 演示智能问答系统"""
    print("初始化智能问答系统...")

    # 可以指定训练好的分类器和问答模型路径
    classifier_path = "./trained_classifiers/question_classifier"  # 如果存在的话
    qa_model_path = "./trained_models/final_model"  # 如果存在的话

    # 检查模型是否存在
    if not os.path.exists(classifier_path):
        classifier_path = None
        print("未找到训练好的分类器，将使用规则分类器")

    if not os.path.exists(qa_model_path):
        qa_model_path = None
        print("未找到训练好的问答模型，将使用基础模型")

    # 初始化系统
    qa_system = IntelligentQASystem(
        classifier_model_path=classifier_path,
        qa_model_path=qa_model_path
    )

    # 显示系统信息
    system_info = qa_system.get_system_info()
    print(f"分类器状态: {'已加载训练模型' if system_info['classifier']['is_model_loaded'] else '使用规则分类'}")
    print(f"问答模型状态: {'已加载训练模型' if system_info['qa_model']['is_trained_model'] else '使用基础模型'}")

    # 进入交互模式
    qa_system.interactive_mode()


if __name__ == "__main__":
    main()
