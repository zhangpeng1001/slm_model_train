"""
统一PEFT模型使用示例
展示如何使用训练好的统一PEFT模型进行各种任务的推理
"""
import os
import sys
import torch
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_peft_trainer import UnifiedPeftTrainer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedPeftInference:
    """统一PEFT模型推理类"""
    
    def __init__(self, model_path, base_model_path):
        """
        初始化推理器
        
        Args:
            model_path: 训练好的PEFT模型路径
            base_model_path: 基础BERT模型路径
        """
        self.trainer = UnifiedPeftTrainer(base_model_path)
        
        # 加载训练好的模型
        if os.path.exists(model_path):
            self.trainer.load_model(model_path)
            logger.info(f"已加载训练好的模型: {model_path}")
        else:
            logger.warning(f"模型路径不存在: {model_path}")
            logger.info("将使用未训练的模型进行演示")
    
    def classify_question(self, question):
        """
        问题分类：判断问题属于数据平台相关、通用对话还是无关问题
        
        Args:
            question: 输入问题
            
        Returns:
            dict: 包含分类结果的字典
        """
        result = self.trainer.predict(question, 'question_classifier')
        
        return {
            'question': question,
            'category': result['predicted_class'],
            'confidence': result['confidence'],
            'task': '问题分类'
        }
    
    def classify_question_type(self, question):
        """
        问题类型分类：判断问题是询问信息还是请求处理任务
        
        Args:
            question: 输入问题
            
        Returns:
            dict: 包含分类结果的字典
        """
        result = self.trainer.predict(question, 'question_type_classifier')
        
        return {
            'question': question,
            'intent_type': result['predicted_class'],
            'confidence': result['confidence'],
            'task': '问题类型分类'
        }
    
    def extract_data_info(self, question):
        """
        数据提取：从问题中提取数据类型信息
        
        Args:
            question: 输入问题
            
        Returns:
            dict: 包含提取结果的字典
        """
        result = self.trainer.predict(question, 'data_extraction')
        
        return {
            'question': question,
            'extracted_entities': result['entities'],
            'task': '数据提取'
        }
    
    def answer_question(self, question):
        """
        问答系统：回答数据平台相关问题
        
        Args:
            question: 输入问题
            
        Returns:
            dict: 包含问答结果的字典
        """
        result = self.trainer.predict(question, 'qa_system')
        
        return {
            'question': question,
            'qa_output_shape': result['qa_output'].shape,
            'task': '问答系统'
        }
    
    def comprehensive_analysis(self, question):
        """
        综合分析：对一个问题进行全面的分析
        
        Args:
            question: 输入问题
            
        Returns:
            dict: 包含所有任务结果的字典
        """
        results = {
            'question': question,
            'analysis': {}
        }
        
        # 1. 问题分类
        classification_result = self.classify_question(question)
        results['analysis']['question_classification'] = {
            'category': classification_result['category'],
            'confidence': classification_result['confidence']
        }
        
        # 2. 问题类型分类
        type_result = self.classify_question_type(question)
        results['analysis']['question_type'] = {
            'intent_type': type_result['intent_type'],
            'confidence': type_result['confidence']
        }
        
        # 3. 数据提取
        extraction_result = self.extract_data_info(question)
        results['analysis']['data_extraction'] = {
            'entities': extraction_result['extracted_entities']
        }
        
        # 4. 问答系统（仅在数据平台相关问题时使用）
        if classification_result['category'] == 'data_platform':
            qa_result = self.answer_question(question)
            results['analysis']['qa_system'] = {
                'output_shape': qa_result['qa_output_shape']
            }
        
        return results


def demo_usage():
    """演示统一PEFT模型的使用"""
    
    # 模型路径配置
    base_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    trained_model_path = r"E:\project\llm\lora\peft\final_unified_peft_model"
    
    # 创建推理器
    inference = UnifiedPeftInference(trained_model_path, base_model_path)
    
    # 测试问题集
    test_questions = [
        # 数据平台相关问题
        "数据清洗流程是什么？",
        "我有一批西安市地类图斑的数据，怎么进行发服务",
        "请帮我把实景三维模型成果数据进行治理",
        "DEM高程数据怎么进行质量检查",
        "如何进行数据入库？",
        
        # 通用对话
        "你好，请问可以帮忙吗？",
        "谢谢你的帮助",
        "再见",
        
        # 无关问题
        "今天天气怎么样？",
        "北京有什么好吃的？",
        "如何学习英语？"
    ]
    
    logger.info("=== 统一PEFT模型使用演示 ===\n")
    
    # 1. 单任务演示
    logger.info("--- 单任务演示 ---")
    
    for i, question in enumerate(test_questions[:3], 1):
        logger.info(f"\n{i}. 测试问题: {question}")
        
        # 问题分类
        classification = inference.classify_question(question)
        logger.info(f"   问题分类: {classification['category']} (置信度: {classification['confidence']:.4f})")
        
        # 问题类型分类
        type_classification = inference.classify_question_type(question)
        logger.info(f"   意图类型: {type_classification['intent_type']} (置信度: {type_classification['confidence']:.4f})")
        
        # 数据提取
        extraction = inference.extract_data_info(question)
        logger.info(f"   提取实体: {extraction['extracted_entities']}")
    
    # 2. 综合分析演示
    logger.info("\n--- 综合分析演示 ---")
    
    comprehensive_test_questions = [
        "我有一批西安市地类图斑的数据，怎么进行发服务",
        "你好，请问可以帮忙吗？",
        "今天天气怎么样？"
    ]
    
    for i, question in enumerate(comprehensive_test_questions, 1):
        logger.info(f"\n{i}. 综合分析问题: {question}")
        
        analysis = inference.comprehensive_analysis(question)
        
        logger.info(f"   问题分类: {analysis['analysis']['question_classification']['category']} "
                   f"(置信度: {analysis['analysis']['question_classification']['confidence']:.4f})")
        
        logger.info(f"   意图类型: {analysis['analysis']['question_type']['intent_type']} "
                   f"(置信度: {analysis['analysis']['question_type']['confidence']:.4f})")
        
        logger.info(f"   提取实体: {analysis['analysis']['data_extraction']['entities']}")
        
        if 'qa_system' in analysis['analysis']:
            logger.info(f"   问答输出: {analysis['analysis']['qa_system']['output_shape']}")
    
    # 3. 批量处理演示
    logger.info("\n--- 批量处理演示 ---")
    
    batch_questions = [
        "数据清洗流程是什么？",
        "请帮我处理遥感影像数据",
        "谢谢你的帮助",
        "如何学习Python？"
    ]
    
    logger.info("批量问题分类结果:")
    for question in batch_questions:
        result = inference.classify_question(question)
        logger.info(f"  '{question}' -> {result['category']} ({result['confidence']:.3f})")
    
    logger.info("\n批量数据提取结果:")
    for question in batch_questions:
        result = inference.extract_data_info(question)
        if result['extracted_entities']:
            logger.info(f"  '{question}' -> {result['extracted_entities']}")


def interactive_demo():
    """交互式演示"""
    
    # 模型路径配置
    base_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    trained_model_path = r"E:\project\llm\lora\peft\final_unified_peft_model"
    
    # 创建推理器
    inference = UnifiedPeftInference(trained_model_path, base_model_path)
    
    logger.info("=== 交互式PEFT模型演示 ===")
    logger.info("输入问题进行测试，输入 'quit' 退出")
    logger.info("可用任务: 1-问题分类, 2-问题类型分类, 3-数据提取, 4-问答系统, 5-综合分析")
    
    while True:
        try:
            question = input("\n请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                logger.info("退出演示")
                break
            
            if not question:
                continue
            
            task_choice = input("选择任务 (1-5, 默认5综合分析): ").strip()
            if not task_choice:
                task_choice = '5'
            
            if task_choice == '1':
                result = inference.classify_question(question)
                logger.info(f"问题分类: {result['category']} (置信度: {result['confidence']:.4f})")
            
            elif task_choice == '2':
                result = inference.classify_question_type(question)
                logger.info(f"意图类型: {result['intent_type']} (置信度: {result['confidence']:.4f})")
            
            elif task_choice == '3':
                result = inference.extract_data_info(question)
                logger.info(f"提取实体: {result['extracted_entities']}")
            
            elif task_choice == '4':
                result = inference.answer_question(question)
                logger.info(f"问答输出维度: {result['qa_output_shape']}")
            
            elif task_choice == '5':
                analysis = inference.comprehensive_analysis(question)
                logger.info("=== 综合分析结果 ===")
                logger.info(f"问题分类: {analysis['analysis']['question_classification']['category']} "
                           f"(置信度: {analysis['analysis']['question_classification']['confidence']:.4f})")
                logger.info(f"意图类型: {analysis['analysis']['question_type']['intent_type']} "
                           f"(置信度: {analysis['analysis']['question_type']['confidence']:.4f})")
                logger.info(f"提取实体: {analysis['analysis']['data_extraction']['entities']}")
                if 'qa_system' in analysis['analysis']:
                    logger.info(f"问答输出: {analysis['analysis']['qa_system']['output_shape']}")
            
            else:
                logger.info("无效的任务选择")
        
        except KeyboardInterrupt:
            logger.info("\n用户中断，退出演示")
            break
        except Exception as e:
            logger.error(f"处理过程中出现错误: {e}")


def performance_test():
    """性能测试"""
    import time
    
    # 模型路径配置
    base_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    trained_model_path = r"E:\project\llm\lora\peft\final_unified_peft_model"
    
    # 创建推理器
    inference = UnifiedPeftInference(trained_model_path, base_model_path)
    
    # 测试问题
    test_questions = [
        "数据清洗流程是什么？",
        "我有一批西安市地类图斑的数据，怎么进行发服务",
        "你好，请问可以帮忙吗？",
        "今天天气怎么样？"
    ] * 10  # 重复10次以测试性能
    
    logger.info("=== 性能测试 ===")
    logger.info(f"测试问题数量: {len(test_questions)}")
    
    # 测试各个任务的性能
    tasks = [
        ('问题分类', 'question_classifier'),
        ('问题类型分类', 'question_type_classifier'),
        ('数据提取', 'data_extraction'),
        ('问答系统', 'qa_system')
    ]
    
    for task_name, task_type in tasks:
        start_time = time.time()
        
        for question in test_questions:
            inference.trainer.predict(question, task_type)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(test_questions)
        
        logger.info(f"{task_name}: 总时间 {total_time:.2f}s, 平均时间 {avg_time:.4f}s/问题, "
                   f"吞吐量 {len(test_questions)/total_time:.2f} 问题/秒")


if __name__ == "__main__":
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="统一PEFT模型使用示例")
    parser.add_argument("--mode", choices=['demo', 'interactive', 'performance'], 
                       default='demo', help="运行模式")
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'demo':
            demo_usage()
        elif args.mode == 'interactive':
            interactive_demo()
        elif args.mode == 'performance':
            performance_test()
    except Exception as e:
        logger.error(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
