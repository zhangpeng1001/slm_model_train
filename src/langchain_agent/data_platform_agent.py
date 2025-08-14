"""
数据平台LangChain Agent
集成训练模型和工具，实现复杂任务的自动化处理
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import AgentType, initialize_agent
from langchain.llms.base import LLM
from langchain.schema import BaseMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any, Dict
import json
import logging

from src.langchain_agent.data_tools import DATA_TOOLS, get_tool_descriptions
from src.controller.question_classifier import QuestionClassifier
from src.data_platform.enhanced_qa_model import EnhancedDataPlatformQAModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataPlatformLLM(LLM):
    """
    自定义LLM，集成我们训练的模型
    """
    
    classifier: Any = None
    qa_model: Any = None
    temperature: float = 0.1
    max_tokens: int = 512
    
    def __init__(self, classifier_path=None, qa_model_path=None):
        super().__init__()
        
        # 初始化我们的训练模型
        object.__setattr__(self, 'classifier', QuestionClassifier(classifier_path))
        object.__setattr__(self, 'qa_model', EnhancedDataPlatformQAModel(trained_model_path=qa_model_path))
        
        # 模型配置
        object.__setattr__(self, 'temperature', 0.1)
        object.__setattr__(self, 'max_tokens', 512)

    @property
    def _llm_type(self) -> str:
        return "custom_data_platform"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """调用我们的自定义LLM"""
        try:
            # 使用我们的分类器判断问题类型
            classification = self.classifier.classify_question(prompt)

            # 如果是数据平台相关问题，使用我们的专业模型
            if classification["category"] == "data_platform":
                qa_result = self.qa_model.answer_question(prompt)

                # 如果是工具调用相关的prompt，需要特殊处理
                if "Action:" in prompt or "tool" in prompt.lower():
                    return self._handle_tool_prompt(prompt, qa_result)
                else:
                    return qa_result["answer"]

            # 其他情况使用规则生成
            return self._generate_response(prompt, classification)

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"

    def _handle_tool_prompt(self, prompt: str, qa_result: Dict) -> str:
        """处理工具调用相关的prompt"""
        # 分析prompt中的工具调用需求
        if "数据查询" in prompt or "查询" in prompt:
            return "Action: data_query\nAction Input: 用户指定的数据|schema"
        elif "数据采集" in prompt or "采集" in prompt:
            return "Action: data_collection\nAction Input: 用户指定的数据|database"
        elif "数据入库" in prompt or "入库" in prompt:
            return "Action: data_storage\nAction Input: 用户指定的数据|采集的数据"
        elif "发服务" in prompt or "服务" in prompt:
            return "Action: data_service\nAction Input: 用户指定的数据|rest_api"
        else:
            return qa_result["answer"]

    def _generate_response(self, prompt: str, classification: Dict) -> str:
        """生成响应"""
        if classification["category"] == "general_chat":
            return "您好！我是数据平台智能助手，可以帮您处理数据相关的复杂任务。"
        elif classification["category"] == "irrelevant":
            return "抱歉，我专注于数据平台相关的任务处理。请告诉我您需要处理什么数据操作。"
        else:
            return "我理解您的需求，让我为您分析并执行相应的数据操作。"


class DataPlatformAgent:
    """数据平台智能Agent"""

    def __init__(self, classifier_path=None, qa_model_path=None):
        """
        初始化Agent
        
        Args:
            classifier_path: 分类器模型路径
            qa_model_path: 问答模型路径
        """
        # 初始化自定义LLM
        self.llm = CustomDataPlatformLLM(classifier_path, qa_model_path)

        # 初始化Agent
        self.agent = initialize_agent(
            tools=DATA_TOOLS,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )

        # 任务模板
        self.task_templates = {
            "data_service_pipeline": """
            用户想要对{data_name}数据进行发服务，需要执行以下步骤：
            1. 首先查询{data_name}数据的基本信息
            2. 采集{data_name}数据
            3. 将采集的数据进行入库
            4. 最后发布数据服务
            
            请按顺序执行这些操作。
            """,

            "data_analysis_pipeline": """
            用户想要分析{data_name}数据，需要执行以下步骤：
            1. 查询{data_name}数据的结构和样本
            2. 采集完整的{data_name}数据
            3. 对数据进行质量检查和入库
            
            请按顺序执行这些操作。
            """,

            "data_migration_pipeline": """
            用户想要迁移{data_name}数据，需要执行以下步骤：
            1. 查询源{data_name}数据信息
            2. 采集{data_name}数据
            3. 将数据存储到新的位置
            4. 验证迁移结果
            
            请按顺序执行这些操作。
            """
        }

        logger.info("数据平台Agent初始化完成")

    def process_complex_request(self, user_request: str) -> Dict[str, Any]:
        """
        处理复杂的用户请求
        
        Args:
            user_request: 用户请求
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"开始处理复杂请求: {user_request}")

            # 分析用户意图
            intent_analysis = self._analyze_user_intent(user_request)

            # 根据意图选择任务模板
            task_prompt = self._generate_task_prompt(user_request, intent_analysis)

            # 执行任务
            result = self.agent.run(task_prompt)

            return {
                "user_request": user_request,
                "intent_analysis": intent_analysis,
                "task_prompt": task_prompt,
                "execution_result": result,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"处理复杂请求失败: {e}")
            return {
                "user_request": user_request,
                "error": str(e),
                "status": "failed"
            }

    def _analyze_user_intent(self, user_request: str) -> Dict[str, Any]:
        """分析用户意图"""
        intent = {
            "data_name": None,
            "operation_type": None,
            "pipeline_type": None,
            "keywords": []
        }

        # 提取数据名称
        if "数据" in user_request:
            # 简单的数据名称提取逻辑
            words = user_request.split()
            for i, word in enumerate(words):
                if "数据" in word and i > 0:
                    intent["data_name"] = words[i - 1] + "数据"
                    break

            if not intent["data_name"]:
                intent["data_name"] = "用户指定数据"

        # 分析操作类型
        if "发服务" in user_request or "服务" in user_request:
            intent["operation_type"] = "service_publish"
            intent["pipeline_type"] = "data_service_pipeline"
        elif "分析" in user_request:
            intent["operation_type"] = "data_analysis"
            intent["pipeline_type"] = "data_analysis_pipeline"
        elif "迁移" in user_request:
            intent["operation_type"] = "data_migration"
            intent["pipeline_type"] = "data_migration_pipeline"
        else:
            intent["operation_type"] = "general_processing"
            intent["pipeline_type"] = "data_service_pipeline"  # 默认使用服务发布流程

        # 提取关键词
        keywords = ["查询", "采集", "入库", "服务", "分析", "迁移", "处理"]
        for keyword in keywords:
            if keyword in user_request:
                intent["keywords"].append(keyword)

        return intent

    def _generate_task_prompt(self, user_request: str, intent_analysis: Dict) -> str:
        """生成任务执行prompt"""
        data_name = intent_analysis.get("data_name", "用户数据")
        pipeline_type = intent_analysis.get("pipeline_type", "data_service_pipeline")

        # 获取对应的任务模板
        template = self.task_templates.get(pipeline_type, self.task_templates["data_service_pipeline"])

        # 填充模板
        task_prompt = template.format(data_name=data_name)

        # 添加用户原始请求
        full_prompt = f"""
        用户请求: {user_request}
        
        分析结果: 用户想要对{data_name}进行{intent_analysis.get("operation_type", "处理")}操作
        
        执行计划:
        {task_prompt}
        
        请使用可用的工具按顺序执行上述操作，并在每个步骤完成后提供详细的结果说明。
        """

        return full_prompt

    def simple_execute(self, task_description: str) -> str:
        """简单执行单个任务"""
        try:
            result = self.agent.run(task_description)
            return result
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            return f"任务执行失败: {str(e)}"

    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return [tool.name for tool in DATA_TOOLS]

    def get_tool_descriptions(self) -> str:
        """获取工具描述"""
        return get_tool_descriptions()


def main():
    """演示Agent功能"""
    print("=" * 60)
    print("🤖 数据平台LangChain Agent演示")
    print("=" * 60)

    # 检查模型路径
    classifier_path = r"/src/train_data/trained_classifiers\question_classifier"
    qa_model_path = r"/src/train_data/trained_models\final_model"

    if not os.path.exists(classifier_path):
        classifier_path = None
        print("未找到分类器模型，将使用规则分类")

    if not os.path.exists(qa_model_path):
        qa_model_path = None
        print("未找到问答模型，将使用基础模型")

    # 初始化Agent
    agent = DataPlatformAgent(
        classifier_path=classifier_path,
        qa_model_path=qa_model_path
    )

    print("Agent初始化完成！")
    print(f"可用工具: {', '.join(agent.get_available_tools())}")
    print("\n" + "=" * 60)

    # 测试复杂任务
    test_requests = [
        "我想对用户行为数据进行发服务",
        "需要分析销售数据的质量",
        "帮我迁移订单数据到新系统"
    ]

    for request in test_requests:
        print(f"\n📝 测试请求: {request}")
        print("-" * 40)

        result = agent.process_complex_request(request)

        if result["status"] == "success":
            print(f"✅ 执行成功")
            print(f"意图分析: {result['intent_analysis']}")
            print(f"执行结果: {result['execution_result']}")
        else:
            print(f"❌ 执行失败: {result['error']}")

        print("-" * 40)

    # 交互模式
    print("\n🎯 进入交互模式 (输入 'quit' 退出)")
    while True:
        try:
            user_input = input("\n👤 请描述您的数据处理需求: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                break

            if not user_input:
                continue

            print("🤖 正在处理您的请求...")
            result = agent.process_complex_request(user_input)

            if result["status"] == "success":
                print(f"✅ 任务执行完成！")
                print(f"执行结果: {result['execution_result']}")
            else:
                print(f"❌ 任务执行失败: {result['error']}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 处理错误: {e}")

    print("\n👋 感谢使用数据平台Agent！")


if __name__ == "__main__":
    main()
