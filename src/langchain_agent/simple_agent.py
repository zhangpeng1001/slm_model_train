"""
简化版数据平台Agent
不依赖LangChain，直接集成训练模型和工具
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from typing import Dict, List, Any

from src.langchain_agent.data_tools import DATA_TOOLS, get_tool_by_name
from src.controller.question_classifier import QuestionClassifier
from src.data_platform.enhanced_qa_model import EnhancedDataPlatformQAModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDataPlatformAgent:
    """简化版数据平台Agent"""

    def __init__(self, classifier_path=None, qa_model_path=None):
        """
        初始化简化版Agent
        
        Args:
            classifier_path: 分类器模型路径
            qa_model_path: 问答模型路径
        """
        # 初始化模型
        self.classifier = QuestionClassifier(classifier_path)
        self.qa_model = EnhancedDataPlatformQAModel(trained_model_path=qa_model_path)

        # 工具映射
        self.tools = {tool.name: tool for tool in DATA_TOOLS}

        # 任务流程定义
        self.workflows = {
            "data_service": [
                {"tool": "data_query", "description": "查询数据基本信息"},
                {"tool": "data_collection", "description": "采集数据"},
                {"tool": "data_storage", "description": "数据入库"},
                {"tool": "data_service", "description": "发布数据服务"}
            ],
            "data_analysis": [
                {"tool": "data_query", "description": "查询数据结构"},
                {"tool": "data_collection", "description": "采集数据"},
                {"tool": "data_storage", "description": "数据质量检查和入库"}
            ],
            "data_migration": [
                {"tool": "data_query", "description": "查询源数据信息"},
                {"tool": "data_collection", "description": "采集数据"},
                {"tool": "data_storage", "description": "迁移数据到新位置"}
            ]
        }

        logger.info("简化版数据平台Agent初始化完成")

    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        处理用户请求
        
        Args:
            user_request: 用户请求
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"处理用户请求: {user_request}")

            # 1. 分析用户意图
            intent = self._analyze_intent(user_request)

            # 2. 选择工作流程
            workflow = self._select_workflow(intent)

            # 3. 执行工作流程
            execution_results = self._execute_workflow(workflow, intent)

            # 4. 生成总结报告
            summary = self._generate_summary(user_request, intent, execution_results)

            return {
                "user_request": user_request,
                "intent": intent,
                "workflow": workflow,
                "execution_results": execution_results,
                "summary": summary,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"处理请求失败: {e}")
            return {
                "user_request": user_request,
                "error": str(e),
                "status": "failed"
            }

    def _analyze_intent(self, user_request: str) -> Dict[str, Any]:
        """分析用户意图"""
        # 使用分类器判断请求类型
        classification = self.classifier.classify_question(user_request)

        intent = {
            "classification": classification,
            "data_name": self._extract_data_name(user_request),
            "operation": self._extract_operation(user_request),
            "workflow_type": self._determine_workflow_type(user_request)
        }

        return intent

    def _extract_data_name(self, user_request: str) -> str:
        """提取数据名称"""
        # 简单的数据名称提取
        if "数据" in user_request:
            words = user_request.replace("数据", " 数据 ").split()
            for i, word in enumerate(words):
                if word == "数据" and i > 0:
                    return words[i - 1] + "数据"

        return "用户指定数据"

    def _extract_operation(self, user_request: str) -> str:
        """提取操作类型"""
        if "发服务" in user_request or "服务" in user_request:
            return "service_publish"
        elif "分析" in user_request:
            return "data_analysis"
        elif "迁移" in user_request:
            return "data_migration"
        else:
            return "general_processing"

    def _determine_workflow_type(self, user_request: str) -> str:
        """确定工作流程类型"""
        if "发服务" in user_request or "服务" in user_request:
            return "data_service"
        elif "分析" in user_request:
            return "data_analysis"
        elif "迁移" in user_request:
            return "data_migration"
        else:
            return "data_service"  # 默认

    def _select_workflow(self, intent: Dict) -> List[Dict]:
        """选择工作流程"""
        workflow_type = intent["workflow_type"]
        return self.workflows.get(workflow_type, self.workflows["data_service"])

    def _execute_workflow(self, workflow: List[Dict], intent: Dict) -> List[Dict]:
        """执行工作流程"""
        results = []
        data_name = intent["data_name"]

        for step in workflow:
            tool_name = step["tool"]
            description = step["description"]

            logger.info(f"执行步骤: {description}")

            try:
                # 获取工具
                tool = self.tools.get(tool_name)
                if not tool:
                    raise Exception(f"工具 {tool_name} 不存在")

                # 准备工具参数
                tool_params = self._prepare_tool_params(tool_name, data_name, results)

                # 执行工具
                result = tool._run(**tool_params)

                step_result = {
                    "step": description,
                    "tool": tool_name,
                    "params": tool_params,
                    "result": result,
                    "status": "success"
                }

                results.append(step_result)
                logger.info(f"步骤完成: {description}")

            except Exception as e:
                error_result = {
                    "step": description,
                    "tool": tool_name,
                    "error": str(e),
                    "status": "failed"
                }
                results.append(error_result)
                logger.error(f"步骤失败: {description}, 错误: {e}")

        return results

    def _prepare_tool_params(self, tool_name: str, data_name: str, previous_results: List[Dict]) -> Dict:
        """准备工具参数"""
        if tool_name == "data_query":
            return {
                "data_name": data_name,
                "query_type": "schema"
            }
        elif tool_name == "data_collection":
            return {
                "data_name": data_name,
                "source_type": "database",
                "collection_config": {}
            }
        elif tool_name == "data_storage":
            # 从前面的采集结果中获取数据路径
            data_content = "采集的数据"
            for result in previous_results:
                if result["tool"] == "data_collection" and result["status"] == "success":
                    try:
                        result_data = json.loads(result["result"])
                        data_content = result_data.get("data_file_path", "采集的数据")
                    except:
                        pass

            return {
                "data_name": data_name,
                "data_content": data_content,
                "storage_config": {}
            }
        elif tool_name == "data_service":
            return {
                "data_name": data_name,
                "service_type": "rest_api",
                "service_config": {}
            }
        else:
            return {"data_name": data_name}

    def _generate_summary(self, user_request: str, intent: Dict, execution_results: List[Dict]) -> str:
        """生成执行总结"""
        data_name = intent["data_name"]
        operation = intent["operation"]

        # 统计执行结果
        total_steps = len(execution_results)
        success_steps = sum(1 for r in execution_results if r["status"] == "success")
        failed_steps = total_steps - success_steps

        # 生成总结
        summary = f"""
📋 任务执行总结

用户请求: {user_request}
数据名称: {data_name}
操作类型: {operation}

执行结果:
- 总步骤数: {total_steps}
- 成功步骤: {success_steps}
- 失败步骤: {failed_steps}

详细步骤:
"""

        for i, result in enumerate(execution_results, 1):
            status_icon = "✅" if result["status"] == "success" else "❌"
            summary += f"{i}. {status_icon} {result['step']}\n"

            if result["status"] == "success":
                # 尝试解析结果中的关键信息
                try:
                    result_data = json.loads(result["result"])
                    if "service_url" in result_data:
                        summary += f"   🔗 服务地址: {result_data['service_url']}\n"
                    elif "collected_records" in result_data:
                        summary += f"   📊 采集记录数: {result_data['collected_records']}\n"
                    elif "stored_records" in result_data:
                        summary += f"   💾 入库记录数: {result_data['stored_records']}\n"
                except:
                    pass
            else:
                summary += f"   ❌ 错误: {result.get('error', '未知错误')}\n"

        # 添加最终结果
        if failed_steps == 0:
            summary += f"\n🎉 任务执行成功！{data_name}的{operation}操作已完成。"
        else:
            summary += f"\n⚠️ 任务部分失败，{failed_steps}个步骤执行失败，请检查错误信息。"

        return summary

    def get_available_workflows(self) -> Dict[str, List[str]]:
        """获取可用的工作流程"""
        return {
            workflow_name: [step["description"] for step in steps]
            for workflow_name, steps in self.workflows.items()
        }

    def get_available_tools(self) -> List[str]:
        """获取可用工具"""
        return list(self.tools.keys())


def main():
    """演示简化版Agent"""
    print("=" * 60)
    print("🤖 简化版数据平台Agent演示")
    print("=" * 60)

    # 检查模型路径
    classifier_path = r"E:\project\python\slm_model_train\src\train\trained_classifiers\question_classifier"
    qa_model_path = r"E:\project\python\slm_model_train\src\train\trained_models\final_model"

    if not os.path.exists(classifier_path):
        classifier_path = None
        print("未找到分类器模型，将使用规则分类")

    if not os.path.exists(qa_model_path):
        qa_model_path = None
        print("未找到问答模型，将使用基础模型")

    # 初始化Agent
    agent = SimpleDataPlatformAgent(
        classifier_path=classifier_path,
        qa_model_path=qa_model_path
    )

    print("Agent初始化完成！")
    print(f"可用工具: {', '.join(agent.get_available_tools())}")
    print(f"可用工作流程: {list(agent.get_available_workflows().keys())}")
    print("\n" + "=" * 60)

    # 测试复杂任务
    test_requests = [
        "我想对用户行为数据进行发服务",
        "需要分析销售数据",
        "帮我迁移订单数据"
    ]

    for request in test_requests:
        print(f"\n📝 测试请求: {request}")
        print("-" * 60)

        result = agent.process_request(request)

        if result["status"] == "success":
            print("✅ 处理成功")
            print(f"意图分析: {result['intent']}")
            print(f"执行总结:\n{result['summary']}")
        else:
            print(f"❌ 处理失败: {result['error']}")

        print("-" * 60)

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
            result = agent.process_request(user_input)

            if result["status"] == "success":
                print("✅ 任务处理完成！")
                print(result["summary"])
            else:
                print(f"❌ 任务处理失败: {result['error']}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 处理错误: {e}")

    print("\n👋 感谢使用简化版数据平台Agent！")


if __name__ == "__main__":
    main()
