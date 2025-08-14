"""
Agent系统完整演示
展示基于训练模型的LangChain Agent系统功能
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.langchain_agent.simple_agent import SimpleDataPlatformAgent


def main():
    """主演示程序"""
    print("🚀" * 20)
    print("🤖 数据平台智能Agent系统")
    print("🚀" * 20)
    print()

    print("📋 系统功能介绍:")
    print("• 智能意图识别 - 使用训练的BERT分类器")
    print("• 自动任务分解 - 复杂任务拆分为子任务")
    print("• 工具链调用 - 自动选择和执行数据工具")
    print("• 结果整合 - 生成完整的执行报告")
    print()

    print("🛠️ 可用工具:")
    print("1. 数据查询工具 - 查询数据基本信息")
    print("2. 数据采集工具 - 从各种数据源采集数据")
    print("3. 数据入库工具 - 将数据存储到数据仓库")
    print("4. 数据服务工具 - 发布数据API服务")
    print()

    print("🔄 支持的工作流程:")
    print("• 数据服务发布流程: 查询 → 采集 → 入库 → 发服务")
    print("• 数据分析流程: 查询 → 采集 → 质量检查入库")
    print("• 数据迁移流程: 查询 → 采集 → 迁移存储")
    print()

    # 检查模型状态
    classifier_path = "src/train/trained_classifiers/question_classifier"
    qa_model_path = "src/train/trained_models/final_model"

    print("🔍 检查模型状态:")
    if os.path.exists(classifier_path):
        print("✅ 分类器模型已加载")
    else:
        print("⚠️ 分类器模型未找到，将使用规则分类")
        classifier_path = None

    if os.path.exists(qa_model_path):
        print("✅ 问答模型已加载")
    else:
        print("⚠️ 问答模型未找到，将使用基础模型")
        qa_model_path = None

    print()

    # 初始化Agent
    print("🤖 正在初始化Agent系统...")
    agent = SimpleDataPlatformAgent(
        classifier_path=classifier_path,
        qa_model_path=qa_model_path
    )
    print("✅ Agent系统初始化完成！")
    print()

    # 演示复杂任务处理
    demo_requests = [
        {
            "request": "我想对用户行为数据进行发服务",
            "description": "完整的数据服务发布流程演示"
        },
        {
            "request": "需要分析销售数据的质量",
            "description": "数据分析和质量检查流程演示"
        },
        {
            "request": "帮我迁移订单数据到新系统",
            "description": "数据迁移流程演示"
        }
    ]

    print("🎯 复杂任务处理演示:")
    print("=" * 60)

    for i, demo in enumerate(demo_requests, 1):
        print(f"\n📝 演示 {i}: {demo['description']}")
        print(f"用户请求: {demo['request']}")
        print("-" * 40)

        # 处理请求
        result = agent.process_request(demo['request'])

        if result["status"] == "success":
            print("✅ 处理成功")

            # 显示意图分析
            intent = result["intent"]
            print(f"🎯 意图分析:")
            print(f"   数据名称: {intent['data_name']}")
            print(f"   操作类型: {intent['operation']}")
            print(f"   工作流程: {intent['workflow_type']}")

            # 显示执行步骤
            print(f"🔄 执行步骤:")
            for j, step_result in enumerate(result["execution_results"], 1):
                status = "✅" if step_result["status"] == "success" else "❌"
                print(f"   {j}. {status} {step_result['step']}")

            # 显示总结
            print(f"\n📋 执行总结:")
            print(result["summary"])

        else:
            print(f"❌ 处理失败: {result['error']}")

        print("=" * 60)

    # 交互模式
    print("\n🎮 进入交互模式")
    print("您可以输入复杂的数据处理需求，Agent将自动分解并执行")
    print("示例输入:")
    print("• '我想对客户数据进行发服务'")
    print("• '需要分析产品数据'")
    print("• '帮我迁移用户数据'")
    print("输入 'quit' 退出")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 请描述您的数据处理需求: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                break

            if not user_input:
                continue

            print("🤖 Agent正在分析您的需求...")
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

    print("\n🎉 感谢使用数据平台智能Agent系统！")
    print("💡 提示: 您可以通过训练更多数据来提升Agent的智能程度")
    print("🔧 可以扩展更多工具来支持更复杂的数据处理任务")


if __name__ == "__main__":
    main()
