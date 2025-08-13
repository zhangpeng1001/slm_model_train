"""
数据平台问答系统启动脚本
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """
    启动数据平台问答系统
    """
    print("🚀 启动数据平台问答系统...")
    
    try:
        from data_platform.qa_system import DataPlatformQASystem
        
        # 创建并启动问答系统
        qa_system = DataPlatformQASystem()
        qa_system.interactive_mode()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保您在项目根目录下运行此脚本")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("正在尝试基础模式...")
        
        # 如果完整系统启动失败，尝试基础模式
        try:
            from data_platform.knowledge_base import DataPlatformKnowledgeBase
            
            print("✓ 使用基础知识库模式")
            kb = DataPlatformKnowledgeBase()
            
            print("\n" + "=" * 50)
            print("🤖 数据平台基础问答系统")
            print("=" * 50)
            print("欢迎使用数据平台问答系统（基础模式）！")
            print("您可以询问关于数据清洗、数据入库、数据处理等相关问题")
            print("输入 'help' 查看可回答的主题")
            print("输入 'quit' 或 'exit' 退出系统")
            print("-" * 50)
            
            while True:
                try:
                    question = input("\n❓ 请输入您的问题：").strip()
                    
                    if not question:
                        continue
                    
                    if question.lower() in ['quit', 'exit', '退出']:
                        print("👋 感谢使用数据平台问答系统，再见！")
                        break
                    
                    if question.lower() in ['help', '帮助']:
                        topics = kb.get_all_topics()
                        print("\n=== 可回答的数据平台相关主题 ===")
                        for i, topic in enumerate(topics, 1):
                            print(f"{i}. {topic}")
                        print("=" * 40)
                        continue
                    
                    # 获取答案
                    answer = kb.get_answer(question)
                    
                    # 显示结果
                    print("\n" + "=" * 30)
                    print("📋 回答：")
                    print(answer)
                    print("=" * 30)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 感谢使用数据平台问答系统，再见！")
                    break
                except Exception as inner_e:
                    print(f"\n❌ 处理问题时出现错误：{inner_e}")
                    
        except Exception as final_e:
            print(f"❌ 基础模式也启动失败: {final_e}")

if __name__ == "__main__":
    main()
