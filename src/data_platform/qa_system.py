"""
数据平台问答系统主接口
提供交互式问答功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_qa_model import SimpleDataPlatformQAModel
from knowledge_base import DataPlatformKnowledgeBase

class DataPlatformQASystem:
    def __init__(self):
        """
        初始化问答系统
        """
        print("正在初始化数据平台问答系统...")
        print("加载BERT模型和知识库...")
        
        try:
            self.qa_model = SimpleDataPlatformQAModel()
            print("✓ 问答系统初始化成功！")
            print("✓ BERT模型加载完成")
            print("✓ 知识库加载完成")
        except Exception as e:
            print(f"✗ 初始化失败: {e}")
            # 如果BERT模型加载失败，回退到基础知识库模式
            print("回退到基础知识库模式...")
            self.qa_model = None
            self.knowledge_base = DataPlatformKnowledgeBase()
            print("✓ 基础知识库模式初始化成功")
    
    def ask_question(self, question):
        """
        提问接口
        """
        if self.qa_model:
            # 使用完整的BERT问答模型
            result = self.qa_model.answer_question(question)
            return result
        else:
            # 使用基础知识库
            answer = self.knowledge_base.get_answer(question)
            return {
                "answer": answer,
                "method": "knowledge_base_only",
                "confidence": 1.0 if answer != "抱歉，我无法回答这个问题。请尝试询问关于数据清洗、数据入库、数据处理、数据监控或数据安全相关的问题。" else 0.0
            }
    
    def show_available_topics(self):
        """
        显示可回答的主题
        """
        if self.qa_model:
            topics = self.qa_model.get_available_topics()
        else:
            topics = self.knowledge_base.get_all_topics()
        
        print("\n=== 可回答的数据平台相关主题 ===")
        for i, topic in enumerate(topics, 1):
            print(f"{i}. {topic}")
        print("=" * 40)
    
    def interactive_mode(self):
        """
        交互式问答模式
        """
        print("\n" + "=" * 50)
        print("🤖 数据平台智能问答系统")
        print("=" * 50)
        print("欢迎使用数据平台问答系统！")
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
                    self.show_available_topics()
                    continue
                
                # 处理问题
                print("\n🔍 正在分析您的问题...")
                result = self.ask_question(question)
                
                # 显示结果
                print("\n" + "=" * 30)
                print("📋 回答：")
                print(result["answer"])
                
                # 显示匹配信息
                if result["method"] == "semantic_match":
                    print(f"\n🎯 匹配到的问题：{result['matched_question']}")
                    print(f"🔢 相似度：{result['confidence']:.2f}")
                elif result["method"] == "direct_match":
                    print("\n✅ 直接匹配成功")
                
                print("=" * 30)
                
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用数据平台问答系统，再见！")
                break
            except Exception as e:
                print(f"\n❌ 处理问题时出现错误：{e}")
    
    def batch_ask(self, questions):
        """
        批量提问
        """
        results = []
        for question in questions:
            result = self.ask_question(question)
            results.append({
                "question": question,
                "result": result
            })
        return results

# def main():
#     """
#     主函数 - 启动问答系统
#     """
#     qa_system = DataPlatformQASystem()
#     qa_system.interactive_mode()
#
# if __name__ == "__main__":
#     main()
