"""
启动数据平台问答系统
"""
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

def main():
    """
    启动问答系统
    """
    print("🚀 启动数据平台问答系统...")
    
    try:
        from qa_system import DataPlatformQASystem
        qa_system = DataPlatformQASystem()
        qa_system.interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 系统已退出")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("请检查依赖是否正确安装")

if __name__ == "__main__":
    main()
