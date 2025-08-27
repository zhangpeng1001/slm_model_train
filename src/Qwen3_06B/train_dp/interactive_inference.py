#!/usr/bin/env python3
"""
交互式多任务推理脚本
支持命令行交互使用训练好的模型
"""

import argparse
import sys
from inference_multi_task import MultiTaskInference
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveInference:
    def __init__(self, base_model_path: str, peft_model_path: str, device: str = "auto"):
        """初始化交互式推理"""
        self.inferencer = MultiTaskInference(
            base_model_path=base_model_path,
            peft_model_path=peft_model_path,
            device=device
        )
        
        self.task_menu = {
            "1": ("数据提取", "data_extraction"),
            "2": ("问答", "question_answer"),
            "3": ("问题分类", "question_classify"),
            "4": ("问题类型分类", "question_type_classify"),
            "5": ("工具调用", "tool_calling"),
            "6": ("自定义任务", "custom")
        }
    
    def show_menu(self):
        """显示任务菜单"""
        print("\n" + "="*50)
        print("Qwen3 多任务推理系统")
        print("="*50)
        print("请选择任务类型:")
        for key, (name, _) in self.task_menu.items():
            print(f"{key}. {name}")
        print("0. 退出")
        print("="*50)
    
    def get_user_input(self, prompt: str) -> str:
        """获取用户输入"""
        try:
            return input(prompt).strip()
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            sys.exit(0)
    
    def run_task(self, task_type: str, user_input: str):
        """执行任务"""
        print(f"\n处理中...")
        
        try:
            if task_type == "data_extraction":
                response = self.inferencer.data_extraction(user_input)
            elif task_type == "question_answer":
                response = self.inferencer.question_answer(user_input)
            elif task_type == "question_classify":
                response = self.inferencer.question_classify(user_input)
            elif task_type == "question_type_classify":
                response = self.inferencer.question_type_classify(user_input)
            elif task_type == "tool_calling":
                response = self.inferencer.tool_calling(user_input)
            elif task_type == "custom":
                # 自定义任务
                print("可用任务类型: dp_qa, data_extraction, question_classifier, question_type_classifier, tool_data_platform")
                custom_task = self.get_user_input("请输入任务类型: ")
                response = self.inferencer.generate_response(
                    user_input=user_input,
                    task_type=custom_task
                )
            else:
                response = "未知任务类型"
            
            print(f"\n{'='*20} 结果 {'='*20}")
            print(f"输入: {user_input}")
            print(f"输出: {response}")
            print("="*50)
            
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            logger.error(f"推理错误: {e}")
    
    def run(self):
        """运行交互式推理"""
        print("模型加载完成，开始交互式推理...")
        
        while True:
            self.show_menu()
            choice = self.get_user_input("请选择 (0-6): ")
            
            if choice == "0":
                print("感谢使用，再见！")
                break
            
            if choice not in self.task_menu:
                print("无效选择，请重新输入")
                continue
            
            task_name, task_type = self.task_menu[choice]
            print(f"\n您选择了: {task_name}")
            
            user_input = self.get_user_input("请输入您的问题或指令: ")
            if not user_input:
                print("输入不能为空")
                continue
            
            self.run_task(task_type, user_input)
            
            # 询问是否继续
            continue_choice = self.get_user_input("\n是否继续? (y/n): ").lower()
            if continue_choice in ['n', 'no', '否']:
                print("感谢使用，再见！")
                break


def main():
    parser = argparse.ArgumentParser(description="Qwen3 多任务交互式推理")
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="/home/user/models/Qwen3-0.6B",
        help="基础模型路径"
    )
    parser.add_argument(
        "--peft_model", 
        type=str, 
        default="/home/user/train-models/Qwen3-multi-task",
        help="训练后的LoRA模型路径"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="推理设备"
    )
    
    args = parser.parse_args()
    
    # 启动交互式推理
    interactive = InteractiveInference(
        base_model_path=args.base_model,
        peft_model_path=args.peft_model,
        device=args.device
    )
    
    interactive.run()


if __name__ == "__main__":
    main()
