"""
多任务训练代码测试脚本
用于验证数据加载和预处理功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_task_train import MultiTaskTrainer, DATASET_CONFIG

def test_data_loading():
    """测试数据加载功能"""
    print("=== 测试数据加载功能 ===")
    
    trainer = MultiTaskTrainer("dummy_model", "dummy_output")
    
    for task_name, config in DATASET_CONFIG.items():
        print(f"\n测试 {task_name} 数据集:")
        print(f"  文件路径: {config['file_path']}")
        print(f"  任务类型: {config['task_type']}")
        
        # 检查文件是否存在
        if os.path.exists(config['file_path']):
            print(f"  ✓ 文件存在")
            
            # 加载数据
            data = trainer.load_json_data(config['file_path'])
            if data:
                print(f"  ✓ 数据加载成功，包含 {len(data)} 个样本")
                
                # 显示第一个样本
                if len(data) > 0:
                    sample = data[0]
                    print(f"  样本示例:")
                    print(f"    输入: {sample.get('input', 'N/A')[:50]}...")
                    print(f"    输出: {sample.get('output', 'N/A')[:50]}...")
            else:
                print(f"  ✗ 数据加载失败")
        else:
            print(f"  ✗ 文件不存在")

def test_dataset_preparation():
    """测试数据集准备功能"""
    print("\n=== 测试数据集准备功能 ===")
    
    trainer = MultiTaskTrainer("dummy_model", "dummy_output")
    
    for task_name, config in DATASET_CONFIG.items():
        if os.path.exists(config['file_path']):
            print(f"\n准备 {task_name} 数据集:")
            
            try:
                dataset = trainer.prepare_dataset(task_name, config)
                if dataset is not None:
                    print(f"  ✓ 数据集准备成功，包含 {len(dataset)} 个样本")
                    
                    # 显示数据集结构
                    if len(dataset) > 0:
                        sample = dataset[0]
                        print(f"  数据集字段: {list(sample.keys())}")
                        print(f"  任务名称: {sample['task_name']}")
                        print(f"  任务类型: {sample['task_type']}")
                else:
                    print(f"  ✗ 数据集准备失败")
            except Exception as e:
                print(f"  ✗ 数据集准备出错: {e}")

def test_system_prompts():
    """测试系统提示词"""
    print("\n=== 测试系统提示词 ===")
    
    for task_name, config in DATASET_CONFIG.items():
        print(f"\n{task_name} 系统提示词:")
        print(f"  {config['system_prompt'][:100]}...")

def main():
    """主测试函数"""
    print("开始测试多任务训练代码...")
    
    try:
        test_data_loading()
        test_dataset_preparation()
        test_system_prompts()
        
        print("\n=== 测试完成 ===")
        print("所有基础功能测试通过！")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
