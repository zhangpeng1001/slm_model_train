"""
Qwen3-0.6B LoRA微调模型推理测试工具
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import time

class QwenLoRAInferenceTester:
    def __init__(self, 
                 base_model_path: str,
                 lora_model_path: str):
        """
        初始化推理测试器
        
        Args:
            base_model_path: 基础模型路径
            lora_model_path: LoRA微调模型路径
        """
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.base_model = None
        self.lora_model = None
        self.tokenizer = None
        
        print(f"基础模型路径: {base_model_path}")
        print(f"LoRA模型路径: {lora_model_path}")
        
    def load_models(self):
        """加载基础模型和LoRA微调模型"""
        print("正在加载模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        print("加载基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 加载LoRA微调模型
        print("加载LoRA微调模型...")
        if os.path.exists(self.lora_model_path):
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                self.lora_model_path,
                torch_dtype=torch.float16
            )
            print("LoRA模型加载成功!")
        else:
            print(f"警告: LoRA模型路径不存在: {self.lora_model_path}")
            print("将只使用基础模型进行测试")
            
        print("模型加载完成!")
        
    def generate_response(self, prompt: str, model, max_new_tokens: int = 256):
        """生成回复"""
        # 构建消息格式
        messages = [{"role": "user", "content": prompt}]
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # 分词
        model_inputs = self.tokenizer([text], return_tensors="pt").to(model.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        return response
        
    def test_single_prompt(self, prompt: str):
        """测试单个提示"""
        print(f"\n{'='*60}")
        print(f"测试提示: {prompt}")
        print(f"{'='*60}")
        
        # 基础模型回复
        print("\n【基础模型回复】:")
        start_time = time.time()
        base_response = self.generate_response(prompt, self.base_model)
        base_time = time.time() - start_time
        print(f"{base_response}")
        print(f"生成时间: {base_time:.2f}秒")
        
        # LoRA微调模型回复
        if self.lora_model is not None:
            print("\n【LoRA微调模型回复】:")
            start_time = time.time()
            lora_response = self.generate_response(prompt, self.lora_model)
            lora_time = time.time() - start_time
            print(f"{lora_response}")
            print(f"生成时间: {lora_time:.2f}秒")
            
            # 对比分析
            print(f"\n【对比分析】:")
            print(f"基础模型长度: {len(base_response)} 字符")
            print(f"微调模型长度: {len(lora_response)} 字符")
            print(f"时间差异: {abs(lora_time - base_time):.2f}秒")
        
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("\n" + "="*80)
        print("开始Qwen3-0.6B LoRA微调模型综合测试")
        print("="*80)
        
        # 测试用例
        test_cases = [
            # 问题场景分类测试
            {
                "category": "问题场景分类",
                "prompts": [
                    "任务: 问题场景分类\n输入: 数据清洗流程是什么？",
                    "任务: 问题场景分类\n输入: 你好",
                    "任务: 问题场景分类\n输入: 今天天气怎么样？",
                    "任务: 问题场景分类\n输入: 如何进行数据入库？",
                ]
            },
            # 问题类型分类测试
            {
                "category": "问题类型分类", 
                "prompts": [
                    "任务: 问题类型分类\n输入: 我有一批西安市地类图斑的数据，怎么进行发服务",
                    "任务: 问题类型分类\n输入: 请帮我把实景三维模型成果数据进行治理",
                    "任务: 问题类型分类\n输入: DEM高程数据怎么进行质量检查",
                ]
            },
            # 问题回答测试
            {
                "category": "问题回答",
                "prompts": [
                    "任务: 问题回答\n输入: 数据清洗流程",
                    "任务: 问题回答\n输入: 数据入库流程",
                    "任务: 问题回答\n输入: 数据质量检查",
                ]
            },
            # 文件名称提取测试
            {
                "category": "文件名称提取",
                "prompts": [
                    "任务: 文件名称提取\n输入: 我有一批西安市地类图斑的数据，怎么进行发服务",
                    "任务: 文件名称提取\n输入: 请帮我把实景三维模型成果数据进行治理",
                    "任务: 文件名称提取\n输入: DEM高程数据怎么进行质量检查",
                ]
            }
        ]
        
        # 执行测试
        for test_case in test_cases:
            print(f"\n{'='*80}")
            print(f"测试类别: {test_case['category']}")
            print(f"{'='*80}")
            
            for i, prompt in enumerate(test_case['prompts'], 1):
                print(f"\n--- 测试 {i}/{len(test_case['prompts'])} ---")
                self.test_single_prompt(prompt)
        
        print(f"\n{'='*80}")
        print("综合测试完成!")
        print(f"{'='*80}")
        
    def interactive_test(self):
        """交互式测试"""
        print("\n" + "="*60)
        print("进入交互式测试模式")
        print("输入 'quit' 或 'exit' 退出")
        print("="*60)
        
        while True:
            try:
                prompt = input("\n请输入测试提示: ").strip()
                if prompt.lower() in ['quit', 'exit', '退出']:
                    print("退出交互式测试模式")
                    break
                    
                if not prompt:
                    print("请输入有效的提示")
                    continue
                    
                self.test_single_prompt(prompt)
                
            except KeyboardInterrupt:
                print("\n退出交互式测试模式")
                break
            except Exception as e:
                print(f"测试出错: {e}")
                
    def evaluate_performance(self):
        """评估模型性能"""
        print("\n" + "="*60)
        print("模型性能评估")
        print("="*60)
        
        # 预期答案测试
        evaluation_cases = [
            {
                "prompt": "任务: 问题场景分类\n输入: 数据清洗流程是什么？",
                "expected": "数据平台相关问题",
                "task": "问题场景分类"
            },
            {
                "prompt": "任务: 问题场景分类\n输入: 你好",
                "expected": "通用问题", 
                "task": "问题场景分类"
            },
            {
                "prompt": "任务: 问题类型分类\n输入: 我有一批西安市地类图斑的数据，怎么进行发服务",
                "expected": "问题回答",
                "task": "问题类型分类"
            },
            {
                "prompt": "任务: 文件名称提取\n输入: 我有一批西安市地类图斑的数据，怎么进行发服务",
                "expected": "西安市地类图斑",
                "task": "文件名称提取"
            }
        ]
        
        correct_base = 0
        correct_lora = 0
        total = len(evaluation_cases)
        
        for i, case in enumerate(evaluation_cases, 1):
            print(f"\n--- 评估测试 {i}/{total} ---")
            print(f"任务: {case['task']}")
            print(f"输入: {case['prompt'].split('输入: ')[1]}")
            print(f"预期输出: {case['expected']}")
            
            # 基础模型测试
            base_response = self.generate_response(case['prompt'], self.base_model, max_new_tokens=50)
            base_match = case['expected'].lower() in base_response.lower()
            if base_match:
                correct_base += 1
            print(f"基础模型输出: {base_response}")
            print(f"基础模型匹配: {'✓' if base_match else '✗'}")
            
            # LoRA模型测试
            if self.lora_model is not None:
                lora_response = self.generate_response(case['prompt'], self.lora_model, max_new_tokens=50)
                lora_match = case['expected'].lower() in lora_response.lower()
                if lora_match:
                    correct_lora += 1
                print(f"LoRA模型输出: {lora_response}")
                print(f"LoRA模型匹配: {'✓' if lora_match else '✗'}")
        
        # 输出评估结果
        print(f"\n{'='*60}")
        print("评估结果:")
        print(f"基础模型准确率: {correct_base}/{total} ({correct_base/total*100:.1f}%)")
        if self.lora_model is not None:
            print(f"LoRA模型准确率: {correct_lora}/{total} ({correct_lora/total*100:.1f}%)")
            improvement = correct_lora - correct_base
            print(f"改进程度: {improvement:+d} 个正确答案")
        print(f"{'='*60}")


def main():
    """主函数"""
    # 配置路径
    base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
    lora_model_path = r"E:\project\llm\model-data\train-models\Qwen3-0.6B"
    
    # 检查路径是否存在
    if not os.path.exists(base_model_path):
        print(f"错误: 基础模型路径不存在: {base_model_path}")
        return
        
    # 创建测试器
    tester = QwenLoRAInferenceTester(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path
    )
    
    # 加载模型
    tester.load_models()
    
    # 运行测试
    print("\n选择测试模式:")
    print("1. 综合测试")
    print("2. 交互式测试") 
    print("3. 性能评估")
    print("4. 全部测试")
    
    try:
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == "1":
            tester.run_comprehensive_test()
        elif choice == "2":
            tester.interactive_test()
        elif choice == "3":
            tester.evaluate_performance()
        elif choice == "4":
            tester.run_comprehensive_test()
            tester.evaluate_performance()
            tester.interactive_test()
        else:
            print("无效选择，运行综合测试")
            tester.run_comprehensive_test()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")


if __name__ == "__main__":
    main()
