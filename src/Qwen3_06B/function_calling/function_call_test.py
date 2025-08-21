"""
Qwen3-0.6B Function Calling 微调模型测试脚本
用于验证微调后模型的Function Calling能力
"""

import os
import json
import torch
import time
import re
from typing import List, Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.dataset.function_calling import function_calling_dataset


class QwenFunctionCallingTester:
    def __init__(self,
                 base_model_path: str,
                 peft_model_path: str,
                 max_length: int = 1024):
        """
        初始化Qwen Function Calling测试器
        
        Args:
            base_model_path: 基础模型路径
            peft_model_path: PEFT微调模型路径
            max_length: 最大序列长度
        """
        self.base_model_path = base_model_path
        self.peft_model_path = peft_model_path
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

        print(f"基础模型路径: {base_model_path}")
        print(f"PEFT模型路径: {peft_model_path}")
        print(f"最大序列长度: {max_length}")

    def load_model(self):
        """加载微调后的模型和分词器"""
        print("\n正在加载微调后的模型...")

        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side="right"
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # 加载PEFT适配器
            self.model = PeftModel.from_pretrained(
                base_model,
                self.peft_model_path,
                torch_dtype=torch.float16
            )

            # 合并适配器权重以提高推理速度（可选）
            # self.model = self.model.merge_and_unload()

            self.model.eval()
            print("✅ 模型加载成功!")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 如果PEFT模型不存在，尝试加载基础模型进行对比测试
            print("尝试加载基础模型进行对比测试...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_path,
                    trust_remote_code=True,
                    padding_side="right"
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("✅ 基础模型加载成功（用于对比测试）!")

            except Exception as e2:
                print(f"❌ 基础模型也加载失败: {e2}")
                raise

    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """从模型响应中提取工具调用"""
        tool_calls = []

        # 查找<tool_call>标签
        tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(tool_call_pattern, response, re.DOTALL)

        for match in matches:
            try:
                # 解析函数调用格式：function_name(arguments)
                func_pattern = r'(\w+)\((.*?)\)'
                func_match = re.match(func_pattern, match.strip())

                if func_match:
                    func_name = func_match.group(1)
                    func_args = func_match.group(2)

                    # 尝试解析JSON参数
                    try:
                        if func_args.strip():
                            args_dict = json.loads(func_args)
                        else:
                            args_dict = {}
                    except:
                        # 如果不是JSON格式，尝试简单解析
                        args_dict = {"raw_args": func_args}

                    tool_calls.append({
                        "function": {
                            "name": func_name,
                            "arguments": args_dict
                        }
                    })
            except Exception as e:
                print(f"解析工具调用时出错: {e}")
                continue

        return tool_calls

    def generate_response(self,
                          messages: List[Dict[str, str]],
                          tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成模型响应"""
        try:
            # 构建系统消息
            system_content = "You are a helpful assistant that can use tools to help users."
            if tools:
                system_content += "\n\nAvailable tools:\n"
                for tool in tools:
                    tool_info = tool["function"]
                    system_content += f"- {tool_info['name']}: {tool_info['description']}\n"
                    system_content += f"  Parameters: {json.dumps(tool_info['parameters'], ensure_ascii=False)}\n"

            # 构建完整对话
            full_messages = [{"role": "system", "content": system_content}] + messages

            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            # 分词
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            # 生成配置
            generation_config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # 生成响应
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            generation_time = time.time() - start_time

            # 解码响应
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # 提取工具调用
            tool_calls = self.extract_tool_calls(response)

            return {
                "content": response,
                "tool_calls": tool_calls,
                "generation_time": generation_time,
                "input_length": inputs['input_ids'].shape[1],
                "output_length": outputs[0].shape[0] - inputs['input_ids'].shape[1]
            }

        except Exception as e:
            print(f"生成响应时出错: {e}")
            return {
                "content": f"生成失败: {str(e)}",
                "tool_calls": [],
                "generation_time": 0,
                "input_length": 0,
                "output_length": 0
            }

    def evaluate_tool_call_accuracy(self,
                                    expected_tool_calls: List[Dict[str, Any]],
                                    actual_tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估工具调用的准确性"""
        if not expected_tool_calls and not actual_tool_calls:
            return {"accuracy": 1.0, "details": "无工具调用，匹配正确"}

        if not expected_tool_calls:
            return {"accuracy": 0.0, "details": "不应该有工具调用，但模型生成了工具调用"}

        if not actual_tool_calls:
            return {"accuracy": 0.0, "details": "应该有工具调用，但模型没有生成"}

        # 检查工具调用数量
        if len(expected_tool_calls) != len(actual_tool_calls):
            return {
                "accuracy": 0.0,
                "details": f"工具调用数量不匹配：期望{len(expected_tool_calls)}个，实际{len(actual_tool_calls)}个"
            }

        # 检查每个工具调用
        correct_calls = 0
        details = []

        for i, (expected, actual) in enumerate(zip(expected_tool_calls, actual_tool_calls)):
            expected_name = expected["function"]["name"]
            actual_name = actual["function"]["name"]

            if expected_name == actual_name:
                correct_calls += 1
                details.append(f"工具{i + 1}: ✅ 函数名正确 ({expected_name})")

                # 检查参数（简化版本，只检查关键参数）
                expected_args = expected["function"]["arguments"]
                actual_args = actual["function"]["arguments"]

                if isinstance(expected_args, str):
                    try:
                        expected_args = json.loads(expected_args)
                    except:
                        pass

                # 检查关键参数是否存在
                key_params_match = True
                if isinstance(expected_args, dict) and isinstance(actual_args, dict):
                    for key in expected_args.keys():
                        if key not in actual_args:
                            key_params_match = False
                            break

                if key_params_match:
                    details.append(f"工具{i + 1}: ✅ 参数结构正确")
                else:
                    details.append(f"工具{i + 1}: ⚠️ 参数结构可能有问题")
            else:
                details.append(f"工具{i + 1}: ❌ 函数名错误 (期望: {expected_name}, 实际: {actual_name})")

        accuracy = correct_calls / len(expected_tool_calls)
        return {
            "accuracy": accuracy,
            "details": "; ".join(details),
            "correct_calls": correct_calls,
            "total_calls": len(expected_tool_calls)
        }

    def test_single_case(self, test_case: Dict[str, Any], case_index: int) -> Dict[str, Any]:
        """测试单个案例"""
        print(f"\n{'=' * 60}")
        print(f"测试案例 {case_index + 1}")
        print(f"{'=' * 60}")

        messages = test_case["messages"]
        tools = test_case.get("tools", [])

        # 提取用户消息和期望的助手回复
        user_message = None
        expected_assistant_message = None

        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "assistant":
                expected_assistant_message = msg

        if not user_message:
            return {"error": "未找到用户消息"}

        print(f"用户输入: {user_message}")

        # 生成模型响应
        test_messages = [{"role": "user", "content": user_message}]
        result = self.generate_response(test_messages, tools)

        print(f"模型回复: {result['content']}")
        print(f"生成时间: {result['generation_time']:.2f}秒")
        print(f"输入长度: {result['input_length']} tokens")
        print(f"输出长度: {result['output_length']} tokens")

        # 评估工具调用准确性
        expected_tool_calls = expected_assistant_message.get("tool_calls", []) if expected_assistant_message else []
        actual_tool_calls = result["tool_calls"]

        print(f"\n期望工具调用: {len(expected_tool_calls)}个")
        for i, tool_call in enumerate(expected_tool_calls):
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            print(f"  {i + 1}. {func_name}({func_args})")

        print(f"\n实际工具调用: {len(actual_tool_calls)}个")
        for i, tool_call in enumerate(actual_tool_calls):
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            print(f"  {i + 1}. {func_name}({func_args})")

        # 计算准确性
        accuracy_result = self.evaluate_tool_call_accuracy(expected_tool_calls, actual_tool_calls)
        print(f"\n准确性评估:")
        print(f"  准确率: {accuracy_result['accuracy']:.2%}")
        print(f"  详情: {accuracy_result['details']}")

        return {
            "case_index": case_index,
            "user_input": user_message,
            "model_response": result['content'],
            "expected_tool_calls": expected_tool_calls,
            "actual_tool_calls": actual_tool_calls,
            "accuracy": accuracy_result['accuracy'],
            "generation_time": result['generation_time'],
            "input_length": result['input_length'],
            "output_length": result['output_length'],
            "accuracy_details": accuracy_result['details']
        }

    def run_dataset_tests(self) -> Dict[str, Any]:
        """运行数据集中的所有测试案例"""
        print(f"\n{'=' * 80}")
        print("开始运行Function Calling数据集测试")
        print(f"总测试案例数: {len(function_calling_dataset)}")
        print(f"{'=' * 80}")

        results = []
        total_accuracy = 0
        total_generation_time = 0

        for i, test_case in enumerate(function_calling_dataset):
            try:
                result = self.test_single_case(test_case, i)
                if "error" not in result:
                    results.append(result)
                    total_accuracy += result["accuracy"]
                    total_generation_time += result["generation_time"]
                else:
                    print(f"案例 {i + 1} 测试失败: {result['error']}")
            except Exception as e:
                print(f"案例 {i + 1} 执行时出错: {e}")
                continue

        # 计算总体统计
        if results:
            avg_accuracy = total_accuracy / len(results)
            avg_generation_time = total_generation_time / len(results)

            # 按准确率分类统计
            perfect_cases = sum(1 for r in results if r["accuracy"] == 1.0)
            good_cases = sum(1 for r in results if 0.5 <= r["accuracy"] < 1.0)
            poor_cases = sum(1 for r in results if r["accuracy"] < 0.5)

            print(f"\n{'=' * 80}")
            print("测试结果汇总")
            print(f"{'=' * 80}")
            print(f"总测试案例数: {len(results)}")
            print(f"平均准确率: {avg_accuracy:.2%}")
            print(f"平均生成时间: {avg_generation_time:.2f}秒")
            print(f"\n准确率分布:")
            print(f"  完美 (100%): {perfect_cases}个案例 ({perfect_cases / len(results):.1%})")
            print(f"  良好 (50%-99%): {good_cases}个案例 ({good_cases / len(results):.1%})")
            print(f"  较差 (<50%): {poor_cases}个案例 ({poor_cases / len(results):.1%})")

            return {
                "total_cases": len(results),
                "avg_accuracy": avg_accuracy,
                "avg_generation_time": avg_generation_time,
                "perfect_cases": perfect_cases,
                "good_cases": good_cases,
                "poor_cases": poor_cases,
                "detailed_results": results
            }
        else:
            print("没有成功执行的测试案例")
            return {"error": "没有成功执行的测试案例"}

    def run_custom_tests(self) -> Dict[str, Any]:
        """运行自定义测试案例"""
        print(f"\n{'=' * 80}")
        print("开始运行自定义测试案例")
        print(f"{'=' * 80}")

        # 自定义测试案例
        custom_test_cases = [
            {
                "name": "天气查询测试",
                "messages": [{"role": "user", "content": "深圳今天天气如何？"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "获取指定地点的天气信息",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string", "description": "城市名称"},
                                    "date": {"type": "string", "description": "日期"}
                                },
                                "required": ["location"]
                            }
                        }
                    }
                ],
                "expected_function": "get_weather"
            },
            {
                "name": "数学计算测试",
                "messages": [{"role": "user", "content": "计算 15 * 8 + 32 的结果"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "description": "执行数学计算",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "expression": {"type": "string", "description": "数学表达式"}
                                },
                                "required": ["expression"]
                            }
                        }
                    }
                ],
                "expected_function": "calculate"
            },
            {
                "name": "时间查询测试",
                "messages": [{"role": "user", "content": "现在是几点？"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_time",
                            "description": "获取当前时间",
                            "parameters": {"type": "object", "properties": {}, "required": []}
                        }
                    }
                ],
                "expected_function": "get_current_time"
            },
            {
                "name": "无工具调用测试",
                "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
                "tools": [],
                "expected_function": None
            },
            {
                "name": "多工具选择测试",
                "messages": [{"role": "user", "content": "帮我搜索人工智能的信息"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "在网络上搜索信息",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "搜索关键词"}
                                },
                                "required": ["query"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "获取天气信息",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string", "description": "城市名称"}
                                },
                                "required": ["location"]
                            }
                        }
                    }
                ],
                "expected_function": "web_search"
            }
        ]

        results = []
        for i, test_case in enumerate(custom_test_cases):
            print(f"\n{'=' * 60}")
            print(f"自定义测试 {i + 1}: {test_case['name']}")
            print(f"{'=' * 60}")

            user_input = test_case["messages"][0]["content"]
            tools = test_case["tools"]
            expected_function = test_case.get("expected_function")

            print(f"用户输入: {user_input}")
            print(f"期望函数: {expected_function or '无'}")

            # 生成响应
            result = self.generate_response(test_case["messages"], tools)

            print(f"模型回复: {result['content']}")
            print(f"生成时间: {result['generation_time']:.2f}秒")

            # 评估结果
            actual_functions = [call["function"]["name"] for call in result["tool_calls"]]

            if expected_function is None:
                # 不应该有工具调用
                success = len(actual_functions) == 0
                print(
                    f"评估结果: {'✅ 正确' if success else '❌ 错误'} - {'无工具调用' if success else f'意外调用了: {actual_functions}'}")
            else:
                # 应该调用特定函数
                success = expected_function in actual_functions
                print(
                    f"评估结果: {'✅ 正确' if success else '❌ 错误'} - {'调用了期望函数' if success else f'实际调用: {actual_functions}'}")

            results.append({
                "name": test_case["name"],
                "success": success,
                "expected_function": expected_function,
                "actual_functions": actual_functions,
                "generation_time": result["generation_time"],
                "response": result["content"]
            })

        # 统计结果
        successful_tests = sum(1 for r in results if r["success"])
        success_rate = successful_tests / len(results) if results else 0

        print(f"\n{'=' * 80}")
        print("自定义测试结果汇总")
        print(f"{'=' * 80}")
        print(f"总测试数: {len(results)}")
        print(f"成功测试数: {successful_tests}")
        print(f"成功率: {success_rate:.2%}")

        return {
            "total_tests": len(results),
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "detailed_results": results
        }

    def interactive_test(self):
        """交互式测试模式"""
        print(f"\n{'=' * 80}")
        print("进入交互式测试模式")
        print("输入 'quit' 或 'exit' 退出")
        print(f"{'=' * 80}")

        # 定义一些常用工具
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定地点的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名称"},
                            "date": {"type": "string", "description": "日期"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "数学表达式"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前时间",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "在网络上搜索信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        while True:
            try:
                user_input = input("\n请输入您的问题: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("退出交互式测试模式")
                    break

                if not user_input:
                    continue

                print(f"\n处理中...")

                # 生成响应
                messages = [{"role": "user", "content": user_input}]
                result = self.generate_response(messages, available_tools)

                print(f"\n{'=' * 60}")
                print(f"用户: {user_input}")
                print(f"助手1: {result}")
                print(f"助手: {result['content']}")

                if result["tool_calls"]:
                    print(f"\n工具调用:")
                    for i, tool_call in enumerate(result["tool_calls"]):
                        func_name = tool_call["function"]["name"]
                        func_args = tool_call["function"]["arguments"]
                        print(f"  {i + 1}. {func_name}({func_args})")

                print(f"\n生成时间: {result['generation_time']:.2f}秒")
                print(f"{'=' * 60}")

            except KeyboardInterrupt:
                print("\n\n用户中断，退出交互式测试模式")
                break
            except Exception as e:
                print(f"处理时出错: {e}")
                continue

    def save_test_results(self, results: Dict[str, Any], output_file: str = "test_results.json"):
        """保存测试结果到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"测试结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存测试结果失败: {e}")


def main():
    """主函数"""
    print("Qwen3-0.6B Function Calling 微调模型测试")
    print("=" * 80)

    # 配置模型路径
    base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
    peft_model_path = r"E:\project\llm\model-data\train-models\Qwen3-function-calling"

    # 检查路径是否存在
    if not os.path.exists(base_model_path):
        print(f"❌ 基础模型路径不存在: {base_model_path}")
        print("请确保Qwen3-0.6B模型已下载到指定路径")
        return

    # 创建测试器
    tester = QwenFunctionCallingTester(
        base_model_path=base_model_path,
        peft_model_path=peft_model_path,
        max_length=1024
    )

    try:
        # 加载模型
        tester.load_model()

        # 选择测试模式
        print(f"\n请选择测试模式:")
        print("1. 运行数据集测试 (测试训练数据集中的所有案例)")
        print("2. 运行自定义测试 (测试预定义的自定义案例)")
        print("3. 交互式测试 (手动输入问题进行测试)")
        print("4. 运行所有测试")

        while True:
            try:
                choice = input("\n请输入选择 (1-4): ").strip()

                if choice == "1":
                    # 运行数据集测试
                    results = tester.run_dataset_tests()
                    if "error" not in results:
                        tester.save_test_results(results, "dataset_test_results.json")
                    break

                elif choice == "2":
                    # 运行自定义测试
                    results = tester.run_custom_tests()
                    tester.save_test_results(results, "custom_test_results.json")
                    break

                elif choice == "3":
                    # 交互式测试
                    tester.interactive_test()
                    break

                elif choice == "4":
                    # 运行所有测试
                    print("\n开始运行所有测试...")

                    # 1. 数据集测试
                    dataset_results = tester.run_dataset_tests()
                    if "error" not in dataset_results:
                        tester.save_test_results(dataset_results, "dataset_test_results.json")

                    # 2. 自定义测试
                    custom_results = tester.run_custom_tests()
                    tester.save_test_results(custom_results, "custom_test_results.json")

                    # 3. 综合报告
                    print(f"\n{'=' * 80}")
                    print("综合测试报告")
                    print(f"{'=' * 80}")

                    if "error" not in dataset_results:
                        print(f"数据集测试:")
                        print(f"  总案例数: {dataset_results['total_cases']}")
                        print(f"  平均准确率: {dataset_results['avg_accuracy']:.2%}")
                        print(f"  完美案例: {dataset_results['perfect_cases']}")

                    print(f"\n自定义测试:")
                    print(f"  总测试数: {custom_results['total_tests']}")
                    print(f"  成功率: {custom_results['success_rate']:.2%}")
                    print(f"  成功测试数: {custom_results['successful_tests']}")

                    # 保存综合结果
                    combined_results = {
                        "dataset_results": dataset_results,
                        "custom_results": custom_results,
                        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    tester.save_test_results(combined_results, "combined_test_results.json")

                    print(f"\n所有测试完成！结果已保存到相应的JSON文件中。")
                    break

                else:
                    print("无效选择，请输入 1-4")

            except KeyboardInterrupt:
                print("\n\n用户中断测试")
                break
            except Exception as e:
                print(f"执行测试时出错: {e}")
                break

    except Exception as e:
        print(f"初始化测试器时出错: {e}")
        print("请检查模型路径和依赖库是否正确安装")


if __name__ == "__main__":
    main()
