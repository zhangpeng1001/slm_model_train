"""
PEFT LoRA 数据平台推理测试系统
实现完整的推理流程：问题场景分类 -> 数据平台相关问题处理 -> 问题类型分类 -> 具体处理
"""

import os
import json
import torch
import re
from transformers import BertTokenizer, BertModel
from peft import PeftModel
import logging
from typing import Dict, Any, Tuple
from peft_lora_data_platform import BertForInstructionTuning

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPlatformInferenceSystem:
    """数据平台推理系统"""

    def __init__(self, model_path: str, bert_model_path: str):
        """
        初始化推理系统
        
        Args:
            model_path: LoRA模型路径
            bert_model_path: BERT基础模型路径
        """
        self.model_path = model_path
        self.bert_model_path = bert_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型和tokenizer
        self._load_model()

        logger.info(f"推理系统初始化完成，使用设备: {self.device}")

    def _load_model(self):
        """加载模型和tokenizer"""
        try:
            # 加载tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token

            # 加载基础BERT模型
            bert_model = BertModel.from_pretrained(self.bert_model_path)

            # 使用自定义包装器
            base_model = BertForInstructionTuning(bert_model)

            # 加载LoRA模型
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.to(self.device)
            self.model.eval()

            logger.info("模型加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def _generate_response(self, instruction: str, input_text: str, max_length: int = 512) -> str:
        """
        生成模型响应
        
        Args:
            instruction: 指令
            input_text: 输入文本
            max_length: 最大长度
            
        Returns:
            生成的响应文本
        """
        try:
            # 构建输入文本
            prompt = f"指令：{instruction}\n输入：{input_text}\n输出："

            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # 获取模型输出
                outputs = self.model(**inputs)
                logits = outputs.logits

                # 获取输出部分的logits（从"输出："之后开始）
                output_start_pos = len(self.tokenizer.encode(prompt, add_special_tokens=False))

                # 简单的贪心解码
                predicted_ids = torch.argmax(logits[0], dim=-1)

                # 解码预测的token
                predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)

                # 提取输出部分
                if "输出：" in predicted_text:
                    response = predicted_text.split("输出：")[-1].strip()
                else:
                    response = predicted_text.strip()

                # 清理响应文本
                response = self._clean_response(response)

                return response

        except Exception as e:
            logger.error(f"生成响应失败: {e}")
            return "抱歉，生成响应时出现错误。"

    def _clean_response(self, response: str) -> str:
        """清理响应文本"""
        # 移除特殊token
        response = response.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")

        # 移除多余的空白字符
        response = re.sub(r'\s+', ' ', response).strip()

        # 如果响应为空或太短，返回默认响应
        if len(response) < 2:
            return "无法生成有效响应"

        return response

    def classify_question_scenario(self, question: str) -> str:
        """
        问题场景分类
        
        Args:
            question: 用户问题
            
        Returns:
            分类结果：'数据平台相关问题'、'通用问题'、'无关问题'
        """
        logger.info(f"进行问题场景分类: {question}")

        response = self._generate_response("问题场景分类", question)

        logger.info(f"场景分类结果: {response}")
        return response

    def classify_question_type(self, question: str) -> str:
        """
        问题类型分类（仅对数据平台相关问题）
        
        Args:
            question: 用户问题
            
        Returns:
            分类结果：'问题回答'、'任务处理'
        """
        logger.info(f"进行问题类型分类: {question}")

        response = self._generate_response("问题类型分类", question)

        logger.info(f"类型分类结果: {response}")
        return response

    def answer_question(self, question: str) -> str:
        """
        回答问题
        
        Args:
            question: 用户问题
            
        Returns:
            问题答案
        """
        logger.info(f"回答问题: {question}")

        response = self._generate_response("问题回答", question)

        logger.info(f"问题答案: {response}")
        return response

    def extract_filename(self, question: str) -> str:
        """
        提取文件名称
        
        Args:
            question: 用户问题
            
        Returns:
            提取的文件名称
        """
        logger.info(f"提取文件名称: {question}")

        response = self._generate_response("文件名称提取", question)

        logger.info(f"提取的文件名称: {response}")
        return response

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        完整的问题处理流程
        
        Args:
            question: 用户问题
            
        Returns:
            处理结果字典
        """
        result = {
            "question": question,
            "scenario": None,
            "question_type": None,
            "response": None,
            "filename": None,
            "processing_steps": []
        }

        try:
            # 步骤1: 问题场景分类
            result["processing_steps"].append("1. 问题场景分类")
            scenario = self.classify_question_scenario(question)
            result["scenario"] = scenario

            # 步骤2: 根据场景进行不同处理
            if "数据平台相关问题" in scenario:
                result["processing_steps"].append("2. 识别为数据平台相关问题，继续分类")

                # 步骤3: 问题类型分类
                result["processing_steps"].append("3. 问题类型分类")
                question_type = self.classify_question_type(question)
                result["question_type"] = question_type

                # 步骤4: 根据问题类型进行处理
                if "问题回答" in question_type:
                    result["processing_steps"].append("4. 问题回答类型，调用问题回答")
                    answer = self.answer_question(question)
                    result["response"] = answer

                elif "任务处理" in question_type:
                    result["processing_steps"].append("4. 任务处理类型，提取文件名称")
                    filename = self.extract_filename(question)
                    result["filename"] = filename
                    result["response"] = f"已识别任务处理请求，相关文件：{filename}。请确认是否需要处理此文件。"

                else:
                    result["processing_steps"].append("4. 未识别的问题类型，使用默认回答")
                    result["response"] = "抱歉，无法识别具体的问题类型，请提供更详细的信息。"

            elif "通用问题" in scenario:
                result["processing_steps"].append("2. 识别为通用问题，使用通用回复")
                result["response"] = self.generate_general_response(question)

            elif "无关问题" in scenario:
                result["processing_steps"].append("2. 识别为无关问题，提示用户")
                result["response"] = "抱歉，我是数据平台专用助手，只能处理数据平台相关的问题。请咨询数据清洗、数据入库、数据质量检查等相关问题。"

            else:
                result["processing_steps"].append("2. 场景分类不明确，使用默认回复")
                result["response"] = "抱歉，无法准确理解您的问题，请重新描述或提供更多信息。"

        except Exception as e:
            logger.error(f"问题处理失败: {e}")
            result["response"] = "处理过程中出现错误，请稍后再试。"
            result["processing_steps"].append(f"错误: {str(e)}")

        return result

    def generate_general_response(self, question: str) -> str:
        """
        生成通用回答（对于非数据平台相关问题）
        
        Args:
            question: 用户问题
            
        Returns:
            通用回答
        """
        # 简单的通用回答逻辑
        if any(greeting in question.lower() for greeting in ["你好", "您好", "早上好", "下午好"]):
            return "您好！我是数据平台智能助手，很高兴为您服务。"
        elif any(thanks in question.lower() for thanks in ["谢谢", "感谢"]):
            return "不客气！如果还有其他数据平台相关问题，随时可以咨询我。"
        elif any(bye in question.lower() for bye in ["再见", "拜拜"]):
            return "再见！祝您工作顺利！"
        else:
            return "我是数据平台专用助手，主要处理数据清洗、数据入库、数据质量检查等相关问题。请问有什么可以帮助您的吗？"


class InferenceTestRunner:
    """推理测试运行器"""

    def __init__(self, inference_system: DataPlatformInferenceSystem):
        """
        初始化测试运行器
        
        Args:
            inference_system: 推理系统实例
        """
        self.inference_system = inference_system

    def run_scenario_tests(self):
        """运行4种场景的测试"""
        print("\n" + "=" * 80)
        print("🧪 开始运行4种场景的推理测试")
        print("=" * 80)

        # 测试用例
        test_cases = [
            # 场景1: 数据平台相关问题 - 问题回答类型
            {
                "category": "数据平台相关问题 - 问题回答",
                "questions": [
                    "数据清洗流程是什么？",
                    "如何进行数据入库？",
                    "数据质量检查包括哪些内容？",
                    "数据监控怎么做？",
                    "如何保证数据安全？"
                ]
            },

            # 场景2: 数据平台相关问题 - 任务处理类型
            {
                "category": "数据平台相关问题 - 任务处理",
                "questions": [
                    "请帮我把实景三维模型成果数据进行治理",
                    "我有一批西安市地类图斑的数据，怎么进行发服务",
                    "我已经上传了单波段浮点投影的数据，现在想进行入库",
                    "有一些遥感影像数据需要处理",
                    "DEM高程数据需要进行质量检查"
                ]
            },

            # 场景3: 通用问题
            {
                "category": "通用问题",
                "questions": [
                    "你好",
                    "您好，请问",
                    "谢谢你的帮助",
                    "再见",
                    "早上好"
                ]
            },

            # 场景4: 无关问题
            {
                "category": "无关问题",
                "questions": [
                    "今天天气怎么样？",
                    "北京有什么好吃的？",
                    "如何学习英语？",
                    "什么是机器学习？",
                    "推荐一部电影"
                ]
            }
        ]

        # 运行测试
        total_tests = 0
        successful_tests = 0

        for test_case in test_cases:
            print(f"\n📋 测试场景: {test_case['category']}")
            print("-" * 60)

            for i, question in enumerate(test_case['questions'], 1):
                total_tests += 1
                print(f"\n🔍 测试 {i}: {question}")

                try:
                    # 执行完整的问题处理流程
                    result = self.inference_system.process_question(question)

                    # 显示处理结果
                    print(f"📊 处理结果:")
                    print(f"   问题场景: {result['scenario']}")
                    if result['question_type']:
                        print(f"   问题类型: {result['question_type']}")
                    if result['filename']:
                        print(f"   提取文件名: {result['filename']}")
                    print(f"   最终回复: {result['response']}")

                    # 显示处理步骤
                    print(f"🔄 处理步骤:")
                    for step in result['processing_steps']:
                        print(f"   {step}")

                    successful_tests += 1
                    print("✅ 测试成功")

                except Exception as e:
                    print(f"❌ 测试失败: {e}")

                print("-" * 40)

        # 显示测试总结
        print(f"\n📈 测试总结:")
        print(f"   总测试数: {total_tests}")
        print(f"   成功数: {successful_tests}")
        print(f"   失败数: {total_tests - successful_tests}")
        print(f"   成功率: {successful_tests / total_tests * 100:.1f}%")

    def run_interactive_test(self):
        """运行交互式测试"""
        print("\n" + "=" * 80)
        print("🎯 交互式推理测试")
        print("输入 'quit' 或 'exit' 退出测试")
        print("=" * 80)

        while True:
            try:
                question = input("\n请输入您的问题: ").strip()

                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 退出交互式测试")
                    break

                if not question:
                    print("⚠️ 请输入有效问题")
                    continue

                print(f"\n🔍 处理问题: {question}")
                print("-" * 50)

                # 执行完整的问题处理流程
                result = self.inference_system.process_question(question)

                # 显示处理结果
                print(f"\n📊 处理结果:")
                print(f"   问题场景: {result['scenario']}")
                if result['question_type']:
                    print(f"   问题类型: {result['question_type']}")
                if result['filename']:
                    print(f"   提取文件名: {result['filename']}")
                print(f"   最终回复: {result['response']}")

                # 显示处理步骤
                print(f"\n🔄 处理步骤:")
                for step in result['processing_steps']:
                    print(f"   {step}")

            except KeyboardInterrupt:
                print("\n👋 退出交互式测试")
                break
            except Exception as e:
                print(f"❌ 处理失败: {e}")

    def run_single_step_tests(self):
        """运行单步功能测试"""
        print("\n" + "=" * 80)
        print("🔧 单步功能测试")
        print("=" * 80)

        test_questions = [
            "数据清洗流程是什么？",
            "请帮我把实景三维模型成果数据进行治理",
            "你好",
            "今天天气怎么样？"
        ]

        for question in test_questions:
            print(f"\n🔍 测试问题: {question}")
            print("-" * 50)

            # 测试问题场景分类
            print("1️⃣ 问题场景分类:")
            scenario = self.inference_system.classify_question_scenario(question)
            print(f"   结果: {scenario}")

            # 如果是数据平台相关问题，继续测试
            if "数据平台相关问题" in scenario:
                print("2️⃣ 问题类型分类:")
                question_type = self.inference_system.classify_question_type(question)
                print(f"   结果: {question_type}")

                if "问题回答" in question_type:
                    print("3️⃣ 问题回答:")
                    answer = self.inference_system.answer_question(question)
                    print(f"   结果: {answer}")

                elif "任务处理" in question_type:
                    print("3️⃣ 文件名提取:")
                    filename = self.inference_system.extract_filename(question)
                    print(f"   结果: {filename}")

            print("=" * 50)


def main():
    """主函数"""
    print("🚀 PEFT LoRA 数据平台推理测试系统")
    print("=" * 60)

    # 模型路径配置
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    lora_model_path = r"E:\project\llm\lora\peft_lora_mixed"

    try:
        # 检查模型路径
        if not os.path.exists(bert_model_path):
            print(f"❌ BERT模型路径不存在: {bert_model_path}")
            return

        if not os.path.exists(lora_model_path):
            print(f"❌ LoRA模型路径不存在: {lora_model_path}")
            return

        print("📦 初始化推理系统...")

        # 初始化推理系统
        inference_system = DataPlatformInferenceSystem(
            model_path=lora_model_path,
            bert_model_path=bert_model_path
        )

        # 创建测试运行器
        test_runner = InferenceTestRunner(inference_system)

        print("\n🎯 选择测试模式:")
        print("1. 完整场景测试 (测试4种场景的所有用例)")
        print("2. 单步功能测试 (测试各个功能模块)")
        print("3. 交互式测试 (手动输入问题测试)")
        print("4. 退出")

        while True:
            try:
                choice = input("\n请选择测试模式 (1-4): ").strip()

                if choice == "1":
                    test_runner.run_scenario_tests()
                elif choice == "2":
                    test_runner.run_single_step_tests()
                elif choice == "3":
                    test_runner.run_interactive_test()
                elif choice == "4":
                    print("👋 退出测试系统")
                    break
                else:
                    print("⚠️ 请输入有效选项 (1-4)")

            except KeyboardInterrupt:
                print("\n👋 退出测试系统")
                break
            except Exception as e:
                print(f"❌ 运行测试时出错: {e}")

    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
