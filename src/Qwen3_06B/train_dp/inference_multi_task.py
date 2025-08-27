import json
import os
import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTaskInference:
    def __init__(self, base_model_path: str, peft_model_path: str, device: str = "auto"):
        """
        多任务推理类
        
        Args:
            base_model_path: 基础模型路径
            peft_model_path: 训练后的LoRA模型路径
            device: 设备选择 ("auto", "cuda", "cpu")
        """
        self.base_model_path = base_model_path
        self.peft_model_path = peft_model_path
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        
        # 任务系统提示词
        self.system_prompts = {
            "data_extraction": "你是一个专业的数据提取助手。任务是分析用户输入的文本，提取用户描述的数据名称",
            "dp_qa": "你是一个专业的数据平台问答助手。任务是分析用户输入的问题，并提供答案给用户",
            "question_classifier": "你是一个专业的问题分类助手。任务是分析用户输入的文本，判断用户的问题类型是（数据平台相关、通用对话、无关问题）中哪一个",
            "question_type_classifier": "你是一个专业的问题类型分类助手。任务是分析用户输入的文本，判断用户的问题类型是（问题回答、任务处理）中哪一个",
            "tool_data_platform": (
                "你是数据中台项目的工具调用助手，可以调用以下函数："
                "\n- get_data_collection(data_source: str,data_type: str,time_range: str,business_platform: str)：用于数据采集工具;"
                "\n- query_data_by_filename(filename: str,query_content: str)：用于文件名查数据工具;"
                "\n- data_warehousing(source_data_path: str,target_db_type: str,target_db: str,target_table: str,order_detail:str)：用于数据入库工具;"
                "\n- data_service_publish(source_db_type: str,source_db: str,dw_sales: str,sales_summary: str,data_filter:str,service_type:str,authorization:str)：用于数据发服务工具;"
                "\n- data_quality_check(source_db_type: str,source_db: str,source_table: str,check_dimensions: str)：用于数据质检工具;"
                "\n- data_cleaning(source_data_path: str,source_data_type: str,clean_rules: str,target_save_path: str)：用于数据清洗工具;"
                "\n请根据指令和输入,选择合适的函数并按指定格式调用。\n"
                "如果需要调用函数，请使用以下格式：\n<FunctionCall>\n{\"name\":\"函数名\",\"parameters\":{\"参数名\":参数值}}\n</FunctionCall>\n"
            )
        }
        
        self.load_model()
    
    def _setup_device(self, device: str):
        """设置推理设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"使用GPU进行推理: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("使用CPU进行推理")
        else:
            device = torch.device(device)
            logger.info(f"使用指定设备进行推理: {device}")
        
        return device
    
    def load_model(self):
        """加载模型和分词器"""
        logger.info("正在加载模型和分词器...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        
        # 加载LoRA权重
        if os.path.exists(self.peft_model_path):
            logger.info(f"加载LoRA权重: {self.peft_model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.peft_model_path)
            logger.info("LoRA权重加载成功")
        else:
            logger.warning(f"LoRA权重路径不存在: {self.peft_model_path}")
            logger.info("使用基础模型进行推理")
        
        # 设置为评估模式
        self.model.eval()
        
        logger.info("模型加载完成")
    
    def _build_prompt(self, task_type: str, user_input: str, instruction: str = "") -> str:
        """构建任务特定的提示词"""
        system_prompt = self.system_prompts.get(task_type, "")
        
        if task_type == "function_calling" or task_type == "tool_data_platform":
            prompt = system_prompt + f"\nInput: {user_input}\nOutput: "
        elif task_type == "classification" or task_type == "question_classifier" or task_type == "question_type_classifier":
            prompt = system_prompt + f"\n问题: {user_input}\n分类结果: "
        elif task_type == "extraction" or task_type == "data_extraction":
            prompt = system_prompt + f"\n输入文本: {user_input}\n提取的数据名称: "
        elif task_type == "qa" or task_type == "dp_qa":
            if instruction:
                prompt = system_prompt + f"\nInstruction: {instruction}\nInput: {user_input}\nOutput: "
            else:
                prompt = system_prompt + f"\n问题: {user_input}\n回答: "
        else:
            prompt = system_prompt + f"\nInput: {user_input}\nOutput: "
        
        return prompt
    
    def generate_response(
        self, 
        user_input: str, 
        task_type: str = "dp_qa",
        instruction: str = "",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        生成响应
        
        Args:
            user_input: 用户输入
            task_type: 任务类型 (dp_qa, data_extraction, question_classifier, etc.)
            instruction: 额外指令
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top_p参数
            do_sample: 是否采样
            
        Returns:
            生成的响应文本
        """
        # 构建提示词
        prompt = self._build_prompt(task_type, user_input, instruction)
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs["input_ids"][0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
        
        # 解码响应
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def batch_inference(self, inputs: List[Dict]) -> List[str]:
        """
        批量推理
        
        Args:
            inputs: 输入列表，每个元素包含 {"input": str, "task_type": str, "instruction": str}
            
        Returns:
            响应列表
        """
        results = []
        for item in inputs:
            user_input = item.get("input", "")
            task_type = item.get("task_type", "dp_qa")
            instruction = item.get("instruction", "")
            
            response = self.generate_response(
                user_input=user_input,
                task_type=task_type,
                instruction=instruction
            )
            results.append(response)
        
        return results
    
    def data_extraction(self, text: str) -> str:
        """数据提取任务"""
        return self.generate_response(
            user_input=text,
            task_type="data_extraction",
            temperature=0.3,
            do_sample=False
        )
    
    def question_answer(self, question: str, instruction: str = "") -> str:
        """问答任务"""
        return self.generate_response(
            user_input=question,
            task_type="dp_qa",
            instruction=instruction,
            temperature=0.7
        )
    
    def question_classify(self, question: str) -> str:
        """问题分类任务"""
        return self.generate_response(
            user_input=question,
            task_type="question_classifier",
            temperature=0.1,
            do_sample=False
        )
    
    def question_type_classify(self, question: str) -> str:
        """问题类型分类任务"""
        return self.generate_response(
            user_input=question,
            task_type="question_type_classifier",
            temperature=0.1,
            do_sample=False
        )
    
    def tool_calling(self, instruction: str) -> str:
        """工具调用任务"""
        return self.generate_response(
            user_input=instruction,
            task_type="tool_data_platform",
            temperature=0.3,
            do_sample=False
        )


def main():
    """示例使用"""
    # 配置路径
    base_model_path = "/home/user/models/Qwen3-0.6B"
    peft_model_path = "/home/user/train-models/Qwen3-multi-task"
    
    # 初始化推理器
    logger.info("初始化多任务推理器...")
    inferencer = MultiTaskInference(
        base_model_path=base_model_path,
        peft_model_path=peft_model_path,
        device="auto"
    )
    
    # 示例测试
    test_cases = [
        {
            "task": "数据提取",
            "input": "我需要获取销售数据和用户行为数据",
            "method": "data_extraction"
        },
        {
            "task": "问答",
            "input": "数据平台如何进行数据质量检查？",
            "method": "question_answer"
        },
        {
            "task": "问题分类",
            "input": "今天天气怎么样？",
            "method": "question_classify"
        },
        {
            "task": "问题类型分类",
            "input": "帮我查询一下用户数据",
            "method": "question_type_classify"
        },
        {
            "task": "工具调用",
            "input": "我需要采集电商平台的销售数据，时间范围是最近一个月",
            "method": "tool_calling"
        }
    ]
    
    print("\n=== 多任务推理测试 ===")
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['task']}任务:")
        print(f"输入: {case['input']}")
        
        # 调用对应方法
        method = getattr(inferencer, case['method'])
        response = method(case['input'])
        
        print(f"输出: {response}")
        print("-" * 50)


if __name__ == "__main__":
    main()
