"""
问题分类器控制器
用于判断用户问题的类型：数据平台相关、通用对话、无关问题
"""
import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionClassifier:
    """问题分类器"""
    
    def __init__(self, model_path=None):
        """
        初始化分类器
        
        Args:
            model_path: 训练好的分类器模型路径
        """
        self.device = torch.device("cpu")
        self.model_path = model_path
        
        # 类别映射
        self.label_map = {
            0: "data_platform",    # 数据平台相关
            1: "general_chat",     # 通用对话
            2: "irrelevant"        # 无关问题
        }
        
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载分类器模型"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # 加载训练好的分类器
                logger.info(f"加载分类器模型: {self.model_path}")
                
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model = BertForSequenceClassification.from_pretrained(self.model_path)
                
                # 加载标签映射
                label_map_path = os.path.join(self.model_path, 'label_map.json')
                if os.path.exists(label_map_path):
                    with open(label_map_path, 'r', encoding='utf-8') as f:
                        loaded_label_map = json.load(f)
                    # 转换键为整数
                    self.label_map = {int(k): v for k, v in loaded_label_map.items()}
                    self.reverse_label_map = {v: k for k, v in self.label_map.items()}
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("✓ 分类器模型加载成功")
                self.is_model_loaded = True
                
            else:
                # 没有训练好的模型，使用规则分类器
                logger.info("未找到训练好的分类器，使用规则分类器")
                self.model = None
                self.tokenizer = None
                self.is_model_loaded = False
                
        except Exception as e:
            logger.error(f"分类器模型加载失败: {e}")
            logger.info("回退到规则分类器")
            self.model = None
            self.tokenizer = None
            self.is_model_loaded = False
    
    def classify_question(self, question):
        """
        分类问题
        
        Args:
            question: 用户问题
            
        Returns:
            dict: 包含分类结果和置信度的字典
        """
        if self.is_model_loaded:
            return self._classify_with_model(question)
        else:
            return self._classify_with_rules(question)
    
    def _classify_with_model(self, question):
        """使用训练好的模型进行分类"""
        try:
            # 编码问题
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 获取预测结果和置信度
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            category = self.label_map[predicted_class]
            
            return {
                "category": category,
                "confidence": confidence,
                "method": "bert_model",
                "predicted_class": predicted_class
            }
            
        except Exception as e:
            logger.error(f"模型分类失败: {e}")
            # 回退到规则分类
            return self._classify_with_rules(question)
    
    def _classify_with_rules(self, question):
        """使用规则进行分类"""
        question_lower = question.lower().strip()
        
        # 数据平台相关关键词
        data_platform_keywords = [
            "数据", "清洗", "入库", "质量", "监控", "安全", "备份",
            "预处理", "转换", "标准化", "权限", "调度", "异常",
            "数据库", "连接", "处理", "流程", "etl", "仓库",
            "平台", "架构", "治理", "血缘", "元数据"
        ]
        
        # 通用对话关键词
        general_chat_keywords = [
            "你好", "您好", "hi", "hello", "早上好", "下午好", "晚上好",
            "谢谢", "感谢", "不客气", "再见", "拜拜", "bye",
            "好的", "知道了", "明白了", "收到", "ok", "可以", "没问题",
            "帮忙", "请问", "能否", "麻烦", "打扰", "不好意思", "对不起", "抱歉"
        ]
        
        # 检查数据平台相关
        for keyword in data_platform_keywords:
            if keyword in question_lower:
                return {
                    "category": "data_platform",
                    "confidence": 0.9,
                    "method": "rule_based",
                    "matched_keyword": keyword
                }
        
        # 检查通用对话
        for keyword in general_chat_keywords:
            if keyword in question_lower:
                return {
                    "category": "general_chat",
                    "confidence": 0.95,
                    "method": "rule_based",
                    "matched_keyword": keyword
                }
        
        # 默认为无关问题
        return {
            "category": "irrelevant",
            "confidence": 0.8,
            "method": "rule_based",
            "reason": "no_matching_keywords"
        }
    
    def is_data_platform_related(self, question):
        """
        判断问题是否与数据平台相关
        
        Args:
            question: 用户问题
            
        Returns:
            bool: 是否与数据平台相关
        """
        result = self.classify_question(question)
        return result["category"] == "data_platform"
    
    def is_general_chat(self, question):
        """
        判断问题是否为通用对话
        
        Args:
            question: 用户问题
            
        Returns:
            bool: 是否为通用对话
        """
        result = self.classify_question(question)
        return result["category"] == "general_chat"
    
    def is_irrelevant(self, question):
        """
        判断问题是否为无关问题
        
        Args:
            question: 用户问题
            
        Returns:
            bool: 是否为无关问题
        """
        result = self.classify_question(question)
        return result["category"] == "irrelevant"
    
    def get_classification_info(self):
        """获取分类器信息"""
        return {
            "model_path": self.model_path,
            "is_model_loaded": self.is_model_loaded,
            "device": str(self.device),
            "categories": list(self.label_map.values())
        }
