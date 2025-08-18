import logging
import os
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertBaseChinese:
    """BERT-base-chinese 模型测试类"""
    
    def __init__(self, model_path):
        """初始化模型"""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.masked_model = None
        self.load_model()
    
    def load_model(self):
        """加载模型和分词器"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            
            # 加载分词器
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            
            # 加载基础模型（用于获取词向量）
            self.model = BertModel.from_pretrained(self.model_path)
            
            # 加载掩码语言模型（用于填空任务）
            self.masked_model = BertForMaskedLM.from_pretrained(self.model_path)
            
            # 设置为评估模式
            self.model.eval()
            self.masked_model.eval()
            
            logger.info("模型加载成功!")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_word_embedding(self, text):
        """获取文本的词向量表示"""
        try:
            # 分词和编码
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的向量作为句子表示
                sentence_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            return sentence_embedding
        
        except Exception as e:
            logger.error(f"获取词向量失败: {e}")
            return None
    
    def calculate_similarity(self, text1, text2):
        """计算两个文本的相似度"""
        try:
            emb1 = self.get_word_embedding(text1)
            emb2 = self.get_word_embedding(text2)
            
            if emb1 is not None and emb2 is not None:
                similarity = cosine_similarity(emb1, emb2)[0][0]
                return similarity
            else:
                return None
                
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return None
    
    def predict_masked_word(self, text):
        """预测掩码词"""
        try:
            # 检查文本中是否包含[MASK]
            if "[MASK]" not in text:
                return "文本中没有找到[MASK]标记"
            
            # 分词和编码
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # 获取预测结果
            with torch.no_grad():
                outputs = self.masked_model(**inputs)
                predictions = outputs.logits
            
            # 找到[MASK]位置
            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
            
            if len(mask_token_index) == 0:
                return "未找到[MASK]标记"
            
            # 获取预测概率最高的词
            mask_token_logits = predictions[0, mask_token_index, :]
            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            
            # 转换为词汇
            predicted_words = []
            for token in top_tokens:
                word = self.tokenizer.decode([token])
                predicted_words.append(word)
            
            return predicted_words
            
        except Exception as e:
            logger.error(f"预测掩码词失败: {e}")
            return None
    
    def tokenize_text(self, text):
        """文本分词测试"""
        try:
            # 基础分词
            tokens = self.tokenizer.tokenize(text)
            
            # 编码
            token_ids = self.tokenizer.encode(text)
            
            # 解码
            decoded_text = self.tokenizer.decode(token_ids)
            
            return {
                "原文": text,
                "分词结果": tokens,
                "编码ID": token_ids,
                "解码结果": decoded_text
            }
            
        except Exception as e:
            logger.error(f"分词测试失败: {e}")
            return None


def show_test_samples():
    """显示测试样例"""
    print("\n" + "="*50)
    print("测试样例:")
    print("="*50)
    
    samples = {
        "1. 文本相似度测试": [
            "今天天气很好",
            "今天的天气非常不错",
            "我喜欢吃苹果",
            "苹果是我最爱的水果"
        ],
        
        "2. 掩码词预测测试": [
            "今天天气很[MASK]",
            "我喜欢吃[MASK]",
            "北京是中国的[MASK]",
            "熊猫是中国的[MASK]动物",
            "程序员需要学习[MASK]语言"
        ],
        
        "3. 文本分词测试": [
            "我爱北京天安门",
            "人工智能技术发展迅速",
            "今天是个好日子",
            "机器学习和深度学习"
        ]
    }
    
    for category, texts in samples.items():
        print(f"\n{category}:")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")


def interactive_test():
    """交互式测试"""
    # 模型路径
    bert_model_path = r"E:\project\llm\model-data\base-models\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

    # 检查模型路径是否存在
    if not os.path.exists(bert_model_path):
        logger.error(f"模型路径不存在: {bert_model_path}")
        print("请检查模型路径是否正确!")
        return
    
    try:
        # 初始化模型
        bert_tester = BertBaseChinese(bert_model_path)
        
        print("\n" + "="*60)
        print("BERT-base-chinese 模型测试工具")
        print("="*60)
        
        # 显示测试样例
        show_test_samples()
        
        while True:
            print("\n" + "-"*50)
            print("请选择测试功能:")
            print("1. 文本相似度计算")
            print("2. 掩码词预测")
            print("3. 文本分词测试")
            print("4. 词向量获取")
            print("5. 使用测试样例")
            print("0. 退出")
            
            choice = input("\n请输入选择 (0-5): ").strip()
            
            if choice == "0":
                print("退出测试程序")
                break
                
            elif choice == "1":
                print("\n文本相似度计算:")
                text1 = input("请输入第一个文本: ").strip()
                text2 = input("请输入第二个文本: ").strip()
                
                if text1 and text2:
                    similarity = bert_tester.calculate_similarity(text1, text2)
                    if similarity is not None:
                        print(f"\n相似度: {similarity:.4f}")
                        print(f"相似度百分比: {similarity * 100:.2f}%")
                    else:
                        print("计算相似度失败")
                else:
                    print("文本不能为空")
            
            elif choice == "2":
                print("\n掩码词预测:")
                text = input("请输入包含[MASK]的文本: ").strip()
                
                if text:
                    predictions = bert_tester.predict_masked_word(text)
                    if predictions and isinstance(predictions, list):
                        print(f"\n原文: {text}")
                        print("预测结果 (按概率排序):")
                        for i, word in enumerate(predictions, 1):
                            print(f"  {i}. {word}")
                    else:
                        print(f"预测失败: {predictions}")
                else:
                    print("文本不能为空")
            
            elif choice == "3":
                print("\n文本分词测试:")
                text = input("请输入要分词的文本: ").strip()
                
                if text:
                    result = bert_tester.tokenize_text(text)
                    if result:
                        print(f"\n原文: {result['原文']}")
                        print(f"分词结果: {result['分词结果']}")
                        print(f"编码ID: {result['编码ID']}")
                        print(f"解码结果: {result['解码结果']}")
                    else:
                        print("分词失败")
                else:
                    print("文本不能为空")
            
            elif choice == "4":
                print("\n词向量获取:")
                text = input("请输入文本: ").strip()
                
                if text:
                    embedding = bert_tester.get_word_embedding(text)
                    if embedding is not None:
                        print(f"\n文本: {text}")
                        print(f"词向量维度: {embedding.shape}")
                        print(f"词向量前10维: {embedding[0][:10]}")
                    else:
                        print("获取词向量失败")
                else:
                    print("文本不能为空")
            
            elif choice == "5":
                print("\n使用测试样例:")
                print("1. 测试文本相似度样例")
                print("2. 测试掩码词预测样例")
                print("3. 测试文本分词样例")
                
                sample_choice = input("请选择样例类型 (1-3): ").strip()
                
                if sample_choice == "1":
                    # 相似度测试样例
                    test_pairs = [
                        ("今天天气很好", "今天的天气非常不错"),
                        ("我喜欢吃苹果", "苹果是我最爱的水果"),
                        ("北京是中国的首都", "上海是中国的经济中心"),
                        ("人工智能很有趣", "机器学习技术发展迅速")
                    ]
                    
                    print("\n文本相似度测试结果:")
                    for text1, text2 in test_pairs:
                        similarity = bert_tester.calculate_similarity(text1, text2)
                        if similarity is not None:
                            print(f"文本1: {text1}")
                            print(f"文本2: {text2}")
                            print(f"相似度: {similarity:.4f} ({similarity * 100:.2f}%)")
                            print("-" * 40)
                
                elif sample_choice == "2":
                    # 掩码词预测样例
                    test_texts = [
                        "今天天气很[MASK]",
                        "我喜欢吃[MASK]",
                        "北京是中国的[MASK]",
                        "熊猫是中国的[MASK]动物",
                        "程序员需要学习[MASK]语言"
                    ]
                    
                    print("\n掩码词预测测试结果:")
                    for text in test_texts:
                        predictions = bert_tester.predict_masked_word(text)
                        if predictions and isinstance(predictions, list):
                            print(f"原文: {text}")
                            print(f"预测: {', '.join(predictions[:3])}")
                            print("-" * 40)
                
                elif sample_choice == "3":
                    # 分词测试样例
                    test_texts = [
                        "我爱北京天安门",
                        "人工智能技术发展迅速",
                        "今天是个好日子",
                        "机器学习和深度学习"
                    ]
                    
                    print("\n文本分词测试结果:")
                    for text in test_texts:
                        result = bert_tester.tokenize_text(text)
                        if result:
                            print(f"原文: {result['原文']}")
                            print(f"分词: {result['分词结果']}")
                            print("-" * 40)
                
                else:
                    print("无效选择")
            
            else:
                print("无效选择，请输入 0-5")
    
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        print(f"程序运行出错: {e}")


if __name__ == "__main__":
    logger.info("启动 BERT-base-chinese 交互式测试...")
    interactive_test()
