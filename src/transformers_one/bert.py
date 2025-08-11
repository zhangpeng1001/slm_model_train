# 分类模型
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# 加载模型和分词器
# model_name = "bert-base-chinese"
model_name = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 创建分类pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 进行分类
result = classifier("你好，我是一位大侠")

print(result)
