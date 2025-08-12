# bert 编码

from transformers import BertTokenizer

# 加载字典和分词工具
model_name = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

tokenizer = BertTokenizer.from_pretrained(model_name)

# 获取字典
vocab = tokenizer.get_vocab()
# print(vocab)
print(len(vocab))  # 21128
# 每个字符在字典中是唯一的
print("2" in vocab)

print("样" in vocab)
# 中文是以单个字在字典中存在
print("阳光" in vocab)

# 添加新词
tokenizer.add_tokens(new_tokens=["阳光", "大地"])

# 添加新的特殊符号
tokenizer.add_special_tokens({"eos_token": "[EOS]"})

# 需要重新获取，刷新字典
vocab = tokenizer.get_vocab()

print(len(vocab))
print("阳光" in vocab)
# print(vocab)
# print(tokenizer)
print("阳光" in vocab, "大地" in vocab, "[EOS]" in vocab)

# 编码句子
out = tokenizer.encode(
    text="阳光照在大地上[EOS]", text_pair=None, truncation=True,
    padding="max_length", max_length=20, add_special_tokens=True,
    return_tensors=None
)
print(out)
# 解码为原字符串
print(tokenizer.decode(out))
