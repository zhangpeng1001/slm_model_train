# 将模型下载到本地调用
# AutoTokenizer 分词工具，词向量工具
from transformers import AutoModelForCausalLM, AutoTokenizer

# 将模型和分词工具：下载到本地，并指定保存路劲
model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "model/uer/gpt2-chinese-cluecorpussmall"

# 下载模型
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
# 下载分词工具
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

#总共800M+
print(f"模型分词器已下载到：{cache_dir}")

#模型是放在这里：E:\project\python\slm_model_train\src\transformers_one\model\uer\gpt2-chinese-cluecorpussmall\models--uer--gpt2-chinese-cluecorpussmall\snapshots
#pytorch_model.bin 是模型，有401M，config是模型的配置文件，其中的"tokenizer_class": "BertTokenizer",说明了当前模型使用的分词器，"vocab_size": 21128 字符量
#把所有的字，都放在vocab.txt中，将词放在文件中，进行位置编码，对应的字翻译成对应的位置数字
