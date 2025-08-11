# 调用本地大模型
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# model_dir = r"./model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"

# 使用本地已经下载的模型
model_dir = r"E:\project\python\slm_model_train\src\transformers_one\model\uer\gpt2-chinese-cluecorpussmall\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 使用pipeline调用模型
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")  # device="cpu" 可以不写
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer,device="cuda")#有GPU环境，可以使用GPU

# 生成文本,max_length控制最大的输出是50个字，num_return_sequences返回的是几段话
# output = generator("你好，我是一位大侠",max_length=50,num_return_sequences=1) #提示错误，修改一下
# output = generator("你好，我是一位大侠", max_new_tokens=100, num_return_sequences=1, truncation=True,
#                    padding='max_length')

# output = generator("你好，我是一位大侠",
#                    max_length=50,
#                    num_return_sequences=1,  # 返回的是1句话
#                    truncation=True,  # 是否截断文本，以适应模型最大输入
#                    temperature=0.7,  # 文本生成的随机性
#                    top_k=50,  # 从模型生成的概率最高的50词，选择下一个词，模型生成时，只考虑概率最高的前50个词
#                    top_p=0.9,  # 核采样，模型会在可能性90%的词中选择下一个词，这个是生成质量控制
#                    clean_up_tokenization_spaces=False)  # 是否保留格式


output = generator("你好，我是一款语言模型",
                   max_new_tokens=100,
                   num_return_sequences=2,  # 返回的是2句话
                   truncation=True,  # 是否截断文本，以适应模型最大输入
                   temperature=0.7,  # 文本生成的随机性
                   top_k=50,  # 从模型生成的概率最高的50词，选择下一个词，模型生成时，只考虑概率最高的前50个词
                   top_p=0.9,  # 核采样，模型会在可能性90%的词中选择下一个词，这个是生成质量控制
                   clean_up_tokenization_spaces=False)  # 是否保留格式
print(output)

# 模型生成文字的原理：模型根据前面输入的提示词，从23328个字中选一个，模型每次从概率前50个字符，从这50个中随机选一个。如果top_k设置为1，每次的答案是固定的
"""
GPT2模型：

现在的模型，都是基于transformer构建

transformers 是由编码器 + 解码器 构成

编码器[bert - 最基础] : 分类模型，主要是文本的特征提取，用于分类的任务

解码器[GPT  - 最基础] ：生成模型，输入向量/文本的特征信息  ， 输出文本数据

"""