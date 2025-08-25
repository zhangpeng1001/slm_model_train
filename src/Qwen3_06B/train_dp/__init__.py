"""
数据中台训练器
"""

"""

我当前的机器环境是CPU，没有英伟达显卡，请写代码的时候注意参数配置需要适应CPU环境。
src/Qwen3_06B/train_dp/train_all.py文件中的代码是使用Qwen3_06B模型，进行function calling 微调的代码。
请重构一下这个代码，我现在需要微调的目标有5个，分别是：问题分类、问题类型分类、数据平台问答、数据提取、工具调用；

src/dataset/data_platform文件夹下已经构造了5个任务的数据集，分别是：data_extraction.json、dp_qa.json、question_classifier.json、question_type_classifier.json、tool_data_platform.json;
data_extraction.json 是 数据提取 数据集、dp_qa.json 是数据平台问答 数据集、question_classifier.json 是 问题分类 数据集、question_type_classifier.json 是 问题类型分类 数据集、tool_data_platform.json 是 工具调用 数据集;


"""
