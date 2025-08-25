import json

# 数据平台相关问题（类别0）
data_platform_questions = [
    "数据清洗流程是什么？",
    "如何进行数据清洗？",
    "数据清洗步骤有哪些？",
    "数据清洗的流程",
    "怎么清洗数据？",
    "数据入库流程",
    "如何进行数据入库？",
    "数据入库的步骤",
    "怎样入库数据？",
    "数据入库操作",
    "数据质量检查",
    "如何检查数据质量？",
    "数据质量检查方法",
    "数据质量如何保证？",
    "质量检查步骤",
    "数据监控",
    "如何监控数据？",
    "数据监控方法",
    "数据监控怎么做？",
    "监控数据流程",
    "数据安全",
    "如何保证数据安全？",
    "数据安全措施",
    "数据安全怎么做？",
    "数据加密方法",
    "数据预处理",
    "数据转换",
    "数据标准化",
    "数据备份",
    "权限管理",
    "任务调度",
    "异常处理",
    "数据库连接",
    "数据处理流程",
    "ETL流程",
    "数据仓库",
    "数据平台架构",
    "数据治理",
    "数据血缘",
    "元数据管理"
]

# 通用对话（类别1）
general_chat_questions = [
    "你好",
    "您好",
    "hi",
    "hello",
    "早上好",
    "下午好",
    "晚上好",
    "谢谢",
    "谢谢你",
    "感谢",
    "不客气",
    "再见",
    "拜拜",
    "bye",
    "好的",
    "知道了",
    "明白了",
    "收到",
    "OK",
    "可以",
    "没问题",
    "帮忙",
    "请问",
    "能否",
    "麻烦",
    "打扰了",
    "不好意思",
    "对不起",
    "抱歉"
]

# 无关问题（类别2）
irrelevant_questions = [
    "今天天气怎么样？",
    "北京有什么好吃的？",
    "如何学习英语？",
    "Python怎么安装？",
    "什么是机器学习？",
    "股票今天涨了吗？",
    "电影推荐一下",
    "音乐好听吗？",
    "游戏攻略",
    "旅游景点推荐",
    "美食制作方法",
    "健身计划",
    "减肥方法",
    "护肤品推荐",
    "汽车保养",
    "房价走势",
    "教育政策",
    "医疗保险",
    "法律咨询",
    "心理健康",
    "what is your name?",
    "how are you?",
    "where are you from?",
    "tell me a joke",
    "sing a song",
    "write a poem",
    "calculate 1+1",
    "what time is it?",
    "random text here",
    "asdfghjkl",
    "123456789",
    "测试文本",
    "随机输入",
    "无意义内容",
    "乱七八糟",
    "胡言乱语",
    "不知所云",
    "莫名其妙",
    "奇怪问题",
    "无关内容"
]


def prepare_training_data():
    """准备训练数据"""

    data_list = []

    for text in data_platform_questions:
        data_list.append({"input": text, "output": "数据平台相关"})

    for text in general_chat_questions:
        data_list.append({"input": text, "output": "通用对话"})

    for text in irrelevant_questions:
        data_list.append({"input": text, "output": "无关问题"})

    print(f"准备了 {len(data_list)} 个训练样本")
    return data_list


print(json.dumps(prepare_training_data(), ensure_ascii=False, indent=2))
