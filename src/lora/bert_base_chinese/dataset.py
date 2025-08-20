# 问题场景分类
question_classifier = [
    {
        "instruction": "问题场景分类",
        "input": "数据清洗流程是什么？",
        "output": "数据平台相关问题"
    }, {
        "instruction": "问题场景分类",
        "input": "你好",
        "output": "通用问题"
    }, {
        "instruction": "问题场景分类",
        "input": "今天天气怎么样？",
        "output": "无关问题"
    },
]

# 问题类型分类
question_type_classifier = [
    {
        "instruction": "问题类型分类",
        "input": "我有一批西安市地类图斑的数据，怎么进行发服务",
        "output": "问题回答"
    }, {
        "instruction": "问题类型分类",
        "input": "请帮我把实景三维模型成果数据进行治理",
        "output": "任务处理"
    },
]

# 问题回答
question_answer = [
    {
        "instruction": "问题回答",
        "input": "数据入库流程",
        "output": "数据入库流程包括：1.数据格式验证 2.数据质量检查 3.数据转换处理 4.目标表结构匹配 5.执行入库操作 6.入库结果验证"
    }, {
        "instruction": "问题回答",
        "input": "数据标准化",
        "output": "数据标准化是将数据按照统一的标准和规范进行处理，包括命名规范、格式规范、编码规范等"
    },
]
# 文件名称提取
file_name_pick = [
    {
        "instruction": "文件名称提取",
        "input": "我有一批西安市地类图斑的数据，怎么进行发服务",
        "output": "西安市地类图斑"
    }, {
        "instruction": "文件名称提取",
        "input": "请帮我把实景三维模型成果数据进行治理",
        "output": "实景三维模型成果"
    },
]
