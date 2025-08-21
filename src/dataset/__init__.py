"""
LoRA指令微调数据集
"""

# 问题场景分类数据集
question_classifier = [
    {
        "instruction": "问题场景分类",
        "input": "数据清洗流程是什么？",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据库连接配置",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "抱歉",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "法律咨询",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "心理健康",
        "output": "无关问题"
    }
]

# 问题类型分类数据集
question_type_classifier = [
    # 问题回答类型
    {
        "instruction": "问题类型分类",
        "input": "我有一批西安市地类图斑的数据，怎么进行发服务",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "我已经上传了单波段浮点投影的数据，现在想进行入库",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "有一些遥感影像数据需要处理",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "正射影像数据需要进行坐标转换",
        "output": "任务处理"
    }
]

# 问题回答数据集
question_answer = [
    {
        "instruction": "问题回答",
        "input": "如何保证数据安全",
        "output": "数据安全包括数据加密、访问控制、审计日志、数据脱敏等措施，保护敏感数据不被非法访问和使用"
    },
    {
        "instruction": "问题回答",
        "input": "数据安全措施",
        "output": "数据安全包括数据加密、访问控制、审计日志、数据脱敏等措施，保护敏感数据不被非法访问和使用"
    },
    {
        "instruction": "问题回答",
        "input": "数据安全怎么做",
        "output": "数据安全包括数据加密、访问控制、审计日志、数据脱敏等措施，保护敏感数据不被非法访问和使用"
    }
]

# 文件名称提取数据集
file_name_pick = [
    {
        "instruction": "文件名称提取",
        "input": "我有一批西安市地类图斑的数据，怎么进行发服务",
        "output": "西安市地类图斑"
    },
    {
        "instruction": "文件名称提取",
        "input": "卫星遥感数据怎么进行预处理？",
        "output": "卫星遥感"
    },
    {
        "instruction": "文件名称提取",
        "input": "无人机航拍数据的质量控制",
        "output": "无人机航拍"
    }
]
