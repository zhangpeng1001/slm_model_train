"""
LoRA指令微调数据集
"""

# 问题场景分类数据集
question_classifier = [
    # 数据平台相关问题
    {
        "instruction": "问题场景分类",
        "input": "数据清洗流程是什么？",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "如何进行数据清洗？",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据清洗步骤有哪些？",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据入库流程",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "如何进行数据入库？",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据质量检查",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "如何检查数据质量？",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据监控怎么做？",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据安全措施",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据预处理流程",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "ETL流程是什么？",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据仓库架构",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据治理方案",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "元数据管理",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据血缘关系",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "任务调度配置",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "异常处理机制",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "权限管理设置",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据备份策略",
        "output": "数据平台相关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "数据库连接配置",
        "output": "数据平台相关问题"
    },
    
    # 通用问题
    {
        "instruction": "问题场景分类",
        "input": "你好",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "您好",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "早上好",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "下午好",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "谢谢",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "感谢",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "再见",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "好的",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "知道了",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "明白了",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "请问",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "麻烦",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "不好意思",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "对不起",
        "output": "通用问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "抱歉",
        "output": "通用问题"
    },
    
    # 无关问题
    {
        "instruction": "问题场景分类",
        "input": "今天天气怎么样？",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "北京有什么好吃的？",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "如何学习英语？",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "Python怎么安装？",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "什么是机器学习？",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "股票今天涨了吗？",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "电影推荐一下",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "音乐好听吗？",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "游戏攻略",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "旅游景点推荐",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "美食制作方法",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "健身计划",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "减肥方法",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "护肤品推荐",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "汽车保养",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "房价走势",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "教育政策",
        "output": "无关问题"
    },
    {
        "instruction": "问题场景分类",
        "input": "医疗保险",
        "output": "无关问题"
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
        "input": "DEM高程数据怎么进行质量检查",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "矢量地形图数据如何入库",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "我的点云数据处理完了，如何发布服务",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "地理信息数据库的数据怎么备份",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "卫星遥感数据的预处理流程是什么",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "无人机航拍数据如何进行几何纠正",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "土地利用现状数据怎么更新",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "基础地理信息数据如何管理",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "遥感影像数据的配准方法有哪些",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "如何选择合适的坐标系统",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "数据质量检查的标准是什么",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "什么是数据血缘关系",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "元数据管理的最佳实践",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "空间数据索引如何建立",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "GIS数据格式转换工具推荐",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "地理编码的准确性如何提高",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "空间分析的常用方法",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "地图投影变换的原理",
        "output": "问题回答"
    },
    {
        "instruction": "问题类型分类",
        "input": "栅格数据重采样方法",
        "output": "问题回答"
    },
    
    # 任务处理类型
    {
        "instruction": "问题类型分类",
        "input": "请帮我把实景三维模型成果数据进行治理",
        "output": "任务处理"
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
    },
    {
        "instruction": "问题类型分类",
        "input": "激光雷达数据进行格式转换",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "地籍调查数据需要标准化处理",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "处理地下管线探测数据",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "倾斜摄影测量数据进行建模",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "帮我清洗这批地形数据",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "需要对影像数据进行几何校正",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "请处理这些GPS轨迹数据",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "批量转换CAD文件格式",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "执行数据质量检查任务",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "启动数据同步作业",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "运行地理编码程序",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "执行空间分析计算",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "进行数据备份操作",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "开始数据迁移任务",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "处理异常数据记录",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "执行数据更新操作",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "运行ETL数据处理流程",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "启动定时调度任务",
        "output": "任务处理"
    },
    {
        "instruction": "问题类型分类",
        "input": "执行数据压缩操作",
        "output": "任务处理"
    }
]

# 问题回答数据集
question_answer = [
    {
        "instruction": "问题回答",
        "input": "数据清洗流程",
        "output": "数据清洗流程需要先配置清洗规则，再配置清洗方案，最后针对要清洗的数据启动任务即可。具体步骤：1.定义数据质量规则 2.配置清洗策略 3.执行清洗任务 4.验证清洗结果"
    },
    {
        "instruction": "问题回答",
        "input": "数据清洗规则",
        "output": "数据清洗规则包括：空值处理规则、重复数据处理规则、数据格式规范化规则、异常值检测规则、数据一致性校验规则等"
    },
    {
        "instruction": "问题回答",
        "input": "数据质量检查",
        "output": "数据质量检查包括完整性检查、准确性检查、一致性检查、时效性检查、唯一性检查等维度，通过设置质量阈值来判断数据是否符合要求"
    },
    {
        "instruction": "问题回答",
        "input": "数据入库流程",
        "output": "数据入库流程包括：1.数据格式验证 2.数据质量检查 3.数据转换处理 4.目标表结构匹配 5.执行入库操作 6.入库结果验证"
    },
    {
        "instruction": "问题回答",
        "input": "数据入库策略",
        "output": "数据入库策略包括全量入库、增量入库、实时入库等方式。全量入库适用于历史数据迁移，增量入库适用于定期数据更新，实时入库适用于流式数据处理"
    },
    {
        "instruction": "问题回答",
        "input": "数据库连接配置",
        "output": "数据库连接需要配置数据库类型、连接地址、端口、用户名、密码、数据库名等参数，建议使用连接池管理数据库连接以提高性能"
    },
    {
        "instruction": "问题回答",
        "input": "数据预处理",
        "output": "数据预处理包括数据清洗、数据转换、数据标准化、特征工程等步骤，目的是将原始数据转换为适合分析和建模的格式"
    },
    {
        "instruction": "问题回答",
        "input": "数据转换",
        "output": "数据转换包括格式转换、编码转换、数据类型转换、字段映射等操作，确保数据在不同系统间的兼容性"
    },
    {
        "instruction": "问题回答",
        "input": "数据标准化",
        "output": "数据标准化是将数据按照统一的标准和规范进行处理，包括命名规范、格式规范、编码规范等"
    },
    {
        "instruction": "问题回答",
        "input": "数据监控",
        "output": "数据监控包括数据质量监控、数据处理任务监控、数据流量监控等，通过设置监控指标和告警规则来保障数据平台的稳定运行"
    },
    {
        "instruction": "问题回答",
        "input": "任务调度",
        "output": "任务调度系统负责管理数据处理任务的执行时间、依赖关系、资源分配等，支持定时调度、事件触发调度等方式"
    },
    {
        "instruction": "问题回答",
        "input": "异常处理",
        "output": "异常处理机制包括任务失败重试、错误日志记录、异常告警通知等，确保数据处理过程的可靠性"
    },
    {
        "instruction": "问题回答",
        "input": "数据安全",
        "output": "数据安全包括数据加密、访问控制、审计日志、数据脱敏等措施，保护敏感数据不被非法访问和使用"
    },
    {
        "instruction": "问题回答",
        "input": "权限管理",
        "output": "权限管理通过角色和权限的分配，控制用户对数据和功能的访问范围，实现最小权限原则"
    },
    {
        "instruction": "问题回答",
        "input": "数据备份",
        "output": "数据备份策略包括定期备份、增量备份、异地备份等，确保数据的安全性和可恢复性"
    },
    {
        "instruction": "问题回答",
        "input": "如何进行数据清洗",
        "output": "数据清洗流程需要先配置清洗规则，再配置清洗方案，最后针对要清洗的数据启动任务即可。具体步骤：1.定义数据质量规则 2.配置清洗策略 3.执行清洗任务 4.验证清洗结果"
    },
    {
        "instruction": "问题回答",
        "input": "数据清洗的流程是什么",
        "output": "数据清洗流程需要先配置清洗规则，再配置清洗方案，最后针对要清洗的数据启动任务即可。具体步骤：1.定义数据质量规则 2.配置清洗策略 3.执行清洗任务 4.验证清洗结果"
    },
    {
        "instruction": "问题回答",
        "input": "数据清洗步骤",
        "output": "数据清洗流程需要先配置清洗规则，再配置清洗方案，最后针对要清洗的数据启动任务即可。具体步骤：1.定义数据质量规则 2.配置清洗策略 3.执行清洗任务 4.验证清洗结果"
    },
    {
        "instruction": "问题回答",
        "input": "数据入库的流程",
        "output": "数据入库流程包括：1.数据格式验证 2.数据质量检查 3.数据转换处理 4.目标表结构匹配 5.执行入库操作 6.入库结果验证"
    },
    {
        "instruction": "问题回答",
        "input": "如何进行数据入库",
        "output": "数据入库流程包括：1.数据格式验证 2.数据质量检查 3.数据转换处理 4.目标表结构匹配 5.执行入库操作 6.入库结果验证"
    },
    {
        "instruction": "问题回答",
        "input": "数据入库步骤",
        "output": "数据入库流程包括：1.数据格式验证 2.数据质量检查 3.数据转换处理 4.目标表结构匹配 5.执行入库操作 6.入库结果验证"
    },
    {
        "instruction": "问题回答",
        "input": "如何检查数据质量",
        "output": "数据质量检查包括完整性检查、准确性检查、一致性检查、时效性检查、唯一性检查等维度，通过设置质量阈值来判断数据是否符合要求"
    },
    {
        "instruction": "问题回答",
        "input": "数据质量检查方法",
        "output": "数据质量检查包括完整性检查、准确性检查、一致性检查、时效性检查、唯一性检查等维度，通过设置质量阈值来判断数据是否符合要求"
    },
    {
        "instruction": "问题回答",
        "input": "数据质量如何保证",
        "output": "数据质量检查包括完整性检查、准确性检查、一致性检查、时效性检查、唯一性检查等维度，通过设置质量阈值来判断数据是否符合要求"
    },
    {
        "instruction": "问题回答",
        "input": "如何监控数据",
        "output": "数据监控包括数据质量监控、数据处理任务监控、数据流量监控等，通过设置监控指标和告警规则来保障数据平台的稳定运行"
    },
    {
        "instruction": "问题回答",
        "input": "数据监控方法",
        "output": "数据监控包括数据质量监控、数据处理任务监控、数据流量监控等，通过设置监控指标和告警规则来保障数据平台的稳定运行"
    },
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
        "input": "请帮我把实景三维模型成果数据进行治理",
        "output": "实景三维模型成果"
    },
    {
        "instruction": "文件名称提取",
        "input": "我已经上传了单波段浮点投影的数据，现在想进行入库",
        "output": "单波段浮点投影"
    },
    {
        "instruction": "文件名称提取",
        "input": "有一些遥感影像数据需要处理",
        "output": "遥感影像"
    },
    {
        "instruction": "文件名称提取",
        "input": "DEM高程数据怎么进行质量检查",
        "output": "DEM高程"
    },
    {
        "instruction": "文件名称提取",
        "input": "矢量地形图数据如何入库",
        "output": "矢量地形图"
    },
    {
        "instruction": "文件名称提取",
        "input": "我的点云数据处理完了，如何发布服务",
        "output": "点云"
    },
    {
        "instruction": "文件名称提取",
        "input": "正射影像数据需要进行坐标转换",
        "output": "正射影像"
    },
    {
        "instruction": "文件名称提取",
        "input": "地理信息数据库的数据怎么备份",
        "output": "地理信息数据库"
    },
    {
        "instruction": "文件名称提取",
        "input": "卫星遥感数据的预处理流程是什么",
        "output": "卫星遥感"
    },
    {
        "instruction": "文件名称提取",
        "input": "无人机航拍数据如何进行几何纠正",
        "output": "无人机航拍"
    },
    {
        "instruction": "文件名称提取",
        "input": "激光雷达数据的格式转换",
        "output": "激光雷达"
    },
    {
        "instruction": "文件名称提取",
        "input": "地籍调查数据需要标准化处理",
        "output": "地籍调查"
    },
    {
        "instruction": "文件名称提取",
        "input": "土地利用现状数据怎么更新",
        "output": "土地利用现状"
    },
    {
        "instruction": "文件名称提取",
        "input": "基础地理信息数据如何管理",
        "output": "基础地理信息"
    },
    {
        "instruction": "文件名称提取",
        "input": "城市三维建模数据的质量控制",
        "output": "城市三维建模"
    },
    {
        "instruction": "文件名称提取",
        "input": "地下管线探测数据处理",
        "output": "地下管线探测"
    },
    {
        "instruction": "文件名称提取",
        "input": "倾斜摄影测量数据建模",
        "output": "倾斜摄影测量"
    },
    {
        "instruction": "文件名称提取",
        "input": "地质勘探数据入库流程",
        "output": "地质勘探"
    },
    {
        "instruction": "文件名称提取",
        "input": "水文监测数据的时序分析",
        "output": "水文监测"
    },
    {
        "instruction": "文件名称提取",
        "input": "气象观测数据处理",
        "output": "气象观测"
    },
    {
        "instruction": "文件名称提取",
        "input": "土壤调查数据分析",
        "output": "土壤调查"
    },
    {
        "instruction": "文件名称提取",
        "input": "森林资源调查数据",
        "output": "森林资源调查"
    },
    {
        "instruction": "文件名称提取",
        "input": "海洋测绘数据处理",
        "output": "海洋测绘"
    },
    {
        "instruction": "文件名称提取",
        "input": "地震监测数据分析",
        "output": "地震监测"
    },
    {
        "instruction": "文件名称提取",
        "input": "环境监测数据入库",
        "output": "环境监测"
    },
    {
        "instruction": "文件名称提取",
        "input": "交通流量数据统计",
        "output": "交通流量"
    },
    {
        "instruction": "文件名称提取",
        "input": "人口普查数据处理",
        "output": "人口普查"
    },
    {
        "instruction": "文件名称提取",
        "input": "经济统计数据分析",
        "output": "经济统计"
    },
    {
        "instruction": "文件名称提取",
        "input": "建筑信息模型数据",
        "output": "建筑信息模型"
    },
    {
        "instruction": "文件名称提取",
        "input": "管网拓扑数据维护",
        "output": "管网拓扑"
    },
    {
        "instruction": "文件名称提取",
        "input": "电力设施数据更新",
        "output": "电力设施"
    },
    {
        "instruction": "文件名称提取",
        "input": "通信基站数据管理",
        "output": "通信基站"
    },
    {
        "instruction": "文件名称提取",
        "input": "道路网络数据处理",
        "output": "道路网络"
    },
    {
        "instruction": "文件名称提取",
        "input": "地名地址数据标准化",
        "output": "地名地址"
    },
    {
        "instruction": "文件名称提取",
        "input": "行政区划数据更新",
        "output": "行政区划"
    },
    {
        "instruction": "文件名称提取",
        "input": "地价监测数据分析",
        "output": "地价监测"
    },
    {
        "instruction": "文件名称提取",
        "input": "规划用地数据管理",
        "output": "规划用地"
    },
    # 添加更多变体以扩充数据集
    {
        "instruction": "文件名称提取",
        "input": "如何处理西安市地类图斑数据？",
        "output": "西安市地类图斑"
    },
    {
        "instruction": "文件名称提取",
        "input": "实景三维模型成果数据怎么入库？",
        "output": "实景三维模型成果"
    },
    {
        "instruction": "文件名称提取",
        "input": "单波段浮点投影数据的质量检查",
        "output": "单波段浮点投影"
    },
    {
        "instruction": "文件名称提取",
        "input": "遥感影像数据如何发布服务？",
        "output": "遥感影像"
    },
    {
        "instruction": "文件名称提取",
        "input": "DEM高程数据的标准化流程",
        "output": "DEM高程"
    },
    {
        "instruction": "文件名称提取",
        "input": "矢量地形图数据处理完成后如何使用？",
        "output": "矢量地形图"
    },
    {
        "instruction": "文件名称提取",
        "input": "点云数据需要进行格式转换",
        "output": "点云"
    },
    {
        "instruction": "文件名称提取",
        "input": "正射影像数据的几何校正",
        "output": "正射影像"
    },
    {
        "instruction": "文件名称提取",
        "input": "地理信息数据库数据的备份恢复",
        "output": "地理信息数据库"
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
    },
    {
        "instruction": "文件名称提取",
        "input": "激光雷达数据需要标准化处理",
        "output": "激光雷达"
    }
]
