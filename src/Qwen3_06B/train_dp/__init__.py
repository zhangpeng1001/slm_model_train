"""
数据中台训练器
"""

"""

我当前的机器环境是CPU，没有英伟达显卡，请写代码的时候注意参数配置需要适应CPU环境。
src/Qwen3_06B/train_dp/train_all.py文件中的代码是使用Qwen3_06B模型，进行function calling 微调的代码。
请重构一下这个代码，我现在需要微调的目标有5个，分别是：问题分类、问题类型分类、数据平台问答、数据提取、工具function calling 调用；

src/dataset/data_platform文件夹下已经构造了5个任务的数据集，分别是：data_extraction.json、dp_qa.json、question_classifier.json、question_type_classifier.json、tool_data_platform.json;
data_extraction.json 是 数据提取 数据集、dp_qa.json 是数据平台问答 数据集、question_classifier.json 是 问题分类 数据集、question_type_classifier.json 是 问题类型分类 数据集、tool_data_platform.json 是 工具调用 数据集;

每个数据集的system_prompt都不一样。具体如下：
data_extraction.json 是 数据提取 数据集、
    data_extraction_system_prompt = "你是一个专业的数据提取助手。任务是分析用户输入的文本，提取用户描述的数据名称"

dp_qa.json 是 数据平台问答 数据集、
    dp_qa_system_prompt = "你是一个专业的数据平台问答助手。任务是分析用户输入的问题，并提供答案给用户"

question_classifier.json 是 问题分类 数据集、
    question_classifier_system_prompt = "你是一个专业的问题分类助手。任务是分析用户输入的文本，判断用户的问题类型是（数据平台相关、通用对话、无关问题）中哪一个"

question_type_classifier.json 是 问题类型分类 数据集、
    question_type_classifier_system_prompt = "你是一个专业的问题类型分类助手。任务是分析用户输入的文本，判断用户的问题类型是（问题回答、任务处理）中哪一个"

tool_data_platform.json 是 工具调用 数据集;
    tool_data_platform_system_prompt = "你是数据中台项目的工具调用助手，可以调用以下函数："
    tool_data_platform_system_prompt += "\n- get_data_collection(data_source: str,data_type: str,time_range: str,business_platform: str)：用于数据采集工具;"
    tool_data_platform_system_prompt += "\n- query_data_by_filename(filename: str,query_content: st)：用于文件名查数据工具;"
    tool_data_platform_system_prompt += "\n- data_warehousing(source_data_path: str,target_db_type: str,target_db: str,target_table: str,order_detail:str)：用于数据入库工具;"
    tool_data_platform_system_prompt += "\n- data_service_publish(source_db_type: str,source_db: str,dw_sales: str,sales_summary: str,data_filter:str,service_type:str,authorization:str)：用于数据发服务工具;"
    tool_data_platform_system_prompt += "\n- data_quality_check(source_db_type: str,source_db: str,source_table: str,check_dimensions: str)：用于数据质检工具;"
    tool_data_platform_system_prompt += "\n- data_cleaning(source_data_path: str,source_data_type: str,clean_rules: str,target_save_path: str)：用于数据清洗工具;"
    tool_data_platform_system_prompt += "\n请根据指令和输入,选择合适的函数并按指定格式调用。\n"
    tool_data_platform_system_prompt +=  "如果需要调用函数，请使用以下格式：\n<FunctionCall>\n{\"name\":\"函数名\",\"parameters\":{\"参数名\":参数值}}\n</FunctionCall>\n"



"""
