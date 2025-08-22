import json

# 单工具调用样本
single_tool_sample = {
    "instruction": "你是一个工具调用助手。当用户需要查询信息时，请调用合适的工具。",
    "input": "查询上海今日气温",
    "output": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Shanghai\", \"date\": \"today\"}}</tool_call>"
}

# 多工具并行调用样本
parallel_tools_sample = {
    "instruction": "你是一个工具调用助手。可以同时调用多个工具解决复杂请求。",
    "input": "同时获取上海天气和人民币兑美元汇率",
    "output": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Shanghai\"}}</tool_call>\n"
              "<tool_call>{\"name\": \"get_currency_rate\", \"arguments\": {\"from\": \"CNY\", \"to\": \"USD\"}}</tool_call>"
}

# 多工具串行调用样本（含思考链）
sequential_tools_sample = {
    "instruction": "你是一个工具调用助手。需要分步骤解决复杂请求时，请先思考再逐步调用工具。",
    "input": "帮我预订下周一上海飞北京的航班，并提醒我当日天气",
    "output": "<thinking>1. 需先查询航班（工具：flight_search）\n"
              "2. 根据航班日期查询天气（工具：get_weather）\n"
              "3. 生成提醒消息（工具：send_reminder）</thinking>\n"
              "<tool_call>{\"name\": \"flight_search\", \"arguments\": {\"departure\": \"Shanghai\", \"arrival\": \"Beijing\", \"date\": \"next Monday\"}}</tool_call>"
}

# 错误修正样本（工具调用失败后的修正）
error_correction_sample = {
    "instruction": "你是一个工具调用助手。当工具调用失败时，请修正参数重新调用。",
    "input": "查询深圳天气（上次调用缺少日期参数）",
    "output": "<thinking>上次调用get_weather失败，缺少日期参数</thinking>\n"
              "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Shenzhen\", \"date\": \"today\"}}</tool_call>"
}

# 创建训练集 (20个样本)
train_samples = []
for i in range(5):
    train_samples.extend([
        {**single_tool_sample, "input": f"查询北京{i + 1}号气温",
         "output": single_tool_sample["output"].replace("Shanghai", "Beijing")},
        {**parallel_tools_sample, "input": f"获取北京天气和日元汇率{i}",
         "output": parallel_tools_sample["output"].replace("Shanghai", "Beijing").replace("USD", "JPY")},
        {**sequential_tools_sample, "input": f"预订{i + 10}号上海飞广州的机票和天气"},
        {**error_correction_sample, "input": f"查询杭州天气（第{i}次修正）"}
    ])

# 创建验证集 (5个样本)
val_samples = [
    {**single_tool_sample, "input": "查询纽约当前温度",
     "output": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"New York\", \"date\": \"today\"}}</tool_call>"},
    {**parallel_tools_sample, "input": "获取伦敦天气和欧元汇率",
     "output": parallel_tools_sample["output"].replace("Shanghai", "London").replace("USD", "EUR")},
    {**sequential_tools_sample, "input": "预订明天巴黎到柏林的火车和天气提醒"},
    {**error_correction_sample, "input": "查询东京天气（时区问题修正）"},
    {
        "instruction": "多工具混合调用",
        "input": "查天气、汇率并翻译'你好'到法语",
        "output": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}</tool_call>\n"
                  "<tool_call>{\"name\": \"get_currency_rate\", \"arguments\": {\"from\": \"EUR\", \"to\": \"CNY\"}}</tool_call>\n"
                  "<tool_call>{\"name\": \"translate_text\", \"arguments\": {\"text\": \"你好\", \"target_lang\": \"fr\"}}</tool_call>"
    }
]


# 保存到JSONL文件
def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("[")
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "," + '\n')
        f.write("]")


save_to_jsonl(train_samples, "tool_train.json")
save_to_jsonl(val_samples, "tool_val.json")

print("数据集生成完成！")
print("训练集样本数:", len(train_samples))
print("验证集样本数:", len(val_samples))
