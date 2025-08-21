"""
Function Calling 微调数据集
用于训练Qwen3-0.6B模型的函数调用能力
"""

# Function Calling 数据集
function_calling_dataset = [
    # 天气查询相关
    {
        "messages": [
            {"role": "user", "content": "今天北京的天气怎么样？"},
            {"role": "assistant", "content": "我来帮您查询北京今天的天气情况。", "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"北京\", \"date\": \"today\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定地点的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "date": {
                                "type": "string",
                                "description": "日期，可以是today、tomorrow或具体日期"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    },
    
    {
        "messages": [
            {"role": "user", "content": "明天上海会下雨吗？"},
            {"role": "assistant", "content": "让我查询一下明天上海的天气预报。", "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"上海\", \"date\": \"tomorrow\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定地点的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "date": {
                                "type": "string",
                                "description": "日期，可以是today、tomorrow或具体日期"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    },

    # 时间查询相关
    {
        "messages": [
            {"role": "user", "content": "现在几点了？"},
            {"role": "assistant", "content": "我来为您查询当前时间。", "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "arguments": "{}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前时间",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "纽约现在是什么时间？"},
            {"role": "assistant", "content": "我来查询纽约当前的时间。", "tool_calls": [
                {
                    "id": "call_4",
                    "type": "function",
                    "function": {
                        "name": "get_timezone_time",
                        "arguments": "{\"timezone\": \"America/New_York\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_timezone_time",
                    "description": "获取指定时区的当前时间",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "时区名称，如Asia/Shanghai、America/New_York等"
                            }
                        },
                        "required": ["timezone"]
                    }
                }
            }
        ]
    },

    # 计算相关
    {
        "messages": [
            {"role": "user", "content": "帮我计算一下 123 + 456 等于多少？"},
            {"role": "assistant", "content": "我来帮您计算这个加法。", "tool_calls": [
                {
                    "id": "call_5",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": "{\"expression\": \"123 + 456\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式，如'2+3*4'、'sqrt(16)'等"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "计算圆周率乘以5的平方"},
            {"role": "assistant", "content": "我来计算π × 5²的结果。", "tool_calls": [
                {
                    "id": "call_6",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": "{\"expression\": \"pi * 5^2\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式，如'2+3*4'、'sqrt(16)'等"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    },

    # 搜索相关
    {
        "messages": [
            {"role": "user", "content": "搜索一下人工智能的最新发展"},
            {"role": "assistant", "content": "我来为您搜索人工智能的最新发展信息。", "tool_calls": [
                {
                    "id": "call_7",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\": \"人工智能最新发展\", \"num_results\": 5}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "在网络上搜索信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "返回结果数量",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    },

    # 文件操作相关
    {
        "messages": [
            {"role": "user", "content": "帮我读取文件 data.txt 的内容"},
            {"role": "assistant", "content": "我来帮您读取指定文件的内容。", "tool_calls": [
                {
                    "id": "call_8",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"file_path\": \"data.txt\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取文件内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "文件路径"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]
    },

    # 邮件发送相关
    {
        "messages": [
            {"role": "user", "content": "发送邮件给 john@example.com，标题是'会议通知'，内容是'明天下午2点开会'"},
            {"role": "assistant", "content": "我来帮您发送邮件。", "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "arguments": "{\"to\": \"john@example.com\", \"subject\": \"会议通知\", \"body\": \"明天下午2点开会\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "发送电子邮件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {
                                "type": "string",
                                "description": "收件人邮箱地址"
                            },
                            "subject": {
                                "type": "string",
                                "description": "邮件主题"
                            },
                            "body": {
                                "type": "string",
                                "description": "邮件正文"
                            }
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            }
        ]
    },

    # 翻译相关
    {
        "messages": [
            {"role": "user", "content": "把'Hello World'翻译成中文"},
            {"role": "assistant", "content": "我来帮您将英文翻译成中文。", "tool_calls": [
                {
                    "id": "call_10",
                    "type": "function",
                    "function": {
                        "name": "translate_text",
                        "arguments": "{\"text\": \"Hello World\", \"source_lang\": \"en\", \"target_lang\": \"zh\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "translate_text",
                    "description": "翻译文本",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "需要翻译的文本"
                            },
                            "source_lang": {
                                "type": "string",
                                "description": "源语言代码，如en、zh、ja等"
                            },
                            "target_lang": {
                                "type": "string",
                                "description": "目标语言代码，如en、zh、ja等"
                            }
                        },
                        "required": ["text", "target_lang"]
                    }
                }
            }
        ]
    },

    # 数据库操作相关
    {
        "messages": [
            {"role": "user", "content": "查询用户表中年龄大于25的所有用户"},
            {"role": "assistant", "content": "我来帮您查询符合条件的用户信息。", "tool_calls": [
                {
                    "id": "call_11",
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "arguments": "{\"query\": \"SELECT * FROM users WHERE age > 25\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "执行SQL查询",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL查询语句"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    },

    # 图像处理相关
    {
        "messages": [
            {"role": "user", "content": "帮我识别这张图片中的文字：image.jpg"},
            {"role": "assistant", "content": "我来帮您识别图片中的文字内容。", "tool_calls": [
                {
                    "id": "call_12",
                    "type": "function",
                    "function": {
                        "name": "ocr_image",
                        "arguments": "{\"image_path\": \"image.jpg\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ocr_image",
                    "description": "识别图片中的文字",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "图片文件路径"
                            },
                            "language": {
                                "type": "string",
                                "description": "识别语言，如zh-cn、en等",
                                "default": "zh-cn"
                            }
                        },
                        "required": ["image_path"]
                    }
                }
            }
        ]
    },

    # 系统命令相关
    {
        "messages": [
            {"role": "user", "content": "查看当前目录下的所有文件"},
            {"role": "assistant", "content": "我来帮您查看当前目录的文件列表。", "tool_calls": [
                {
                    "id": "call_13",
                    "type": "function",
                    "function": {
                        "name": "execute_command",
                        "arguments": "{\"command\": \"ls -la\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "执行系统命令",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "要执行的命令"
                            },
                            "working_directory": {
                                "type": "string",
                                "description": "工作目录（可选）"
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
    },

    # 数据分析相关
    {
        "messages": [
            {"role": "user", "content": "分析CSV文件data.csv中的数据统计信息"},
            {"role": "assistant", "content": "我来帮您分析CSV文件的统计信息。", "tool_calls": [
                {
                    "id": "call_14",
                    "type": "function",
                    "function": {
                        "name": "analyze_csv",
                        "arguments": "{\"file_path\": \"data.csv\", \"analysis_type\": \"statistics\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "analyze_csv",
                    "description": "分析CSV文件数据",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "CSV文件路径"
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "分析类型：statistics、correlation、visualization等"
                            }
                        },
                        "required": ["file_path", "analysis_type"]
                    }
                }
            }
        ]
    },

    # 多函数调用示例
    {
        "messages": [
            {"role": "user", "content": "帮我查询北京天气，然后计算一下今天的温度转换成华氏度"},
            {"role": "assistant", "content": "我来帮您查询北京天气并进行温度转换。", "tool_calls": [
                {
                    "id": "call_15",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"北京\", \"date\": \"today\"}"
                    }
                },
                {
                    "id": "call_16",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": "{\"expression\": \"25 * 9/5 + 32\"}"
                    }
                }
            ]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定地点的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "date": {
                                "type": "string",
                                "description": "日期，可以是today、tomorrow或具体日期"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式，如'2+3*4'、'sqrt(16)'等"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    }
]

# 数据集统计信息
dataset_info = {
    "total_samples": len(function_calling_dataset),
    "categories": {
        "天气查询": 2,
        "时间查询": 2,
        "数学计算": 2,
        "网络搜索": 1,
        "文件操作": 1,
        "邮件发送": 1,
        "文本翻译": 1,
        "数据库操作": 1,
        "图像处理": 1,
        "系统命令": 1,
        "数据分析": 1,
        "多函数调用": 1
    },
    "description": "包含多种常见场景的Function Calling训练数据，适用于Qwen3-0.6B模型微调"
}

if __name__ == "__main__":
    print(f"Function Calling数据集加载完成")
    print(f"总样本数: {dataset_info['total_samples']}")
    print("类别分布:")
    for category, count in dataset_info['categories'].items():
        print(f"  {category}: {count}个样本")
