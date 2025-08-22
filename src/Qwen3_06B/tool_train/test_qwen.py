from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型路径
base_model_path = r"E:\project\llm\model-data\base-models\Qwen3-0.6B"
fine_tuned_model_path = r"E:\project\llm\model-data\train-models\Qwen3-tool"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True).eval()
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, trust_remote_code=True).eval()

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.to(device)
fine_tuned_model.to(device)

# 测试提示（模拟一个需要调用函数的场景）
prompt = "请调用天气API，查询北京今天的天气情况。"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 生成输出
with torch.no_grad():
    base_output = base_model.generate(**inputs, max_length=100)
    fine_tuned_output = fine_tuned_model.generate(**inputs, max_length=100)

# 解码输出
base_response = tokenizer.decode(base_output[0], skip_special_tokens=True)
fine_tuned_response = tokenizer.decode(fine_tuned_output[0], skip_special_tokens=True)

# 输出结果
print("原始模型输出：", base_response)
print("微调模型输出：", fine_tuned_response)
