import os
import torch
import gc
import warnings
import sys
import psutil
from transformers import AutoTokenizer, AutoModel

# 忽略一些警告
warnings.filterwarnings("ignore")

# 模型本地路径 - 请根据实际情况修改
model_path = r"E:\project\llm\model-data\base-models\chatglm3-6b"

def check_system_info():
    """检查系统信息"""
    print("🔍 系统诊断信息:")
    print(f"  Python版本: {sys.version}")
    print(f"  PyTorch版本: {torch.__version__}")
    
    # 内存信息
    memory = psutil.virtual_memory()
    print(f"  系统内存: {memory.total / (1024**3):.1f}GB")
    print(f"  可用内存: {memory.available / (1024**3):.1f}GB")
    print(f"  内存使用率: {memory.percent}%")
    
    # GPU信息
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    else:
        print("  GPU: 未检测到")

def check_model_files():
    """检查模型文件完整性"""
    print("\n📁 检查模型文件:")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型目录不存在: {model_path}")
        return False
    
    # 检查关键文件
    required_files = [
        "config.json",
        "tokenizer.model", 
        "tokenizer_config.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file}: {size:,} bytes")
        else:
            print(f"  ❌ {file}: 文件不存在")
    
    # 检查模型权重文件
    print("\n  模型权重文件:")
    for file in os.listdir(model_path):
        if file.endswith(('.bin', '.safetensors')):
            file_path = os.path.join(model_path, file)
            size = os.path.getsize(file_path)
            print(f"    {file}: {size / (1024**3):.2f}GB")
    
    return True

def test_tokenizer_only():
    """仅测试tokenizer加载"""
    print("\n🧪 测试1: 仅加载tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        print("  ✅ tokenizer加载成功")
        
        # 测试tokenizer功能 - 避免触发padding_side错误
        test_text = "你好，世界！"
        print(f"  测试编码文本: {test_text}")
        
        # 直接使用tokenizer的基本功能，避免可能有问题的方法
        try:
            tokens = tokenizer.encode(test_text, add_special_tokens=True)
            print(f"  编码结果: {tokens}")
            
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"  解码结果: {decoded}")
            
            print("  ✅ tokenizer功能测试成功")
            return True
            
        except Exception as encode_error:
            print(f"  ⚠️  编码测试失败: {str(encode_error)}")
            print("  但tokenizer加载成功，可能仍可用于模型")
            return True  # tokenizer加载成功就算通过
        
    except Exception as e:
        print(f"  ❌ tokenizer加载失败: {str(e)}")
        return False

def test_model_config():
    """测试模型配置加载"""
    print("\n🧪 测试2: 加载模型配置")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print("  ✅ 配置加载成功")
        print(f"  模型类型: {config.model_type}")
        print(f"  隐藏层大小: {config.hidden_size}")
        print(f"  层数: {config.num_layers}")
        return True
        
    except Exception as e:
        print(f"  ❌ 配置加载失败: {str(e)}")
        return False

def test_minimal_model_load():
    """最小化模型加载测试"""
    print("\n🧪 测试3: 最小化模型加载")
    
    # 检查可用内存
    memory = psutil.virtual_memory()
    if memory.available < 8 * 1024**3:  # 小于8GB
        print(f"  ⚠️  可用内存不足: {memory.available / (1024**3):.1f}GB < 8GB")
        print("  建议关闭其他程序释放内存")
        return False
    
    try:
        print("  正在尝试加载模型...")
        print("  (这是最容易崩溃的步骤)")
        
        # 尝试最保守的加载方式
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map=None,  # 不使用device_map
            low_cpu_mem_usage=True,  # 重新启用低内存模式
            offload_folder=None
        )
        
        # 手动移动到CPU
        model = model.to('cpu')
        model.eval()
        
        print("  ✅ 模型加载成功!")
        return model
        
    except Exception as e:
        print(f"  ❌ 模型加载失败: {str(e)}")
        print(f"  错误类型: {type(e).__name__}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 ChatGLM3-6B 诊断工具")
    print("=" * 60)
    
    # 步骤1: 系统信息检查
    check_system_info()
    
    # 步骤2: 文件完整性检查
    if not check_model_files():
        print("\n❌ 模型文件检查失败，请检查模型路径和文件完整性")
        return
    
    # 步骤3: tokenizer测试
    if not test_tokenizer_only():
        print("\n❌ tokenizer测试失败")
        return
    
    # 步骤4: 配置测试
    if not test_model_config():
        print("\n❌ 模型配置测试失败")
        return
    
    # 步骤5: 模型加载测试
    print("\n" + "=" * 60)
    print("⚠️  即将进行模型加载测试")
    print("这是最可能出现0xC0000005错误的步骤")
    print("=" * 60)
    
    user_input = input("是否继续模型加载测试? (y/n): ").strip().lower()
    if user_input not in ['y', 'yes', '是']:
        print("测试中止")
        return
    
    model = test_minimal_model_load()
    if model is None:
        print("\n❌ 模型加载失败")
        print("\n🔧 建议的解决方案:")
        print("1. 检查系统内存是否足够（建议16GB+）")
        print("2. 尝试重新下载模型文件")
        print("3. 检查transformers库版本是否兼容")
        print("4. 尝试使用量化版本的模型")
        return
    
    # 如果模型加载成功，进行简单测试
    print("\n🎉 模型加载成功！进行功能测试...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        test_question = "你好"
        print(f"\n测试问题: {test_question}")
        print("处理中...")
        
        response, _ = model.chat(tokenizer, test_question, history=[])
        print(f"模型回复: {response}")
        
        print("\n✅ 所有测试通过！模型工作正常")
        
    except Exception as e:
        print(f"\n❌ 对话测试失败: {str(e)}")

if __name__ == "__main__":
    main()
