import os
import sys
import warnings
import subprocess

# 忽略警告
warnings.filterwarnings("ignore")

# 模型本地路径
model_path = r"E:\project\llm\model-data\base-models\chatglm3-6b"

def check_and_fix_environment():
    """检查并修复环境问题"""
    print("=" * 60)
    print("🔧 ChatGLM3-6B 环境修复工具")
    print("=" * 60)
    
    print("🔍 检查当前环境...")
    
    # 检查transformers版本
    try:
        import transformers
        print(f"  transformers版本: {transformers.__version__}")
        
        # 检查版本兼容性
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        
        if major > 4 or (major == 4 and minor > 35):
            print("  ⚠️  transformers版本可能过新，建议降级")
            print("  建议版本: 4.30.0 - 4.35.0")
            
            user_input = input("是否自动降级transformers? (y/n): ").strip().lower()
            if user_input in ['y', 'yes', '是']:
                print("正在降级transformers...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "transformers==4.30.0", "--force-reinstall"
                    ])
                    print("✅ transformers降级成功，请重启Python环境")
                    return False
                except Exception as e:
                    print(f"❌ 降级失败: {e}")
        
    except ImportError:
        print("❌ transformers未安装")
        return False
    
    # 检查torch版本
    try:
        import torch
        print(f"  torch版本: {torch.__version__}")
    except ImportError:
        print("❌ torch未安装")
        return False
    
    return True

def test_with_different_methods():
    """使用不同方法测试模型加载"""
    if not check_and_fix_environment():
        return
    
    print("\n🧪 尝试不同的加载方法...")
    
    # 方法1: 使用safetensors (如果存在)
    print("\n方法1: 检查safetensors格式...")
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    if safetensors_files:
        print(f"  发现safetensors文件: {len(safetensors_files)}个")
        try:
            from transformers import AutoTokenizer, AutoModel
            print("  尝试使用safetensors加载...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype="auto",
                use_safetensors=True
            )
            print("  ✅ safetensors加载成功!")
            return model
        except Exception as e:
            print(f"  ❌ safetensors加载失败: {str(e)[:100]}...")
    else:
        print("  未发现safetensors文件")
    
    # 方法2: 分片加载
    print("\n方法2: 尝试分片加载...")
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        print("  设置内存限制...")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            max_memory={0: "8GB", "cpu": "16GB"},
            offload_folder="./temp_offload",
            low_cpu_mem_usage=True
        )
        print("  ✅ 分片加载成功!")
        return model
    except Exception as e:
        print(f"  ❌ 分片加载失败: {str(e)[:100]}...")
    
    # 方法3: 手动加载
    print("\n方法3: 尝试手动分步加载...")
    try:
        from transformers import AutoConfig, AutoTokenizer
        import torch
        
        # 先加载配置
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print("  ✅ 配置加载成功")
        
        # 尝试使用更低级的API
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
        print("  尝试使用低级API...")
        
        # 这里可能需要更具体的实现
        print("  ❌ 手动加载需要更多实现")
        
    except Exception as e:
        print(f"  ❌ 手动加载失败: {str(e)[:100]}...")
    
    # 方法4: 使用不同的库
    print("\n方法4: 建议使用其他方案...")
    print("  💡 可能的替代方案:")
    print("  1. 使用ChatGLM官方代码库")
    print("  2. 使用vLLM或FastChat等推理框架")
    print("  3. 使用量化版本的模型")
    print("  4. 使用Docker容器环境")
    
    return None

def create_alternative_script():
    """创建替代脚本"""
    print("\n📝 创建替代解决方案...")
    
    alternative_code = '''
# ChatGLM3-6B 替代方案
# 如果transformers加载失败，可以尝试以下方法：

# 方案1: 使用官方代码
# git clone https://github.com/THUDM/ChatGLM3
# 然后使用官方的加载方式

# 方案2: 使用量化模型
# 下载int4或int8量化版本，内存占用更小

# 方案3: 使用API方式
# 考虑使用在线API或本地部署的API服务

# 方案4: 环境隔离
# 创建新的conda环境，使用特定版本的依赖
# conda create -n chatglm python=3.9
# conda activate chatglm
# pip install transformers==4.30.0 torch==1.13.1

print("请参考注释中的替代方案")
'''
    
    with open("src/chatglm3_6b/alternative_solutions.py", "w", encoding="utf-8") as f:
        f.write(alternative_code)
    
    print("✅ 已创建替代方案文件: src/chatglm3_6b/alternative_solutions.py")

def main():
    """主函数"""
    model = test_with_different_methods()
    
    if model is None:
        create_alternative_script()
        print("\n" + "=" * 60)
        print("❌ 所有加载方法都失败了")
        print("=" * 60)
        print("🔧 建议的解决方案:")
        print("1. 降级transformers: pip install transformers==4.30.0")
        print("2. 重新下载模型文件")
        print("3. 使用官方ChatGLM3代码库")
        print("4. 考虑使用量化版本模型")
        print("5. 创建新的conda环境测试")
    else:
        print("\n✅ 模型加载成功！可以进行测试")
        
        # 简单测试
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            test_question = "你好"
            response, _ = model.chat(tokenizer, test_question, history=[])
            print(f"测试对话: {test_question} -> {response}")
            
        except Exception as e:
            print(f"对话测试失败: {str(e)}")

if __name__ == "__main__":
    main()
