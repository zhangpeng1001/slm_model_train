import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModel

# 忽略警告
warnings.filterwarnings("ignore")

# 模型本地路径
model_path = r"E:\project\llm\model-data\base-models\chatglm3-6b"

def test_chatglm3_fixed():
    """
    修复版ChatGLM3-6B测试
    专门解决padding_side兼容性问题
    """
    print("=" * 60)
    print("🤖 ChatGLM3-6B 修复版测试")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    try:
        print("🔄 加载tokenizer...")
        # 使用最基本的tokenizer加载方式
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✅ tokenizer加载成功")
        
        print("\n🔄 加载模型...")
        print("(使用CPU模式，请耐心等待...)")
        
        # 使用最稳定的模型加载方式
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        ).to('cpu').eval()
        
        print("✅ 模型加载成功")
        
        print("\n" + "=" * 60)
        print("开始对话测试")
        print("=" * 60)
        
        # 预设测试问题
        test_questions = [
            "你好",
            "请介绍一下自己",
            "1+1等于几？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 测试 {i}: {question}")
            try:
                # 使用最简单的chat调用，避免复杂参数
                response, _ = model.chat(tokenizer, question, history=[])
                print(f"🤖 回复: {response}")
                print("-" * 40)
            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")
                # 尝试更基础的方式
                try:
                    inputs = tokenizer(question, return_tensors="pt")
                    print("⚠️  使用基础生成方式...")
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_length=100, do_sample=False)
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"🤖 回复: {response}")
                except Exception as e2:
                    print(f"❌ 基础生成也失败: {str(e2)}")
        
        # 交互模式
        print(f"\n{'='*60}")
        print("🎯 交互模式 (输入 'quit' 退出)")
        print("="*60)
        
        while True:
            try:
                question = input("\n请输入问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出', 'q']:
                    break
                
                if not question:
                    continue
                
                print("🤔 思考中...")
                
                # 优先使用chat方法
                try:
                    response, _ = model.chat(tokenizer, question, history=[])
                    print(f"🤖 回复: {response}")
                except:
                    # 备用方案：直接生成
                    try:
                        inputs = tokenizer(question, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=200,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # 移除输入部分，只保留回复
                        if question in response:
                            response = response.replace(question, "").strip()
                        print(f"🤖 回复: {response}")
                    except Exception as e:
                        print(f"❌ 生成失败: {str(e)}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 错误: {str(e)}")
        
        print("\n👋 测试完成")
        
    except Exception as e:
        print(f"❌ 程序失败: {str(e)}")
        print("\n💡 可能的解决方案:")
        print("1. 检查transformers版本: pip install transformers==4.30.0")
        print("2. 检查torch版本兼容性")
        print("3. 尝试重新安装依赖")

if __name__ == "__main__":
    test_chatglm3_fixed()
