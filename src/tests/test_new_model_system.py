"""
测试新的模型管理系统
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_registry():
    """测试模型注册仓库"""
    print("🧪 Testing Model Registry...")
    
    from src.global_configuration.model_registry import get_model_registry, get_model
    
    registry = get_model_registry()
    
    # 列出所有注册的模型
    print(f"📋 Available models: {registry.list_models()}")
    print(f"🏢 Available providers: {registry.list_providers()}")
    
    # 测试获取特定模型
    azure_model = get_model("azure/gpt-4o")
    openai_model = get_model("openai/gpt-4o")
    
    if azure_model:
        print(f"✅ Azure model found: {azure_model}")
        print(f"   Model info: {azure_model.get_model_info()}")
    else:
        print("❌ Azure model not found")
    
    if openai_model:
        print(f"✅ OpenAI model found: {openai_model}")
        print(f"   Model info: {openai_model.get_model_info()}")
    else:
        print("❌ OpenAI model not found")


def test_settings_integration():
    """测试与settings的集成"""
    print("\n🧪 Testing Settings Integration...")
    
    from src.config.settings import get_llm_model
    
    try:
        llm = get_llm_model()
        print(f"✅ LLM from settings: {llm}")
        print(f"   Type: {type(llm)}")
        
        # 测试invoke方法
        if hasattr(llm, 'invoke'):
            print("✅ Model has invoke method")
        else:
            print("❌ Model missing invoke method")
            
    except Exception as e:
        print(f"❌ Error getting LLM from settings: {e}")


def test_model_invocation():
    """测试模型调用"""
    print("\n🧪 Testing Model Invocation...")
    
    try:
        from src.config.settings import get_llm_model
        
        llm = get_llm_model()
        
        # 测试简单的调用
        test_prompt = "Say 'Hello, this is a test from the new model system!'"
        
        print(f"🔄 Testing model invocation with prompt: {test_prompt}")
        
        # 注意：这里可能会因为API密钥或网络问题失败，这是正常的
        try:
            response = llm.invoke(test_prompt)
            print(f"✅ Model response: {response}")
        except Exception as invoke_error:
            print(f"⚠️ Model invocation failed (this might be due to API keys/network): {invoke_error}")
            
    except Exception as e:
        print(f"❌ Error in model invocation test: {e}")


if __name__ == "__main__":
    print("🚀 Starting Model System Tests...")
    
    test_model_registry()
    test_settings_integration()
    test_model_invocation()
    
    print("\n✅ Model system tests completed!")
