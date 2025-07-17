"""
快速启动脚本 - 测试LangGraph Agent系统
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试基本功能...")
    
    try:
        # 测试数据库
        from src.database.chat_db import ChatDatabase
        db = ChatDatabase("test_quick.db")
        
        # 测试会话创建
        session_id = "quick_test"
        db.create_session(session_id)
        db.add_message(session_id, "human", "Hello")
        db.add_message(session_id, "ai", "Hi there!")
        
        history = db.get_session_history(session_id)
        assert len(history) == 2
        print("✅ 数据库测试通过")
        
        # 清理测试数据库
        os.remove("test_quick.db")
        
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}")
        return False
    
    try:
        # 测试工具系统
        from src.tools.tool_manager import ToolMatcher
        matcher = ToolMatcher()
        
        # 测试工具获取
        add_tool = matcher.get_tool_by_name("add")
        assert add_tool is not None
        print("✅ 工具系统测试通过")
        
    except Exception as e:
        print(f"❌ 工具系统测试失败: {e}")
        return False
    
    try:
        # 测试记忆管理
        from src.memory.smart_memory_manager import SmartMemoryManager
        from src.database.chat_db import ChatDatabase
        
        db = ChatDatabase("test_memory.db")
        memory_manager = SmartMemoryManager(db)
        
        session_id = "memory_test"
        memory_manager.initialize_session(session_id)
        
        history = memory_manager.get_session_history(session_id)
        assert history is not None
        print("✅ 记忆管理测试通过")
        
        # 清理测试数据库
        os.remove("test_memory.db")
        
    except Exception as e:
        print(f"❌ 记忆管理测试失败: {e}")
        return False
    
    return True

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ 缺少环境变量: {', '.join(missing_vars)}")
        print("请确保.env文件包含以下变量:")
        for var in required_vars:
            print(f"  {var}=your_value")
        return False
    
    print("✅ 环境配置检查通过")
    return True

def main():
    """主函数"""
    print("🚀 LangGraph Agent 系统快速启动")
    print("=" * 50)
    
    # 检查环境
    # if not check_environment():
    #     print("\n❌ 环境检查失败，请配置环境变量后重试")
    #     return
    
    # 测试基本功能
    if not test_basic_functionality():
        print("\n❌ 基本功能测试失败")
        return
    
    print("\n✅ 所有测试通过！")
    print("=" * 50)
    
    # 询问是否启动交互式对话
    user_input = input("\n是否启动交互式对话? (y/n): ").strip().lower()
    
    if user_input in ['y', 'yes', '是']:
        print("\n🎯 启动交互式对话...")
        try:
            from src.tests.agent.run_langgraph_conversation_agent import main as run_agent
            run_agent()
        except Exception as e:
            print(f"❌ 启动对话失败: {e}")
    else:
        print("\n👋 再见！")

if __name__ == "__main__":
    main()
