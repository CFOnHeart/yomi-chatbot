"""
快速启动Multi-Agent系统的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.supervisor_agent import SupervisorAgent
from src.agent.conversation_agent import ConversationAgent
from src.agent.document_agent import DocumentAgent
from src.agent.tool_agent import ToolAgent


def quick_test():
    """快速测试Multi-Agent系统"""
    print("🚀 启动Multi-Agent系统测试...")
    
    try:
        # 创建各种Agent
        print("📝 创建ConversationAgent...")
        conversation_agent = ConversationAgent()
        
        print("📚 创建DocumentAgent...")
        document_agent = DocumentAgent()
        
        print("🔧 创建ToolAgent...")
        tool_agent = ToolAgent()
        
        # 创建SupervisorAgent
        print("👨‍💼 创建SupervisorAgent...")
        supervisor = SupervisorAgent()
        
        # 注册Agent
        print("📋 注册子Agent...")
        supervisor.register_agent(conversation_agent)
        supervisor.register_agent(document_agent)
        supervisor.register_agent(tool_agent)
        
        print("✅ Multi-Agent系统创建成功!")
        
        # 简单测试
        test_query = "你好，请介绍一下你的功能"
        print(f"\n🎯 测试查询: {test_query}")
        
        result = supervisor.invoke(test_query, {"session_id": "test_session"})
        
        print(f"\n🎉 测试结果:")
        print(result)
        
        return supervisor
        
    except Exception as e:
        print(f"❌ 创建系统时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    supervisor = quick_test()
    
    if supervisor:
        print("\n" + "="*50)
        print("🎮 系统已就绪，您可以开始测试!")
        print("="*50)
        
        # 可以继续添加更多测试
        print("\n已注册的Agent:")
        for agent in supervisor.managed_agents:
            print(f"  - {agent.__class__.__name__}: {agent.description[:50]}...")
