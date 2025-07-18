"""
Multi-Agent系统演示示例
展示SupervisorAgent如何协调不同的ManagedAgent
"""

from src.agent.supervisor_agent import SupervisorAgent
from src.agent.conversation_agent import ConversationAgent
from src.agent.document_agent import DocumentAgent
from src.agent.tool_agent import ToolAgent


def create_multi_agent_system():
    """
    创建一个完整的multi-agent系统
    """
    print("🚀 正在创建Multi-Agent系统...")
    
    # 创建各种ManagedAgent
    conversation_agent = ConversationAgent()
    document_agent = DocumentAgent()
    tool_agent = ToolAgent()
    
    # 创建SupervisorAgent并注册子Agent
    supervisor = SupervisorAgent()
    supervisor.register_agent(conversation_agent)
    supervisor.register_agent(document_agent)
    supervisor.register_agent(tool_agent)
    
    print("✅ Multi-Agent系统创建完成!")
    print(f"📋 已注册的Agent:")
    for agent in supervisor.managed_agents:
        print(f"   - {agent.__class__.__name__}: {agent.description}")
    
    return supervisor


def demo_multi_agent_interaction():
    """
    演示multi-agent系统的交互
    """
    print("\n" + "="*60)
    print("🎭 Multi-Agent系统演示")
    print("="*60)
    
    # 创建系统
    supervisor = create_multi_agent_system()
    
    # 测试用例
    test_queries = [
        "你好，请介绍一下自己的功能",
        "帮我搜索关于人工智能的文档",
        "我需要计算一些数据，有什么工具可以用吗？",
        "能帮我分析一下最近上传的PDF文件吗？",
        "请总结一下系统中所有的功能模块"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*40}")
        print(f"测试 {i}: {query}")
        print('='*40)
        
        try:
            # 构建上下文
            context = {
                "session_id": f"demo_session_{i}",
                "chat_history": f"这是第{i}次交互的演示"
            }
            
            # 调用SupervisorAgent
            result = supervisor.invoke(query, context)
            
            print(f"\n🎯 最终结果:")
            print(result)
            
        except Exception as e:
            print(f"❌ 处理查询时发生错误: {str(e)}")
        
        print("\n" + "-"*40)
        input("按Enter继续下一个测试...")


def interactive_demo():
    """
    交互式演示
    """
    print("\n" + "="*60)
    print("🎮 Multi-Agent系统交互式演示")
    print("="*60)
    
    supervisor = create_multi_agent_system()
    
    print("\n💬 请输入您的问题，输入 'quit' 退出:")
    
    session_id = "interactive_session"
    chat_history = ""
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
            
            if not user_input:
                continue
            
            # 构建上下文
            context = {
                "session_id": session_id,
                "chat_history": chat_history
            }
            
            print(f"\n🤖 正在处理您的请求...")
            
            # 调用SupervisorAgent
            result = supervisor.invoke(user_input, context)
            
            print(f"\n🎯 系统回答:")
            print(result)
            
            # 更新聊天历史
            chat_history += f"\nUser: {user_input}\nAssistant: {result}"
            
        except KeyboardInterrupt:
            print("\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"\n❌ 处理请求时发生错误: {str(e)}")


if __name__ == "__main__":
    print("🎯 Multi-Agent系统演示程序")
    print("\n选择演示模式:")
    print("1. 自动演示 (预设测试用例)")
    print("2. 交互式演示 (手动输入)")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        demo_multi_agent_interaction()
    elif choice == "2":
        interactive_demo()
    else:
        print("❌ 无效选择")
