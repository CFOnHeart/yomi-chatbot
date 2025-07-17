"""
LangGraph Agent 示例运行文件
演示完整的对话Agent功能
"""

from src.agent.conversation_agent import create_agent
import uuid

def main():
    """主函数"""
    print("🤖 LangGraph 对话Agent 启动")
    print("="*50)
    
    # 创建Agent
    agent = create_agent()
    
    # 创建会话ID
    session_id = str(uuid.uuid4())[:8]
    print(f"📱 会话ID: {session_id}")
    
    # 交互式对话
    print("\n💬 开始对话 (输入 'quit' 退出, 'info' 查看会话信息, 'sessions' 查看所有会话)")
    print("-" * 50)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 用户: ").strip()
            
            if not user_input:
                continue
            
            # 特殊命令处理
            if user_input.lower() == 'quit':
                print("👋 再见！")
                break
            elif user_input.lower() == 'info':
                show_session_info(agent, session_id)
                continue
            elif user_input.lower() == 'sessions':
                show_all_sessions(agent)
                continue
            elif user_input.lower().startswith('switch '):
                # 切换会话
                new_session = user_input[7:].strip()
                if new_session:
                    session_id = new_session
                    print(f"📱 切换到会话: {session_id}")
                continue
            
            # 处理用户输入
            response = agent.chat(session_id, user_input)
            
            print(f"\n🤖 助手: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 收到中断信号，退出...")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")

def show_session_info(agent, session_id):
    """显示会话信息"""
    info = agent.get_session_info(session_id)
    if info:
        print(f"\n📊 会话 {session_id} 信息:")
        print(f"   • 消息数量: {info['message_count']}")
        print(f"   • 文本长度: {info['text_length']}")
        print(f"   • 需要摘要: {'是' if info['needs_summary'] else '否'}")
    else:
        print(f"\n❌ 会话 {session_id} 不存在")

def show_all_sessions(agent):
    """显示所有会话"""
    sessions = agent.list_sessions()
    if sessions:
        print(f"\n📋 所有会话 ({len(sessions)} 个):")
        for session in sessions:
            print(f"   • {session['session_id']} - {session['updated_at']}")
    else:
        print("\n📋 暂无会话记录")

def demo_conversation():
    """演示对话功能"""
    print("\n🎯 运行演示对话...")
    
    agent = create_agent()
    session_id = "demo_session"
    
    # 演示对话
    demo_inputs = [
        "你好！我是张三",
        "请计算 15 + 25",
        "再计算 6 × 8",
        "我刚才说我叫什么名字？",
        "请帮我解决一个复杂的数学问题",
        "总结一下我们的对话"
    ]
    
    for i, user_input in enumerate(demo_inputs, 1):
        print(f"\n--- 演示对话 {i}/{len(demo_inputs)} ---")
        print(f"👤 用户: {user_input}")
        
        response = agent.chat(session_id, user_input)
        print(f"🤖 助手: {response}")
        
        # 等待用户按键继续
        input("\n按回车键继续...")
    
    print("\n✅ 演示对话完成")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_conversation()
    else:
        main()
