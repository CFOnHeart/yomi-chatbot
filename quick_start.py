"""
快速启动脚本 - 测试LangGraph Agent系统 (包含RAG功能和Multi-Agent系统)
"""

import os
import sys
from pathlib import Path

from src.embeddings.azure_openai_embeddings import get_azure_openai_embeddings
from src.global_configuration.model_registry import get_model
from src.rag import RAGSystem
from src.config.settings_store import default_setting_store

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

knowledge_files = [ "Files/ReAct.pdf" ]

def test_multi_agent_system():
    """测试Multi-Agent系统"""
    print("🤖 测试Multi-Agent系统...")
    
    try:
        from src.config.multi_agent_config import get_multi_agent_system
        
        # 获取multi-agent系统
        system = get_multi_agent_system()
        
        print("✅ Multi-Agent系统创建成功!")
        
        # 列出所有Agent
        agents = system.list_agents()
        print(f"📋 已注册的Agent ({len(agents)}个):")
        for agent in agents:
            print(f"  - {agent['name']}: {agent['description'][:50]}...")
        
        # 测试查询
        test_queries = [
            "你好，请介绍一下自己",
            "帮我搜索文档",
            "有什么工具可以使用？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🎯 测试 {i}: {query}")
            try:
                result = system.invoke(query, session_id=f"test_session_{i}")
                print(f"✅ 结果: {result[:100]}...")
            except Exception as e:
                print(f"❌ 错误: {str(e)}")
        
        return system
        
    except Exception as e:
        print(f"❌ Multi-Agent系统测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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
        # 测试RAG系统 - 使用全局设置
        from src.config.settings_store import default_setting_store
        rag = RAGSystem(default_setting_store.document_database, get_azure_openai_embeddings())
        
        #添加测试文档
        doc_ids = rag.add_document(
            title="测试文档",
            content="这是一个测试文档，用于验证RAG功能。",
            category="test"
        )

        # 搜索文档
        results = rag.search_relevant_documents("测试", top_k=1)
        assert len(results) > 0
        print("✅ RAG系统测试通过")
        
    except Exception as e:
        print(f"❌ RAG系统测试失败: {e}")
        return False
    
    try:
        # 测试工具系统
        from src.tools.tool_manager import ToolMatcher
        matcher = ToolMatcher(get_model(default_setting_store.llm_model_name), default_setting_store.tools)
        
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
        memory_manager = SmartMemoryManager(get_model(default_setting_store.llm_model_name), db)
        
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

def upload_files_to_rag():
    """上传文件到RAG系统"""
    print("📂 上传文件到RAG系统...")

    try:
        # 使用全局设置获取RAG系统
        from src.config.settings_store import default_setting_store
        rag_system = RAGSystem(default_setting_store.document_database, get_azure_openai_embeddings())

        for file in knowledge_files:
            if not Path(file).exists():
                print(f"❌ 文件不存在: {file}")
                continue

            # 直接使用RAG系统的add_document_from_file方法
            doc_ids = rag_system.add_document_from_file(
                file, 
                title=Path(file).stem,
                category="knowledge",
                author="系统"
            )
            print(f"✅ 文件已添加: {file}")
            print(f"📄 生成文档块数: {len(doc_ids)}")

        print("📚 所有文件已成功上传到RAG系统")

    except Exception as e:
        print(f"❌ 上传文件失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 LangGraph Agent 系统快速启动")
    print("=" * 50)
    
    # 检查环境
    # if not check_environment():
    #     print("\n❌ 环境检查失败，请配置环境变量后重试")
    #     return
    
    # 测试基本功能
    # if not test_basic_functionality():
    #     print("\n❌ 基本功能测试失败")
    #     return
    #
    # # 测试Multi-Agent系统
    # multi_agent_system = test_multi_agent_system()
    # if not multi_agent_system:
    #     print("\n❌ Multi-Agent系统测试失败")
    #     return
    
    print("\n✅ 所有测试通过！")
    print("=" * 50)
    
    # 询问用户想要做什么
    print("\n请选择操作:")
    print("1. 启动交互式对话 (传统单Agent)")
    print("2. 启动Multi-Agent对话")
    print("3. 加载RAG知识库文件")
    print("4. 演示RAG功能")
    print("5. 退出")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == '1':
        print("\n🎯 启动交互式对话...")
        try:
            from src.tests.agent.run_langgraph_conversation_agent import main as run_agent
            run_agent()
        except Exception as e:
            print(f"❌ 启动对话失败: {e}")

    elif choice == '2':
        print("\n🤖 启动Multi-Agent对话...")
        try:
            interactive_multi_agent_chat(multi_agent_system)
        except Exception as e:
            print(f"❌ 启动Multi-Agent对话失败: {e}")

    elif choice == '3':
        print(f"\n📂 添加已有知识进入RAG数据库，我们默认使用的文件是 {knowledge_files.__str__()}")
        upload_files_to_rag()

    elif choice == '4':
        print("\n📚 演示RAG功能...")
        demo_rag_functionality()
    
    else:
        print("\n👋 再见！")

def interactive_multi_agent_chat(system):
    """交互式Multi-Agent对话"""
    print("\n" + "="*60)
    print("🎮 Multi-Agent系统交互式对话")
    print("="*60)
    print("💬 输入您的问题，输入 'quit' 退出")
    
    session_id = "interactive_session"
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
            
            if not user_input:
                continue
            
            print(f"\n🤖 Multi-Agent系统正在处理...")
            
            # 调用Multi-Agent系统
            result = system.invoke(user_input, session_id=session_id)
            
            print(f"\n🎯 系统回答:")
            print(result)
            
        except KeyboardInterrupt:
            print("\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"\n❌ 处理请求时发生错误: {str(e)}")

def demo_rag_functionality():
    """演示RAG功能"""
    print("🔍 RAG功能演示开始...")
    print("=" * 40)
    
    try:
        from src.agent.conversation_agent import create_agent
        
        # 创建Agent
        agent = create_agent()
        
        # 添加示例文档
        print("\n1. 添加示例文档...")
        
        # 添加Python相关文档
        doc_id1 = agent.add_document(
            title="Python基础语法",
            content="""
            Python是一种高级编程语言，具有简洁的语法和强大的功能。
            
            基本数据类型：
            - 整数 (int): 1, 2, 3
            - 浮点数 (float): 1.5, 2.7
            - 字符串 (str): "hello", 'world'
            - 布尔值 (bool): True, False
            
            变量赋值：
            name = "张三"
            age = 25
            is_student = True
            
            函数定义：
            def greet(name):
                return f"你好, {name}!"
            
            条件语句：
            if age >= 18:
                print("成年人")
            else:
                print("未成年人")
            """,
            category="programming",
            tags="python,basic,syntax",
            author="Python教程"
        )
        
        # 添加项目相关文档
        doc_id2 = agent.add_document(
            title="项目使用说明",
            content="""
            本项目是一个基于LangGraph的智能对话系统，集成了RAG功能。
            
            主要功能：
            1. 智能对话：支持多轮对话和上下文理解
            2. 工具调用：可以调用各种工具来执行任务
            3. RAG检索：能够从文档库中检索相关信息
            4. 记忆管理：自动管理对话历史和摘要
            
            使用方法：
            1. 运行 python quick_start.py 进行快速测试
            2. 使用 python rag_manager.py 管理文档
            3. 运行 python test_rag.py 测试RAG功能
            
            配置要求：
            - Python 3.13+
            - Azure OpenAI API密钥
            - 必要的依赖包
            """,
            category="documentation",
            tags="project,usage,guide",
            author="项目团队"
        )
        
        print(f"✅ 添加文档1: {doc_id1[:8]}...")
        print(f"✅ 添加文档2: {doc_id2[:8]}...")
        
        # 演示搜索功能
        print("\n2. 测试文档搜索...")
        
        search_queries = [
            "Python函数怎么定义？",
            "如何使用这个项目？",
            "项目有哪些功能？"
        ]
        
        for query in search_queries:
            print(f"\n🔍 搜索: '{query}'")
            results = agent.search_documents(query, top_k=2)
            for i, doc in enumerate(results, 1):
                print(f"   {i}. {doc.title} (相似度: {doc.similarity_score:.2f})")
        
        # 演示RAG增强的对话
        print("\n3. 测试RAG增强的对话...")
        
        test_questions = [
            "Python中如何定义函数？",
            "这个项目有什么功能？",
            "如何运行这个项目？"
        ]
        
        session_id = "demo_session"
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🤔 问题{i}: {question}")
            print("-" * 30)
            
            response = agent.chat(session_id, question)
            
            # 显示响应的前200个字符
            print(f"🤖 回答: {response[:200]}...")
            if len(response) > 200:
                print("    (完整回答已截断)")
        
        # 显示统计信息
        print("\n4. 文档统计信息...")
        stats = agent.get_document_stats()
        print(f"📊 文档数量: {stats.get('total_documents', 0)}")
        print(f"📝 总字数: {stats.get('total_words', 0)}")
        print(f"🏷️  分类统计: {stats.get('by_category', {})}")
        
        print("\n✅ RAG功能演示完成!")
        
    except Exception as e:
        print(f"❌ RAG演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
