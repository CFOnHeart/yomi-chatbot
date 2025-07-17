#!/usr/bin/env python3
"""
RAG系统测试脚本
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.agent.conversation_agent import create_agent
from src.rag.rag_system import RAGSystem
from src.database.document_db import DocumentDatabase

def test_rag_basic():
    """基本RAG功能测试"""
    print("🧪 开始RAG系统基本功能测试")
    print("=" * 50)
    
    # 创建RAG系统
    rag_system = RAGSystem()
    
    # 添加测试文档
    print("\n1. 添加测试文档...")
    doc_id1 = rag_system.add_document(
        title="Python基础教程",
        content="""
        Python是一种高级编程语言，具有以下特点：
        1. 简洁易读的语法
        2. 丰富的标准库
        3. 强大的第三方库生态
        4. 跨平台支持
        
        变量定义：
        name = "张三"
        age = 25
        
        函数定义：
        def greet(name):
            return f"Hello, {name}!"
        """,
        file_path="python_tutorial.md",
        category="programming",
        tags="python,tutorial,basic",
        author="教程作者"
    )
    
    doc_id2 = rag_system.add_document(
        title="数据库设计规范",
        content="""
        数据库设计的基本原则：
        1. 表结构设计要合理
        2. 建立适当的索引
        3. 选择合适的数据类型
        4. 考虑数据的完整性约束
        
        SQLite是一个轻量级的关系数据库：
        - 无需安装服务器
        - 数据存储在单个文件中
        - 支持标准SQL语法
        - 适合小型应用
        """,
        file_path="database_design.md",
        category="database",
        tags="database,design,sqlite",
        author="数据库专家"
    )
    
    print(f"✅ 添加文档1: {doc_id1}")
    print(f"✅ 添加文档2: {doc_id2}")
    
    # 测试搜索
    print("\n2. 测试文档搜索...")
    
    # 搜索Python相关内容
    print("\n🔍 搜索 'Python编程':")
    results = rag_system.search_relevant_documents("Python编程", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.title} (相似度: {doc.similarity_score:.2f})")
    
    # 搜索数据库相关内容
    print("\n🔍 搜索 'SQLite数据库':")
    results = rag_system.search_relevant_documents("SQLite数据库", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.title} (相似度: {doc.similarity_score:.2f})")
    
    # 测试上下文格式化
    print("\n3. 测试上下文格式化...")
    results = rag_system.search_relevant_documents("Python函数", top_k=2)
    context = rag_system.format_context_for_llm(results)
    print("\n📖 生成的LLM上下文:")
    print(context[:300] + "...")
    
    # 测试引用格式化
    references = rag_system.format_source_references(results)
    print("\n📚 生成的引用信息:")
    print(references)
    
    # 获取统计信息
    print("\n4. 获取统计信息...")
    stats = rag_system.get_document_stats()
    print(f"📊 总文档数: {stats.get('total_documents', 0)}")
    print(f"📝 总字数: {stats.get('total_words', 0)}")
    print(f"📁 文件类型: {stats.get('by_type', {})}")
    print(f"🏷️  分类: {stats.get('by_category', {})}")
    
    print("\n✅ 基本功能测试完成!")

def test_agent_with_rag():
    """测试Agent与RAG集成"""
    print("\n🤖 开始Agent与RAG集成测试")
    print("=" * 50)
    
    # 创建Agent
    agent = create_agent()
    
    # 添加一些测试文档
    print("\n1. 添加测试文档到Agent...")
    
    doc_id1 = agent.add_document(
        title="LangChain使用指南",
        content="""
        LangChain是一个用于构建LLM应用的框架：
        
        核心概念：
        1. Chain - 处理链条
        2. Agent - 智能代理
        3. Memory - 记忆管理
        4. Tool - 工具调用
        
        安装方法：
        pip install langchain
        
        基本用法：
        from langchain.llms import OpenAI
        llm = OpenAI()
        response = llm("Hello world")
        """,
        category="framework",
        tags="langchain,llm,ai",
        author="LangChain团队"
    )
    
    doc_id2 = agent.add_document(
        title="项目架构说明",
        content="""
        本项目采用模块化架构设计：
        
        目录结构：
        src/
        ├── agent/          # 智能代理
        ├── database/       # 数据库层
        ├── memory/         # 记忆管理
        ├── tools/          # 工具系统
        ├── rag/           # RAG系统
        └── embeddings/     # 嵌入向量
        
        核心流程：
        1. 用户输入
        2. 工具检测
        3. RAG检索
        4. LLM生成
        5. 响应输出
        """,
        category="architecture",
        tags="project,architecture,design",
        author="项目团队"
    )
    
    print(f"✅ 添加文档1: {doc_id1}")
    print(f"✅ 添加文档2: {doc_id2}")
    
    # 测试对话
    print("\n2. 测试RAG增强的对话...")
    
    test_queries = [
        "如何使用LangChain？",
        "项目的架构是什么样的？",
        "什么是Chain和Agent？",
        "如何安装LangChain？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🤔 测试问题 {i}: {query}")
        print("-" * 40)
        
        response = agent.chat(f"test_session_{i}", query)
        print(f"🤖 AI回复: {response[:200]}...")
        print()
    
    # 显示文档统计
    print("\n3. 显示文档统计信息...")
    stats = agent.get_document_stats()
    print(f"📊 RAG系统中共有 {stats.get('total_documents', 0)} 个文档")
    
    print("\n✅ Agent与RAG集成测试完成!")

def main():
    """主函数"""
    print("🚀 开始RAG系统测试")
    print("=" * 60)
    
    try:
        # 基本功能测试
        test_rag_basic()
        
        # Agent集成测试
        test_agent_with_rag()
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
