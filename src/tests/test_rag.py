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
from src.database.faiss_document_db import FAISSDocumentDatabase

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

def test_rag_pdf_integration():
    """测试RAG系统与PDF文档的集成"""
    print("\n🧪 测试RAG系统PDF集成功能")
    print("=" * 50)
    
    # 创建RAG系统
    rag_system = RAGSystem()
    
    # 创建测试PDF文件
    import tempfile
    import os
    
    def create_test_pdf(content: str, file_path: str):
        """创建测试用的PDF文件"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # 分段添加内容
            paragraphs = content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    story.append(Paragraph(paragraph, styles['Normal']))
                    story.append(Spacer(1, 12))
            
            doc.build(story)
            return True
        except ImportError:
            print("⚠️ 需要安装reportlab库来创建测试PDF文件")
            print("运行: pip install reportlab")
            return False
        except Exception as e:
            print(f"⚠️ 创建测试PDF文件失败: {e}")
            return False
    
    # 创建临时PDF文件
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # 创建测试内容
        test_content = """
        Python编程指南

        第一章：Python基础
        Python是一种高级编程语言，具有简洁的语法和强大的功能。它广泛应用于Web开发、数据分析、人工智能等领域。

        基本语法：
        变量定义：name = "Python"
        列表操作：numbers = [1, 2, 3, 4, 5]
        字典操作：person = {"name": "张三", "age": 25}

        第二章：函数和类
        函数定义：
        def greet(name):
            return f"Hello, {name}!"

        类定义：
        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age
            
            def introduce(self):
                return f"我是{self.name}，今年{self.age}岁"

        第三章：异常处理
        try:
            result = 10 / 0
        except ZeroDivisionError:
            print("不能除以零")
        finally:
            print("清理资源")

        这是一个多页的PDF文档，用于测试RAG系统的PDF处理能力。我们需要确保系统能够正确地处理PDF文件，将其分割成适当的块，并为每个块生成合适的元数据。
        """
        
        # 创建测试PDF文件
        if not create_test_pdf(test_content, tmp_path):
            print("❌ 无法创建测试PDF文件，跳过PDF集成测试")
            return
        
        print("\n1. 测试PDF文档添加...")
        
        # 添加PDF文档
        doc_ids = rag_system.add_document_from_file(
            tmp_path,
            title="Python编程指南",
            category="programming",
            tags="python,tutorial,pdf",
            author="测试作者"
        )
        
        print(f"   成功添加PDF文档，生成 {len(doc_ids)} 个文档块")
        assert len(doc_ids) > 0, "应该至少生成一个文档块"
        
        # 验证返回的是列表
        assert isinstance(doc_ids, list), "应该返回文档ID列表"
        
        print("\n2. 测试PDF文档搜索...")
        
        # 搜索测试
        search_queries = [
            "Python编程语言",
            "函数定义",
            "异常处理",
            "类和对象"
        ]
        
        for query in search_queries:
            print(f"\n   搜索: {query}")
            results = rag_system.search_relevant_documents(query, top_k=3)
            
            print(f"   找到 {len(results)} 个相关文档")
            if results:
                for i, result in enumerate(results, 1):
                    print(f"     {i}. {result.title} (相似度: {result.similarity_score:.2f})")
                    # 验证PDF特有的元数据
                    if 'page_number' in result.metadata:
                        print(f"        页码: {result.metadata['page_number']}")
                    if 'chunk_index' in result.metadata:
                        print(f"        块索引: {result.metadata['chunk_index']}")
        
        print("\n3. 测试PDF文档格式化...")
        
        # 测试文档格式化
        results = rag_system.search_relevant_documents("Python编程", top_k=2)
        if results:
            context = rag_system.format_context_for_llm(results)
            assert "相关文档内容" in context, "应该包含格式化的上下文"
            
            references = rag_system.format_source_references(results)
            assert "参考文档" in references, "应该包含格式化的引用"
            
            print("   ✅ 文档格式化成功")
        
        print("\n4. 验证PDF文档元数据...")
        
        # 验证元数据
        results = rag_system.search_relevant_documents("Python", top_k=1)
        if results:
            result = results[0]
            metadata = result.metadata
            
            # 验证基本元数据
            assert metadata.get('file_name'), "应该包含文件名"
            assert metadata.get('file_type') == '.pdf', "应该标记为PDF文件"
            assert metadata.get('loader_type') == 'pdf', "应该标记为PDF加载器"
            
            # 验证PDF特有元数据
            assert 'page_number' in metadata, "应该包含页码"
            assert 'chunk_index' in metadata, "应该包含块索引"
            assert 'total_pages' in metadata, "应该包含总页数"
            
            print("   ✅ 元数据验证通过")
        
        print("\n✅ RAG系统PDF集成测试完成!")
        
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass


def test_pdf_vs_regular_documents():
    """测试PDF文档与常规文档的处理差异"""
    print("\n🧪 测试PDF文档与常规文档的处理差异")
    print("=" * 50)
    
    rag_system = RAGSystem()
    
    # 创建测试内容
    test_content = """
    这是一个测试文档，包含一些Python编程的基本概念。

    变量定义：
    name = "测试"
    age = 25

    函数定义：
    def hello():
        print("Hello, World!")

    这个文档用于对比PDF和常规文档的处理方式。
    """
    
    # 1. 添加常规文档
    print("\n1. 添加常规文档...")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(test_content)
        txt_path = tmp_file.name
    
    try:
        txt_doc_ids = rag_system.add_document_from_file(
            txt_path,
            title="测试文档",
            category="test"
        )
        
        print(f"   常规文档生成 {len(txt_doc_ids)} 个文档块")
        assert len(txt_doc_ids) == 1, "常规文档应该生成一个文档块"
        
        # 2. 创建相同内容的PDF文档
        print("\n2. 添加PDF文档...")
        
        def create_test_pdf(content: str, file_path: str):
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet
                
                doc = SimpleDocTemplate(file_path, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                paragraphs = content.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        story.append(Paragraph(paragraph, styles['Normal']))
                        story.append(Spacer(1, 12))
                
                doc.build(story)
                return True
            except ImportError:
                print("⚠️ 需要reportlab库")
                return False
            except Exception as e:
                print(f"⚠️ 创建PDF失败: {e}")
                return False
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        try:
            if create_test_pdf(test_content, pdf_path):
                pdf_doc_ids = rag_system.add_document_from_file(
                    pdf_path,
                    title="测试PDF文档",
                    category="test"
                )
                
                print(f"   PDF文档生成 {len(pdf_doc_ids)} 个文档块")
                
                # 3. 对比搜索结果
                print("\n3. 对比搜索结果...")
                
                results = rag_system.search_relevant_documents("Python编程", top_k=5)
                
                txt_results = [r for r in results if r.metadata.get('file_type') == '.txt']
                pdf_results = [r for r in results if r.metadata.get('file_type') == '.pdf']
                
                print(f"   找到 {len(txt_results)} 个TXT文档结果")
                print(f"   找到 {len(pdf_results)} 个PDF文档结果")
                
                # 验证元数据差异
                if txt_results:
                    txt_meta = txt_results[0].metadata
                    print(f"   TXT元数据: {list(txt_meta.keys())}")
                
                if pdf_results:
                    pdf_meta = pdf_results[0].metadata
                    print(f"   PDF元数据: {list(pdf_meta.keys())}")
                    
                    # PDF应该有额外的元数据
                    assert 'page_number' in pdf_meta, "PDF应该包含页码"
                    assert 'chunk_index' in pdf_meta, "PDF应该包含块索引"
                    assert 'total_pages' in pdf_meta, "PDF应该包含总页数"
                
                print("   ✅ 元数据对比验证通过")
                
            else:
                print("   ⚠️ 跳过PDF文档测试")
                
        finally:
            try:
                os.unlink(pdf_path)
            except:
                pass
        
    finally:
        try:
            os.unlink(txt_path)
        except:
            pass
    
    print("\n✅ PDF与常规文档对比测试完成!")

def main():
    """主函数"""
    print("🚀 开始RAG系统测试")
    print("=" * 60)
    
    try:
        # 基本功能测试
        test_rag_basic()
        
        # Agent集成测试
        test_agent_with_rag()
        
        # PDF集成测试
        test_rag_pdf_integration()
        
        # PDF与常规文档对比测试
        test_pdf_vs_regular_documents()
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
