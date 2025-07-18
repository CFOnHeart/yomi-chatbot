#!/usr/bin/env python3
"""
测试Prompt模板系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from src.config.prompt_manager import get_prompt_manager

@dataclass
class MockDocument:
    """模拟文档对象"""
    document_id: str
    title: str
    content: str
    file_path: str
    similarity_score: float

def test_prompt_manager():
    """测试Prompt管理器"""
    
    print("🧪 开始测试Prompt模板系统")
    print("=" * 80)
    
    # 获取Prompt管理器
    prompt_manager = get_prompt_manager()
    
    # 列出所有模板
    templates = prompt_manager.list_templates()
    print(f"📋 可用模板: {templates}")
    print()
    
    # 创建模拟文档
    documents = [
        MockDocument(
            document_id="doc_001",
            title="Python 编程基础",
            content="Python是一种高级编程语言，具有简洁的语法和强大的功能。它广泛用于数据科学、Web开发、自动化脚本等领域。Python的设计哲学强调代码的可读性，通过使用缩进来表示代码块。",
            file_path="docs/python_basics.md",
            similarity_score=0.92
        ),
        MockDocument(
            document_id="doc_002",
            title="机器学习入门",
            content="机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进性能。常见的机器学习算法包括线性回归、决策树、神经网络等。",
            file_path="docs/ml_intro.pdf",
            similarity_score=0.75
        ),
        MockDocument(
            document_id="doc_003",
            title="数据可视化指南",
            content="数据可视化是将复杂数据转换为图形表示的过程，以便更好地理解和分析数据。常用的可视化库包括matplotlib、seaborn、plotly等。",
            file_path="guides/data_visualization.md",
            similarity_score=0.68
        )
    ]
    
    # 测试结构化RAG prompt
    user_question = "如何开始学习Python编程？"
    
    print("🔍 测试结构化RAG Prompt:")
    print("-" * 40)
    print(f"用户问题: {user_question}")
    print()
    
    try:
        structured_prompt = prompt_manager.get_structured_rag_prompt(user_question, documents)
        print("✅ 生成的结构化Prompt:")
        print(structured_prompt)
        print()
        
    except Exception as e:
        print(f"❌ 生成结构化Prompt失败: {e}")
        return
    
    # 测试错误响应prompt
    print("🚫 测试错误响应Prompt:")
    print("-" * 40)
    
    try:
        error_prompt = prompt_manager.get_error_response_prompt("连接超时")
        print("✅ 生成的错误响应Prompt:")
        print(error_prompt)
        print()
        
    except Exception as e:
        print(f"❌ 生成错误响应Prompt失败: {e}")
    
    # 测试备用响应prompt
    print("🔄 测试备用响应Prompt:")
    print("-" * 40)
    
    try:
        fallback_prompt = prompt_manager.get_fallback_response_prompt(user_question)
        print("✅ 生成的备用响应Prompt:")
        print(fallback_prompt)
        print()
        
    except Exception as e:
        print(f"❌ 生成备用响应Prompt失败: {e}")
    
    # 测试模板重新加载
    print("🔄 测试模板重新加载:")
    print("-" * 40)
    
    try:
        prompt_manager.reload_templates()
        print("✅ 模板重新加载成功")
        
    except Exception as e:
        print(f"❌ 模板重新加载失败: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 Prompt模板系统测试完成！")

def test_template_edge_cases():
    """测试模板边界情况"""
    
    print("\n🧪 测试模板边界情况")
    print("=" * 80)
    
    prompt_manager = get_prompt_manager()
    
    # 测试空文档列表
    print("📋 测试空文档列表:")
    try:
        empty_prompt = prompt_manager.get_structured_rag_prompt("测试问题", [])
        print("✅ 空文档列表处理成功")
        print(f"生成的prompt长度: {len(empty_prompt)} 字符")
        
    except Exception as e:
        print(f"❌ 空文档列表处理失败: {e}")
    
    # 测试超长内容
    print("\n📝 测试超长内容:")
    long_content_doc = MockDocument(
        document_id="long_doc",
        title="超长文档",
        content="这是一个" + "非常" * 200 + "长的文档内容",
        file_path="long_document.txt",
        similarity_score=0.9
    )
    
    try:
        long_prompt = prompt_manager.get_structured_rag_prompt("测试", [long_content_doc])
        print("✅ 超长内容处理成功")
        print(f"生成的prompt长度: {len(long_prompt)} 字符")
        
        # 检查内容是否被截断
        if "..." in long_prompt:
            print("✅ 内容正确截断")
        
    except Exception as e:
        print(f"❌ 超长内容处理失败: {e}")
    
    # 测试不存在的模板
    print("\n❓ 测试不存在的模板:")
    try:
        prompt_manager.populate_template("non_existent_template", {})
        print("❌ 应该抛出错误但没有")
        
    except ValueError as e:
        print(f"✅ 正确抛出ValueError: {e}")
    except Exception as e:
        print(f"❌ 抛出了意外的错误: {e}")

if __name__ == "__main__":
    test_prompt_manager()
    test_template_edge_cases()
