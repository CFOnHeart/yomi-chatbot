#!/usr/bin/env python3
"""
测试结构化RAG响应的脚本
"""

import json
from dataclasses import dataclass
from typing import List

@dataclass
class MockDocument:
    """模拟文档对象"""
    id: str
    title: str
    content: str
    file_path: str
    similarity_score: float
    start_line: int = 0
    end_line: int = 0

def test_structured_rag_prompt():
    """测试结构化RAG prompt的构建"""
    
    # 模拟AgentNodes类的方法
    class MockAgentNodes:
        def _build_structured_rag_prompt(self, user_question: str, documents: list) -> str:
            """构建结构化的RAG prompt"""
            # 构建文档信息
            doc_info = []
            for i, doc in enumerate(documents, 1):
                doc_metadata = {
                    "doc_id": doc.id,
                    "title": doc.title,
                    "file_path": doc.file_path or "Unknown",
                    "similarity_score": round(doc.similarity_score, 3)
                }
                
                doc_info.append(f"""
Document {i} (ID: {doc.id}):
Title: {doc.title}
Source: {doc.file_path or "Unknown"}
Similarity Score: {doc.similarity_score:.3f}
Content: {doc.content[:500]}{'...' if len(doc.content) > 500 else ''}
""")
            
            docs_text = "\n".join(doc_info)
            
            prompt = f"""You are an intelligent assistant that analyzes provided documents to answer user questions. Your task is to:

1. First, determine which documents are relevant to the user's question
2. Answer based on relevant documents if available
3. Provide a comprehensive response combining document information with your knowledge

**User Question:** {user_question}

**Provided Documents:**
{docs_text}

**Instructions:**
- Carefully analyze each document's relevance to the question
- If documents are relevant, use them as primary sources for your answer
- Combine document information with your general knowledge for a complete response
- Be honest about what information comes from documents vs. your knowledge

**Response Format (JSON):**
{{
    "related_doc": ["doc_id1", "doc_id2", ...],
    "answer_from_provided_doc": "Answer based on the related documents. Leave empty if no documents are relevant.",
    "answer_from_llm": "Comprehensive answer combining document information and your knowledge."
}}

Please respond in valid JSON format only."""
            
            return prompt
        
        def _process_structured_response(self, llm_response: str, documents: list) -> str:
            """处理结构化的LLM响应，转换为markdown格式"""
            import json
            import re
            
            try:
                # 尝试解析JSON响应
                # 先清理可能的markdown代码块
                json_content = llm_response.strip()
                if json_content.startswith("```json"):
                    json_content = re.sub(r'^```json\s*', '', json_content)
                    json_content = re.sub(r'\s*```$', '', json_content)
                elif json_content.startswith("```"):
                    json_content = re.sub(r'^```\s*', '', json_content)
                    json_content = re.sub(r'\s*```$', '', json_content)
                
                response_data = json.loads(json_content)
                
                # 构建markdown响应
                markdown_response = []
                
                # 主要回答
                main_answer = response_data.get("answer_from_llm", "")
                if main_answer:
                    markdown_response.append(main_answer)
                
                # 添加分隔线
                markdown_response.append("\n---\n")
                
                # 添加相关文档信息
                related_doc_ids = response_data.get("related_doc", [])
                doc_answer = response_data.get("answer_from_provided_doc", "")
                
                if related_doc_ids and doc_answer:
                    markdown_response.append("### 📚 基于文档的回答")
                    markdown_response.append(doc_answer)
                    markdown_response.append("")
                
                if related_doc_ids:
                    markdown_response.append("### 📖 相关文档")
                    
                    # 创建文档ID到文档对象的映射
                    doc_map = {doc.id: doc for doc in documents}
                    
                    for doc_id in related_doc_ids:
                        if doc_id in doc_map:
                            doc = doc_map[doc_id]
                            markdown_response.append(f"- **{doc.title}**")
                            if doc.file_path:
                                markdown_response.append(f"  - 📁 文件: `{doc.file_path}`")
                            if hasattr(doc, 'start_line') and doc.start_line > 0:
                                markdown_response.append(f"  - 📍 位置: 第 {doc.start_line}-{doc.end_line} 行")
                            markdown_response.append(f"  - 🎯 相似度: {doc.similarity_score:.3f}")
                            markdown_response.append("")
                else:
                    markdown_response.append("### ℹ️ 文档信息")
                    markdown_response.append("未找到与问题直接相关的文档，回答主要基于AI的通用知识。")
                
                return "\n".join(markdown_response)
                
            except (json.JSONDecodeError, KeyError) as e:
                # 如果JSON解析失败，返回原始响应
                print(f"⚠️ 无法解析结构化响应，返回原始内容: {e}")
                
                # 构建基本的markdown格式
                markdown_response = [llm_response]
                markdown_response.append("\n---\n")
                markdown_response.append("### 📖 相关文档")
                
                for i, doc in enumerate(documents, 1):
                    markdown_response.append(f"{i}. **{doc.title}**")
                    if doc.file_path:
                        markdown_response.append(f"   - 📁 `{doc.file_path}`")
                    markdown_response.append(f"   - 🎯 相似度: {doc.similarity_score:.3f}")
                    markdown_response.append("")
                
                return "\n".join(markdown_response)
    
    # 创建模拟文档
    documents = [
        MockDocument(
            id="doc1",
            title="Python 基础教程",
            content="Python是一种高级编程语言，具有简洁的语法和强大的功能。它广泛用于数据科学、Web开发、自动化等领域。",
            file_path="docs/python_tutorial.md",
            similarity_score=0.85,
            start_line=1,
            end_line=50
        ),
        MockDocument(
            id="doc2", 
            title="机器学习概述",
            content="机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。常见的算法包括监督学习、无监督学习和强化学习。",
            file_path="docs/ml_overview.md",
            similarity_score=0.72,
            start_line=10,
            end_line=80
        )
    ]
    
    # 测试prompt构建
    nodes = MockAgentNodes()
    user_question = "如何学习Python编程？"
    
    prompt = nodes._build_structured_rag_prompt(user_question, documents)
    
    print("=" * 80)
    print("🔍 生成的结构化RAG Prompt:")
    print("=" * 80)
    print(prompt)
    
    # 模拟LLM响应
    mock_llm_response = {
        "related_doc": ["doc1"],
        "answer_from_provided_doc": "根据提供的文档，Python是一种高级编程语言，具有简洁的语法和强大的功能，广泛用于数据科学、Web开发、自动化等领域。",
        "answer_from_llm": "学习Python编程建议从基础语法开始，然后逐步学习数据结构、面向对象编程等概念。Python的语法简洁易懂，是很好的入门编程语言。可以通过官方教程、在线课程、实践项目等方式学习。建议多动手编程，从简单的程序开始，逐步提高难度。"
    }
    
    mock_response_json = json.dumps(mock_llm_response, ensure_ascii=False, indent=2)
    
    # 测试响应处理
    final_response = nodes._process_structured_response(mock_response_json, documents)
    
    print("\n" + "=" * 80)
    print("🤖 模拟的LLM JSON响应:")
    print("=" * 80)
    print(mock_response_json)
    
    print("\n" + "=" * 80)
    print("📝 最终的Markdown格式响应:")
    print("=" * 80)
    print(final_response)
    
    # 测试JSON解析失败的情况
    invalid_response = "这是一个无效的JSON响应，无法解析。"
    fallback_response = nodes._process_structured_response(invalid_response, documents)
    
    print("\n" + "=" * 80)
    print("⚠️ JSON解析失败时的fallback响应:")
    print("=" * 80)
    print(fallback_response)

if __name__ == "__main__":
    test_structured_rag_prompt()
