"""
RAG (Retrieval-Augmented Generation) 模块

提供文档检索增强生成功能，包括：
- 文档存储和索引
- 语义搜索
- 上下文生成
- 引用管理
"""

from .rag_system import RAGSystem, DocumentSearchResult

__all__ = ['RAGSystem', 'DocumentSearchResult']
