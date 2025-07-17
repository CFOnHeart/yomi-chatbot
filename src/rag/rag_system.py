import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.database.faiss_document_db import FAISSDocumentDatabase
from src.embeddings.azure_openai_embeddings import get_azure_openai_embeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from dataclasses import dataclass

@dataclass
class DocumentSearchResult:
    """文档搜索结果"""
    document_id: str
    title: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    similarity_score: float
    search_type: str
    snippet: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RAGSystem:
    """检索增强生成系统 - 基于FAISS的版本"""
    
    def __init__(self, document_db: FAISSDocumentDatabase = None, 
                 embeddings: AzureOpenAIEmbeddings = None):
        self.document_db = document_db or FAISSDocumentDatabase()
        self.embeddings = embeddings or get_azure_openai_embeddings()
        self.similarity_threshold = 0.7  # 相似度阈值
        
    def search_relevant_documents(self, query: str, top_k: int = 5, 
                                session_id: str = None) -> List[DocumentSearchResult]:
        """搜索相关文档"""
        start_time = time.time()
        
        # 1. 生成查询的embedding
        query_embedding = None
        try:
            if self.embeddings:
                query_embedding = self.embeddings.embed_query(query)
                query_embedding = np.array(query_embedding)
        except Exception as e:
            print(f"⚠️ 生成查询embedding失败: {e}")
        
        # 2. 使用FAISS进行语义搜索和传统搜索
        search_type = 'semantic' if query_embedding is not None else 'hybrid'
        results = self.document_db.search_documents(
            query=query,
            query_embedding=query_embedding,
            limit=top_k * 2,
            search_type=search_type
        )
        
        # 3. 转换为搜索结果对象
        search_results = []
        for result in results[:top_k]:
            search_result = DocumentSearchResult(
                document_id=result['id'],
                title=result['title'],
                content=result['content'],
                file_path=result.get('file_path', ''),
                start_line=result.get('start_line') or 0,
                end_line=result.get('end_line') or 0,
                similarity_score=result.get('similarity_score', 0.0),
                search_type=result.get('search_type', 'hybrid'),
                snippet=result.get('snippet', ''),
                metadata={
                    'category': result.get('category'),
                    'tags': result.get('tags'),
                    'author': result.get('author'),
                    'created_at': result.get('created_at'),
                    'file_type': result.get('file_type'),
                    'word_count': result.get('word_count'),
                    'char_count': result.get('char_count')
                }
            )
            search_results.append(search_result)
        
        # 4. 记录搜索日志
        execution_time = time.time() - start_time
        self.document_db.log_search(
            query=query,
            results_count=len(search_results),
            search_type='rag_faiss',
            execution_time=execution_time,
            session_id=session_id
        )
        
        return search_results
    
    def add_document(self, title: str, content: str, file_path: str = None, 
                    **metadata) -> str:
        """添加文档到RAG系统"""
        # 生成embedding
        embedding = None
        try:
            if self.embeddings:
                embedding_vector = self.embeddings.embed_documents([content])[0]
                embedding = np.array(embedding_vector)
        except Exception as e:
            print(f"⚠️ 生成embedding失败: {e}")
        
        # 添加到数据库
        doc_id = self.document_db.add_document(
            title=title,
            content=content,
            embedding=embedding,
            file_path=file_path,
            **metadata
        )
        
        return doc_id
    
    def format_context_for_llm(self, documents: List[DocumentSearchResult]) -> str:
        """为LLM格式化上下文"""
        if not documents:
            return ""
        
        context_parts = []
        context_parts.append("=== 相关文档内容 ===")
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"\n【文档 {i}】")
            context_parts.append(f"标题: {doc.title}")
            if doc.file_path:
                context_parts.append(f"文件: {doc.file_path}")
            if doc.start_line is not None and doc.start_line > 0:
                context_parts.append(f"位置: 第 {doc.start_line}-{doc.end_line} 行")
            context_parts.append(f"内容: {doc.content[:500]}...")  # 限制长度
            context_parts.append(f"相似度: {doc.similarity_score:.2f}")
            context_parts.append("-" * 50)
        
        context_parts.append("=== 请基于以上文档内容回答用户问题 ===")
        
        return "\n".join(context_parts)
    
    def format_source_references(self, documents: List[DocumentSearchResult]) -> str:
        """格式化引用来源"""
        if not documents:
            return ""
        
        references = []
        references.append("\n\n📚 **参考文档**:")
        
        for i, doc in enumerate(documents, 1):
            ref_parts = [f"{i}. **{doc.title}**"]
            
            if doc.file_path:
                ref_parts.append(f"   📁 文件: `{doc.file_path}`")
            
            if doc.start_line is not None and doc.start_line > 0:
                ref_parts.append(f"   📍 位置: 第 {doc.start_line}-{doc.end_line} 行")
            
            if doc.metadata.get('author'):
                ref_parts.append(f"   👤 作者: {doc.metadata['author']}")
            
            if doc.metadata.get('created_at'):
                ref_parts.append(f"   📅 创建时间: {doc.metadata['created_at']}")
            
            if doc.metadata.get('category'):
                ref_parts.append(f"   🏷️ 分类: {doc.metadata['category']}")
            
            references.append("\n".join(ref_parts))
        
        return "\n".join(references)
    
    def add_document_from_file(self, file_path: str, **metadata) -> str:
        """从文件添加文档"""
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取文件内容
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='gbk') as f:
                content = f.read()
        
        # 提取文件信息
        title = metadata.get('title', path.stem)
        file_type = path.suffix.lower()
        
        # 如果是代码文件，按行分割并添加行号信息
        if file_type in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go']:
            lines = content.split('\n')
            metadata.update({
                'file_type': file_type,
                'start_line': 1,
                'end_line': len(lines),
                'search_keywords': self._extract_keywords_from_code(content)
            })
        else:
            metadata.update({
                'file_type': file_type,
                'search_keywords': self._extract_keywords_from_text(content)
            })
        
        return self.add_document(
            title=title,
            content=content,
            file_path=str(path),
            **metadata
        )
    
    def _extract_keywords_from_code(self, code: str) -> str:
        """从代码中提取关键词"""
        # 简单的关键词提取，可以根据需要改进
        import re
        
        # 提取函数名、类名、变量名等
        patterns = [
            r'def\s+(\w+)',      # Python函数
            r'class\s+(\w+)',    # Python类
            r'function\s+(\w+)', # JavaScript函数
            r'const\s+(\w+)',    # 常量
            r'let\s+(\w+)',      # 变量
            r'var\s+(\w+)',      # 变量
        ]
        
        keywords = []
        for pattern in patterns:
            matches = re.findall(pattern, code)
            keywords.extend(matches)
        
        return ', '.join(set(keywords))
    
    def _extract_keywords_from_text(self, text: str) -> str:
        """从文本中提取关键词"""
        # 简单的关键词提取，可以集成更复杂的NLP工具
        import re
        
        # 提取中文词汇（长度大于1的中文字符串）
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        
        # 提取英文单词（长度大于3的英文单词）
        english_words = re.findall(r'[a-zA-Z]{4,}', text)
        
        keywords = list(set(chinese_words + english_words))
        
        return ', '.join(keywords[:20])  # 限制关键词数量
    
    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        return self.document_db.get_document_stats()
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        return self.document_db.delete_document(doc_id)
    
    def update_document_embedding(self, doc_id: str):
        """更新文档的embedding"""
        doc = self.document_db.get_document_by_id(doc_id)
        if not doc:
            return False
        
        try:
            embedding_vector = self.embeddings.embed_documents([doc['content']])[0]
            embedding = np.array(embedding_vector)
            self.document_db.update_document_embedding(doc_id, embedding)
            return True
        except Exception as e:
            print(f"⚠️ 更新embedding失败: {e}")
            return False
    
    def rebuild_index(self):
        """重建FAISS索引"""
        return self.document_db.rebuild_faiss_index()
