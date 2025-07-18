"""
全局配置和单例对象管理
使用延迟初始化避免循环依赖
"""

from typing import Optional, TYPE_CHECKING
import threading

# 只在类型检查时导入，避免运行时循环依赖
if TYPE_CHECKING:
    from src.rag.rag_system import RAGSystem
    from src.model.azure_openai_model import AzureOpenAIModel
    from src.embeddings.azure_openai_embeddings import AzureOpenAIEmbeddings
    from src.database.faiss_document_db import FAISSDocumentDatabase


class GlobalSettings:
    """全局设置管理器 - 使用延迟初始化避免循环依赖"""
    
    def __init__(self):
        self._rag_system: Optional['RAGSystem'] = None
        self._llm_model: Optional['AzureOpenAIModel'] = None
        self._llm_embeddings: Optional['AzureOpenAIEmbeddings'] = None
        self._faiss_document_db: Optional['FAISSDocumentDatabase'] = None
        self._lock = threading.Lock()
    
    @property
    def rag_system(self):
        """获取RAG系统实例（延迟初始化）"""
        if self._rag_system is None:
            with self._lock:
                if self._rag_system is None:
                    from src.rag.rag_system import RAGSystem
                    self._rag_system = RAGSystem()
        return self._rag_system
    
    @property
    def llm_model(self):
        """获取LLM模型实例（延迟初始化）"""
        if self._llm_model is None:
            with self._lock:
                if self._llm_model is None:
                    from src.model.azure_openai_model import get_azure_openai_model
                    self._llm_model = get_azure_openai_model()
        return self._llm_model
    
    @property
    def llm_embeddings(self):
        """获取嵌入模型实例（延迟初始化）"""
        if self._llm_embeddings is None:
            with self._lock:
                if self._llm_embeddings is None:
                    from src.embeddings.azure_openai_embeddings import get_azure_openai_embeddings
                    self._llm_embeddings = get_azure_openai_embeddings()
        return self._llm_embeddings
    
    @property
    def faiss_document_db(self):
        """获取FAISS文档数据库实例（延迟初始化）"""
        if self._faiss_document_db is None:
            with self._lock:
                if self._faiss_document_db is None:
                    from src.database.faiss_document_db import FAISSDocumentDatabase
                    self._faiss_document_db = FAISSDocumentDatabase()
        return self._faiss_document_db
    
    def reset_rag_system(self):
        """重置RAG系统（用于测试或重新配置）"""
        with self._lock:
            self._rag_system = None
    
    def reset_llm_model(self):
        """重置LLM模型（用于测试或重新配置）"""
        with self._lock:
            self._llm_model = None
    
    def reset_llm_embeddings(self):
        """重置嵌入模型（用于测试或重新配置）"""
        with self._lock:
            self._llm_embeddings = None
    
    def reset_faiss_document_db(self):
        """重置FAISS文档数据库（用于测试或重新配置）"""
        with self._lock:
            self._faiss_document_db = None
    
    def reset_all(self):
        """重置所有组件"""
        with self._lock:
            self._rag_system = None
            self._llm_model = None
            self._llm_embeddings = None
            self._faiss_document_db = None


# 全局单例实例
_global_settings = GlobalSettings()


def get_rag_system():
    """获取RAG系统实例"""
    return _global_settings.rag_system


def get_llm_model():
    """获取LLM模型实例"""
    return _global_settings.llm_model


def get_llm_embeddings():
    """获取嵌入模型实例"""
    return _global_settings.llm_embeddings


def get_faiss_document_db():
    """获取FAISS文档数据库实例"""
    return _global_settings.faiss_document_db


def reset_global_settings():
    """重置所有全局设置（主要用于测试）"""
    _global_settings.reset_all()


# 向后兼容的别名
def get_global_settings():
    """获取全局设置管理器实例"""
    return _global_settings

