"""
Embedding模型基类定义，提供统一的Embedding模型接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from langchain_core.embeddings import Embeddings


class BaseManagedEmbedding(ABC):
    """
    管理所有Embedding模型的抽象基类
    提供统一的接口来封装不同的LangChain Embedding模型
    """
    
    def __init__(self, model_name: str, model_provider: str):
        self._model_name = model_name
        self._model_provider = model_provider
        self._embedding_instance: Optional[Embeddings] = None
        self._is_initialized = False
    
    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name
    
    @property
    def model_provider(self) -> str:
        """获取模型提供商"""
        return self._model_provider
    
    @property
    def full_name(self) -> str:
        """获取完整模型名称，格式：provider/model_name"""
        return f"{self._model_provider}/{self._model_name}"
    
    @property
    def is_initialized(self) -> bool:
        """检查模型是否已初始化"""
        return self._is_initialized
    
    @abstractmethod
    def _create_embedding(self) -> Embeddings:
        """
        创建具体的LangChain Embedding模型实例
        子类必须实现此方法
        """
        pass
    
    def _initialize_embedding(self) -> Embeddings:
        """
        懒加载初始化embedding模型
        """
        if self._embedding_instance is None:
            self._embedding_instance = self._create_embedding()
            self._is_initialized = True
        return self._embedding_instance
    
    @property
    def embedding(self) -> Embeddings:
        """
        获取LangChain Embedding模型实例（懒加载）
        """
        return self._initialize_embedding()
    
    # 代理方法，直接调用底层embedding模型的方法
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成嵌入
        
        Args:
            texts: 要嵌入的文档列表
            
        Returns:
            嵌入向量列表
        """
        return self.embedding.embed_documents(texts)

        # 代理方法，直接调用底层embedding模型的方法
    def embed_documents(self, text: str) -> list[list[float]]:
        """
        为文档列表生成嵌入

        Args:
            texts: 要嵌入的文档列表

        Returns:
            嵌入向量列表
        """
        return self.embedding.embed_documents(text)
    
    def embed_query(self, text: str) -> List[float]:
        """
        为查询文本生成嵌入
        
        Args:
            text: 要嵌入的查询文本
            
        Returns:
            嵌入向量
        """
        return self.embedding.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步为文档列表生成嵌入
        
        Args:
            texts: 要嵌入的文档列表
            
        Returns:
            嵌入向量列表
        """
        return await self.embedding.aembed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """
        异步为查询文本生成嵌入
        
        Args:
            text: 要嵌入的查询文本
            
        Returns:
            嵌入向量
        """
        return await self.embedding.aembed_query(text)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        获取embedding模型信息
        
        Returns:
            包含模型信息的字典
        """
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "full_name": self.full_name,
            "is_initialized": self.is_initialized,
            "embedding_type": type(self.embedding).__name__ if self._embedding_instance else None
        }
    
    def __str__(self) -> str:
        return f"ManagedEmbedding({self.full_name})"
    
    def __repr__(self) -> str:
        return self.__str__()
