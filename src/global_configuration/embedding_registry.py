"""
Embedding模型注册仓库，管理所有可用的embedding模型实例
"""
import threading
from typing import Type, Optional, Dict, Any, List
from src.model.embedding.base_embedding import BaseManagedEmbedding
from src.global_configuration.embedding_decorators import (
    get_pending_embedding_registrations,
    clear_pending_embedding_registrations
)

class EmbeddingRegistry:
    """
    Embedding模型注册仓库
    支持以统一风格的名称注册和获取embedding模型
    """
    
    def __init__(self):
        self._embeddings: Dict[str, BaseManagedEmbedding] = {}
        self._embedding_classes: Dict[str, Type[BaseManagedEmbedding]] = {}
        self._lock = threading.Lock()
    
    def register(self, embedding_class: Type[BaseManagedEmbedding], 
                 model_name: str, model_provider: str, 
                 alias: Optional[str] = None) -> str:
        """
        注册embedding模型类到仓库
        
        Args:
            embedding_class: embedding模型类
            model_name: 模型名称
            model_provider: 模型提供商
            alias: 可选的别名
            
        Returns:
            注册的完整模型名称
        """
        full_name = f"{model_provider}/{model_name}"
        
        with self._lock:
            # 创建模型实例（懒加载）
            embedding_instance = embedding_class(model_name, model_provider)
            self._embeddings[full_name] = embedding_instance
            self._embedding_classes[full_name] = embedding_class
            
            # 如果有别名，也注册别名
            if alias:
                self._embeddings[alias] = embedding_instance
                self._embedding_classes[alias] = embedding_class
        
        return full_name
    
    def get(self, model_identifier: str) -> Optional[BaseManagedEmbedding]:
        """
        根据模型标识符获取embedding模型实例
        
        Args:
            model_identifier: 模型标识符（如 "azure/text-embedding-ada-002"）
            
        Returns:
            embedding模型实例，如果未找到则返回None
        """
        with self._lock:
            return self._embeddings.get(model_identifier)
    
    def list_embeddings(self) -> List[str]:
        """
        列出所有已注册的embedding模型
        
        Returns:
            模型标识符列表
        """
        with self._lock:
            return list(self._embeddings.keys())
    
    def list_providers(self) -> List[str]:
        """
        列出所有支持的提供商
        
        Returns:
            提供商列表
        """
        providers = set()
        with self._lock:
            for model_id in self._embeddings.keys():
                if "/" in model_id:
                    provider = model_id.split("/")[0]
                    providers.add(provider)
        return list(providers)
    
    def get_embeddings_by_provider(self, provider: str) -> List[str]:
        """
        获取指定提供商的所有embedding模型
        
        Args:
            provider: 提供商名称
            
        Returns:
            该提供商的模型列表
        """
        models = []
        with self._lock:
            for model_id in self._embeddings.keys():
                if model_id.startswith(f"{provider}/"):
                    models.append(model_id)
        return models
    
    def remove(self, model_identifier: str) -> bool:
        """
        从仓库中移除embedding模型
        
        Args:
            model_identifier: 模型标识符
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if model_identifier in self._embeddings:
                del self._embeddings[model_identifier]
                if model_identifier in self._embedding_classes:
                    del self._embedding_classes[model_identifier]
                return True
        return False
    
    def clear(self):
        """清空所有注册的embedding模型"""
        with self._lock:
            self._embeddings.clear()
            self._embedding_classes.clear()
    
    def get_embedding_info(self, model_identifier: str) -> Optional[Dict[str, Any]]:
        """
        获取embedding模型信息
        
        Args:
            model_identifier: 模型标识符
            
        Returns:
            模型信息字典
        """
        embedding = self.get(model_identifier)
        if embedding:
            return embedding.get_embedding_info()
        return None
    
    def get_all_embeddings_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有embedding模型的信息
        
        Returns:
            所有模型信息的字典
        """
        info = {}
        with self._lock:
            for model_id, embedding in self._embeddings.items():
                info[model_id] = embedding.get_embedding_info()
        return info


# 全局embedding注册仓库实例
_global_embedding_registry = EmbeddingRegistry()
_embedding_registry_initialized = False


def _process_pending_embedding_registrations():
    """处理所有待注册的embedding模型"""
    global _embedding_registry_initialized
    
    if _embedding_registry_initialized:
        return
        
    try:
        pending = get_pending_embedding_registrations()
        
        for registration in pending:
            cls = registration['cls']
            models = registration['models']
            provider = registration['provider']
            
            for model_config in models:
                model_name = model_config.get('name')
                alias = model_config.get('alias')
                model_provider = model_config.get('provider', provider)
                
                if model_name:
                    try:
                        full_name = _global_embedding_registry.register(
                            embedding_class=cls,
                            model_name=model_name,
                            model_provider=model_provider,
                            alias=alias
                        )
                        print(f"✅ Registered embedding {full_name}" + (f" (alias: {alias})" if alias else ""))
                    except Exception as e:
                        print(f"❌ Failed to register embedding {model_provider}/{model_name}: {e}")
        
        clear_pending_embedding_registrations()
        _embedding_registry_initialized = True
    except Exception as e:
        print(f"⚠️ Embedding延迟注册过程中发生错误: {e}")


def get_embedding_registry() -> EmbeddingRegistry:
    """获取全局embedding注册仓库"""
    if not _embedding_registry_initialized:
        try:
            _process_pending_embedding_registrations()
        except:
            # 如果处理失败，忽略错误，让注册在后续调用时再次尝试
            pass
    
    return _global_embedding_registry


def register_embedding(embedding_class: Type[BaseManagedEmbedding], 
                      model_name: str, model_provider: str, 
                      alias: Optional[str] = None) -> str:
    """
    快捷函数：注册embedding模型到全局仓库
    
    Args:
        embedding_class: embedding模型类
        model_name: 模型名称
        model_provider: 模型提供商
        alias: 可选的别名
        
    Returns:
        注册的完整模型名称
    """
    registry = get_embedding_registry()
    return registry.register(embedding_class, model_name, model_provider, alias)


def get_embedding(model_identifier: str) -> Optional[BaseManagedEmbedding]:
    """
    快捷函数：从全局仓库获取embedding模型
    
    Args:
        model_identifier: 模型标识符
        
    Returns:
        embedding模型实例
    """
    registry = get_embedding_registry()
    
    # 如果模型不存在，尝试强制处理待注册的模型
    embedding = registry.get(model_identifier)
    if embedding is None and not _embedding_registry_initialized:
        try:
            _process_pending_embedding_registrations()
            embedding = registry.get(model_identifier)
        except:
            # 如果仍然失败，忽略错误
            pass
    
    return embedding
