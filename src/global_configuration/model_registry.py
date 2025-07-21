"""
模型注册仓库，管理所有可用的模型实例
"""
import threading
from typing import Type, Optional, Dict, Any, List
from src.model.chat.base_model import BaseManagedModel
from src.global_configuration.decorators import get_pending_registrations, clear_pending_registrations

class ModelRegistry:
    """
    模型注册仓库
    支持以litellm风格的名称注册和获取模型
    """
    
    def __init__(self):
        self._models: Dict[str, BaseManagedModel] = {}
        self._model_classes: Dict[str, Type[BaseManagedModel]] = {}
        self._lock = threading.Lock()
    
    def register(self, model_class: Type[BaseManagedModel], 
                 model_name: str, model_provider: str, 
                 alias: Optional[str] = None) -> str:
        """
        注册模型类到仓库
        
        Args:
            model_class: 模型类
            model_name: 模型名称
            model_provider: 模型提供商
            alias: 可选的别名
            
        Returns:
            注册的完整模型名称
        """
        full_name = f"{model_provider}/{model_name}"
        
        with self._lock:
            # 创建模型实例（懒加载）
            model_instance = model_class(model_name, model_provider)
            self._models[full_name] = model_instance
            self._model_classes[full_name] = model_class
            
            # 如果有别名，也注册别名
            if alias:
                self._models[alias] = model_instance
                self._model_classes[alias] = model_class
        
        return full_name
    
    def get(self, model_identifier: str) -> Optional[BaseManagedModel]:
        """
        根据模型标识符获取模型实例
        
        Args:
            model_identifier: 模型标识符（如 "azure/gpt-4o" 或 "openai/gpt-4o"）
            
        Returns:
            模型实例，如果未找到则返回None
        """
        with self._lock:
            return self._models.get(model_identifier)
    
    def list_models(self) -> List[str]:
        """
        列出所有已注册的模型
        
        Returns:
            模型标识符列表
        """
        with self._lock:
            return list(self._models.keys())
    
    def list_providers(self) -> List[str]:
        """
        列出所有支持的提供商
        
        Returns:
            提供商列表
        """
        providers = set()
        with self._lock:
            for model_id in self._models.keys():
                if "/" in model_id:
                    provider = model_id.split("/")[0]
                    providers.add(provider)
        return list(providers)
    
    def get_models_by_provider(self, provider: str) -> List[str]:
        """
        获取指定提供商的所有模型
        
        Args:
            provider: 提供商名称
            
        Returns:
            该提供商的模型列表
        """
        models = []
        with self._lock:
            for model_id in self._models.keys():
                if model_id.startswith(f"{provider}/"):
                    models.append(model_id)
        return models
    
    def remove(self, model_identifier: str) -> bool:
        """
        从仓库中移除模型
        
        Args:
            model_identifier: 模型标识符
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if model_identifier in self._models:
                del self._models[model_identifier]
                if model_identifier in self._model_classes:
                    del self._model_classes[model_identifier]
                return True
        return False
    
    def clear(self):
        """清空所有注册的模型"""
        with self._lock:
            self._models.clear()
            self._model_classes.clear()
    
    def get_model_info(self, model_identifier: str) -> Optional[Dict[str, Any]]:
        """
        获取模型信息
        
        Args:
            model_identifier: 模型标识符
            
        Returns:
            模型信息字典
        """
        model = self.get(model_identifier)
        if model:
            return model.get_model_info()
        return None
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有模型的信息
        
        Returns:
            所有模型信息的字典
        """
        info = {}
        with self._lock:
            for model_id, model in self._models.items():
                info[model_id] = model.get_model_info()
        return info


# 全局模型注册仓库实例
_global_registry = ModelRegistry()
_registry_initialized = False


def _process_pending_registrations():
    """处理所有待注册的模型"""
    global _registry_initialized
    
    if _registry_initialized:
        return
        
    try:
        pending = get_pending_registrations()
        
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
                        full_name = _global_registry.register(
                            model_class=cls,
                            model_name=model_name,
                            model_provider=model_provider,
                            alias=alias
                        )
                        print(f"✅ Registered {full_name}" + (f" (alias: {alias})" if alias else ""))
                    except Exception as e:
                        print(f"❌ Failed to register {model_provider}/{model_name}: {e}")
        
        clear_pending_registrations()
        _registry_initialized = True
    except Exception as e:
        print(f"⚠️ 延迟注册过程中发生错误: {e}")


def get_model_registry() -> ModelRegistry:
    """获取全局模型注册仓库"""
    # 只在需要时处理待注册的模型，并且确保模块已完全加载
    if not _registry_initialized:
        try:
            _process_pending_registrations()
        except:
            # 如果处理失败，忽略错误，让注册在后续调用时再次尝试
            pass
    
    return _global_registry


def register_model(model_class: Type[BaseManagedModel], 
                  model_name: str, model_provider: str, 
                  alias: Optional[str] = None) -> str:
    """
    快捷函数：注册模型到全局仓库
    
    Args:
        model_class: 模型类
        model_name: 模型名称
        model_provider: 模型提供商
        alias: 可选的别名
        
    Returns:
        注册的完整模型名称
    """
    registry = get_model_registry()  # 这会触发延迟注册
    return registry.register(model_class, model_name, model_provider, alias)


def get_model(model_identifier: str) -> Optional[BaseManagedModel]:
    """
    快捷函数：从全局仓库获取模型
    
    Args:
        model_identifier: 模型标识符
        
    Returns:
        模型实例
    """
    registry = get_model_registry()  # 这会触发延迟注册
    
    # 如果模型不存在，尝试强制处理待注册的模型
    model = registry.get(model_identifier)
    if model is None and not _registry_initialized:
        try:
            _process_pending_registrations()
            model = registry.get(model_identifier)
        except:
            # 如果仍然失败，忽略错误
            pass
    
    return model
