"""
模型注册仓库，管理所有可用的模型实例
"""
import threading
from typing import Type, Optional, Dict, Any, List
from src.model.base_model import BaseManagedModel

# define decorator for model registration
def model_register(**register_kwargs):
    """
    装饰器，用于注册模型类
    
    Args:
        **register_kwargs: 注册参数，包括：
            - models: 模型配置列表
            - provider: 默认提供商
            - auto_register: 是否自动注册
    
    Example:
        @model_register(
            models=[
                {"name": "gpt-4o", "alias": "gpt4"},
                {"name": "gpt-4", "alias": "gpt4-legacy"}
            ],
            provider="azure"
        )
        class AzureOpenAIModel(BaseManagedModel):
            pass
    """
    def decorator(cls: Type[BaseManagedModel]):
        models = register_kwargs.get('models', [])
        provider = register_kwargs.get('provider', 'unknown')
        auto_register = register_kwargs.get('auto_register', True)
        
        if auto_register:
            for model_config in models:
                model_name = model_config.get('name')
                alias = model_config.get('alias')
                model_provider = model_config.get('provider', provider)
                
                if model_name:
                    try:
                        registry = get_model_registry()
                        full_name = registry.register(
                            model_class=cls,
                            model_name=model_name,
                            model_provider=model_provider,
                            alias=alias
                        )
                        print(f"✅ Registered {full_name}" + (f" (alias: {alias})" if alias else ""))
                    except Exception as e:
                        print(f"❌ Failed to register {model_provider}/{model_name}: {e}")
        
        # 将配置信息附加到类上
        cls._registration_config = register_kwargs
        cls._registered_models = models
        
        return cls
    
    return decorator

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


def get_model_registry() -> ModelRegistry:
    """获取全局模型注册仓库"""
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
    return _global_registry.register(model_class, model_name, model_provider, alias)


def get_model(model_identifier: str) -> Optional[BaseManagedModel]:
    """
    快捷函数：从全局仓库获取模型
    
    Args:
        model_identifier: 模型标识符
        
    Returns:
        模型实例
    """
    return _global_registry.get(model_identifier)
