"""
Embedding模型装饰器模块
支持embedding模型的注册和管理
"""

from typing import Type

# 存储待注册的embedding模型类，延迟注册以避免循环导入
_pending_embedding_registrations = []


def embedding_register(**register_kwargs):
    """
    装饰器，用于注册embedding模型类
    
    Args:
        **register_kwargs: 注册参数，包括：
            - models: 模型配置列表
            - provider: 默认提供商
            - auto_register: 是否自动注册
    
    Example:
        @embedding_register(
            models=[
                {"name": "text-embedding-ada-002", "alias": "ada-002"},
                {"name": "text-embedding-3-small", "alias": "ada-3-small"}
            ],
            provider="azure"
        )
        class AzureOpenAIEmbedding(BaseManagedEmbedding):
            pass
    """
    def decorator(cls):
        models = register_kwargs.get('models', [])
        provider = register_kwargs.get('provider', 'unknown')
        auto_register = register_kwargs.get('auto_register', True)
        
        # 将配置信息附加到类上
        cls._embedding_registration_config = register_kwargs
        cls._registered_embedding_models = models
        
        if auto_register:
            # 延迟注册，避免循环导入
            _pending_embedding_registrations.append({
                'cls': cls,
                'models': models,
                'provider': provider
            })
        
        return cls
    
    return decorator


def get_pending_embedding_registrations():
    """获取待注册的embedding模型列表"""
    return _pending_embedding_registrations.copy()


def clear_pending_embedding_registrations():
    """清空待注册列表"""
    _pending_embedding_registrations.clear()


def force_process_embedding_registrations():
    """强制处理所有待注册的embedding模型（需要在embedding注册仓库模块加载后调用）"""
    # 暂时不处理，让注册延迟到真正需要时
    pass
