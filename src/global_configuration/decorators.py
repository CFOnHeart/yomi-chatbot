"""
模型装饰器模块
将装饰器与注册仓库分离以避免循环导入
"""

from typing import Type

# 存储待注册的模型类，延迟注册以避免循环导入
_pending_registrations = []

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
    def decorator(cls):
        models = register_kwargs.get('models', [])
        provider = register_kwargs.get('provider', 'unknown')
        auto_register = register_kwargs.get('auto_register', True)
        
        # 将配置信息附加到类上
        cls._registration_config = register_kwargs
        cls._registered_models = models
        
        if auto_register:
            # 延迟注册，避免循环导入
            _pending_registrations.append({
                'cls': cls,
                'models': models,
                'provider': provider
            })
        
        return cls
    
    return decorator


def get_pending_registrations():
    """获取待注册的模型列表"""
    return _pending_registrations.copy()


def clear_pending_registrations():
    """清空待注册列表"""
    _pending_registrations.clear()


def force_process_registrations():
    """强制处理所有待注册的模型（需要在模型注册仓库模块加载后调用）"""
    # 暂时不处理，让注册延迟到真正需要时
    pass
