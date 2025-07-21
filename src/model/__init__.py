"""
模型自动注册模块
导入所有模型类会自动触发装饰器注册
"""

# 导入所有模型类，这会自动触发装饰器执行模型注册
from src.model.chat.azure_openai_model import AzureOpenAIModel
from src.model.chat.openai_model import OpenAIModel

# 导入扩展模型以触发注册
try:
    from src.model.chat.extended_models import AnthropicModel, GoogleModel
    _EXTENDED_MODELS_AVAILABLE = True
except ImportError:
    _EXTENDED_MODELS_AVAILABLE = False

# 导出主要接口
__all__ = [
    'AzureOpenAIModel',
    'OpenAIModel'
]

# 如果扩展模型可用，也导出它们
if _EXTENDED_MODELS_AVAILABLE:
    __all__.extend(['AnthropicModel', 'GoogleModel'])

# 模块加载完成后，强制处理待注册的模型
from src.global_configuration.decorators import force_process_registrations
force_process_registrations()