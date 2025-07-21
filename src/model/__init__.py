"""
模型自动注册模块
导入所有模型类会自动触发装饰器注册
"""

# 导入所有模型类，这会自动触发装饰器执行模型注册
from src.model.azure_openai_model import AzureOpenAIModel
from src.model.openai_model import OpenAIModel

# 导出主要接口
__all__ = [
    'AzureOpenAIModel',
    'OpenAIModel'
]