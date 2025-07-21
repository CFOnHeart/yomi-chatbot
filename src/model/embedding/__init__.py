"""
Embedding模块 - 提供统一的embedding模型管理
"""

# 基础模块
from .base_embedding import BaseManagedEmbedding

# 具体的embedding实现
from .azure_openai_embeddings import YomiAzureOpenAIEmbedding
from .openai_embeddings import YomiOpenAIEmbedding

# 可选的embedding实现（如果依赖可用）
try:
    from .huggingface_embeddings import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None

__all__ = [
    # 基础类和函数
    'BaseManagedEmbedding',
    # 具体实现
    'YomiAzureOpenAIEmbedding',
    'YomiOpenAIEmbedding',
    'HuggingFaceEmbedding',
]