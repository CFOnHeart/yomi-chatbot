# Embedding模块化系统使用指南

## 概述

这个embedding模块化系统模拟了chat model的架构设计，提供了统一的embedding模型管理和使用接口。系统支持多种embedding提供商，包括Azure OpenAI、OpenAI和HuggingFace。

## 系统架构

### 核心组件

1. **BaseManagedEmbedding**: 抽象基类，定义统一的embedding接口
2. **EmbeddingRegistry**: 模型注册仓库，管理所有embedding模型实例
3. **EmbeddingManager**: 高级管理器，提供便捷的操作接口
4. **embedding_register**: 装饰器，用于自动注册embedding模型

### 文件结构

```
src/model/embedding/
├── __init__.py                    # 模块导出
├── base_embedding.py              # 抽象基类
├── embedding_registry.py          # 注册仓库
├── embedding_decorators.py        # 装饰器系统
├── embedding_manager.py           # 管理器
├── azure_openai_embeddings.py     # Azure OpenAI实现
├── openai_embeddings.py           # OpenAI实现
└── huggingface_embeddings.py      # HuggingFace实现
```

## 快速开始

### 1. 基本使用

```python
from src.model.embedding import get_embedding_manager

# 获取管理器
manager = get_embedding_manager()

# 使用默认模型进行嵌入
texts = ["Hello world", "Python programming"]
embeddings = manager.embed_documents(texts)

query = "machine learning"
query_embedding = manager.embed_query(query)
```

### 2. 指定模型使用

```python
from src.model.embedding import get_embedding

# 通过完整标识符获取模型
embedding = get_embedding("azure/text-embedding-ada-002")

# 通过别名获取模型
embedding = get_embedding("ada-002")

if embedding:
    result = embedding.embed_query("test query")
```

### 3. 异步使用

```python
import asyncio
from src.model.embedding import get_embedding_manager

async def async_embedding_example():
    manager = get_embedding_manager()
    
    # 异步文档嵌入
    texts = ["Document 1", "Document 2"]
    doc_embeddings = await manager.aembed_documents(texts)
    
    # 异步查询嵌入
    query_embedding = await manager.aembed_query("search query")
    
    return doc_embeddings, query_embedding

# 运行异步函数
embeddings = asyncio.run(async_embedding_example())
```

## 创建新的Embedding模型

### 1. 实现BaseManagedEmbedding

```python
from src.model.embedding.base_embedding import BaseManagedEmbedding
from src.model.embedding.embedding_decorators import embedding_register

@embedding_register(
    models=[
        {"name": "your-model-name", "alias": "your-alias"}
    ],
    provider="your-provider"
)
class YourEmbedding(BaseManagedEmbedding):
    """你的embedding模型实现"""
    
    def _create_embedding(self) -> Embeddings:
        """实现这个方法来创建具体的embedding实例"""
        # 在这里创建和配置你的embedding模型
        # 例如：
        return YourEmbeddingClass(
            model=self.model_name,
            # 其他配置参数
        )
```

### 2. 环境配置

根据你的embedding提供商，配置相应的环境变量：

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=your_version

# OpenAI
OPENAI_API_KEY=your_api_key

# HuggingFace (可选)
HUGGINGFACE_API_TOKEN=your_token
```

## 模型注册和管理

### 装饰器注册

使用`@embedding_register`装饰器自动注册模型：

```python
@embedding_register(
    models=[
        {"name": "text-embedding-3-small", "alias": "openai-small"},
        {"name": "text-embedding-3-large", "alias": "openai-large"}
    ],
    provider="openai"
)
class OpenAIEmbedding(BaseManagedEmbedding):
    # 实现细节...
```

### 手动注册

也可以手动注册模型：

```python
from src.model.embedding import register_embedding

register_embedding(
    embedding_class=YourEmbedding,
    model_name="your-model",
    model_provider="your-provider",
    alias="your-alias"
)
```

## 管理器功能

### 查看系统状态

```python
from src.model.embedding import get_embedding_manager

manager = get_embedding_manager()

# 显示详细摘要
manager.print_summary()

# 列出所有可用模型
available_models = manager.list_available_embeddings()

# 按提供商列出
azure_models = manager.get_embeddings_by_provider("azure")
```

### 模型信息查询

```python
# 获取单个模型信息
info = manager.get_embedding_info("azure/text-embedding-ada-002")

# 获取所有模型信息
all_info = manager.get_all_embeddings_info()
```

## 支持的Embedding提供商

### 1. Azure OpenAI

```python
# 可用模型
- azure/text-embedding-ada-002 (别名: ada-002)
- azure/text-embedding-3-small (别名: ada-3-small)
- azure/text-embedding-3-large (别名: ada-3-large)

# 所需环境变量
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_API_KEY (或使用 az login)
```

### 2. OpenAI

```python
# 可用模型
- openai/text-embedding-3-small (别名: openai-small)
- openai/text-embedding-3-large (别名: openai-large)
- openai/text-embedding-ada-002 (别名: openai-ada)

# 所需环境变量
OPENAI_API_KEY
```

### 3. HuggingFace

```python
# 可用模型
- huggingface/sentence-transformers/all-MiniLM-L6-v2 (别名: minilm)
- huggingface/sentence-transformers/all-mpnet-base-v2 (别名: mpnet)
- huggingface/BAAI/bge-large-en-v1.5 (别名: bge-large)

# 可选环境变量
HUGGINGFACE_API_TOKEN

# 所需依赖
pip install langchain-huggingface sentence-transformers
```

## 错误处理

系统提供了完善的错误处理机制：

```python
from src.model.embedding import get_embedding

try:
    embedding = get_embedding("non-existent-model")
    if embedding is None:
        print("模型不存在")
    else:
        result = embedding.embed_query("test")
except ValueError as e:
    print(f"配置错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
```

## 最佳实践

### 1. 懒加载

所有embedding模型都采用懒加载机制，只有在实际使用时才会初始化：

```python
# 获取模型但不立即初始化
embedding = get_embedding("azure/text-embedding-ada-002")

# 第一次调用时才会初始化
result = embedding.embed_query("first query")  # 初始化发生在这里
```

### 2. 默认模型管理

建议设置一个默认的embedding模型：

```python
manager = get_embedding_manager()
manager.set_default_embedding("azure/text-embedding-ada-002")

# 后续可以直接使用，无需指定模型
embeddings = manager.embed_documents(["doc1", "doc2"])
```

### 3. 批量处理

对于大量文档，使用批量处理更高效：

```python
# 好的做法
large_document_list = ["doc1", "doc2", ..., "doc1000"]
embeddings = manager.embed_documents(large_document_list)

# 避免循环调用
# for doc in large_document_list:
#     embedding = manager.embed_query(doc)  # 避免这样做
```

## 测试和调试

### 运行演示脚本

```bash
cd yomi-chatbot
python demo_embedding_system.py
```

### 运行测试

```bash
python test_embedding_system.py
```

### 调试技巧

1. 启用详细日志输出
2. 检查环境变量配置
3. 使用`print_summary()`查看系统状态
4. 验证模型初始化状态

## 扩展和自定义

### 添加新的提供商

1. 创建继承自`BaseManagedEmbedding`的新类
2. 实现`_create_embedding()`方法
3. 使用`@embedding_register`装饰器注册
4. 添加相应的依赖和环境变量

### 自定义管理器

```python
from src.model.embedding.embedding_manager import EmbeddingManager

class CustomEmbeddingManager(EmbeddingManager):
    def custom_method(self):
        # 添加自定义功能
        pass
```

## 与Chat Model系统的对比

本embedding系统的设计完全模仿了chat model的架构：

| Chat Model | Embedding Model | 说明 |
|------------|----------------|------|
| BaseManagedModel | BaseManagedEmbedding | 抽象基类 |
| ModelRegistry | EmbeddingRegistry | 注册仓库 |
| model_register | embedding_register | 装饰器 |
| get_model | get_embedding | 获取模型 |

这种一致性确保了系统的可维护性和开发者的学习成本最小化。
