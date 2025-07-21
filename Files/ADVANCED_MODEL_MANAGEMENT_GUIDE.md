# 高级模型管理系统使用指南

## 概述

这个新的模型管理系统使用装饰器和属性注册的方式来自动管理模型，让添加新模型变得非常简单和优雅。

## 核心特性

1. **装饰器自动注册** - 使用 `@model_register` 装饰器自动注册模型
2. **懒加载** - 模型只在需要时才初始化
3. **类型安全** - 完整的类型注解支持
4. **配置化** - 通过环境变量和配置文件管理默认模型
5. **向后兼容** - 与现有代码完全兼容
6. **扩展性强** - 轻松添加新的模型提供商

## 快速开始

### 1. 使用已注册的模型

```python
from src.model.model_registry import get_model

# 获取 Azure GPT-4o 模型
azure_model = get_model("azure/gpt-4o")
response = azure_model.invoke("Hello!")

# 使用别名
gpt4_model = get_model("azure-gpt4")  # 等同于 azure/gpt-4o

# 获取 OpenAI 模型
openai_model = get_model("openai/gpt-4o")
```

### 2. 使用配置的默认模型

```python
from src.model.model_config import get_default_model

# 获取配置的默认模型
model = get_default_model()
response = model.invoke("Hello!")
```

### 3. 在现有代码中使用

```python
from src.config.settings import get_llm_model

# 这会自动使用新的模型管理系统
llm = get_llm_model()
response = llm.invoke("Hello!")
```

## 添加新模型

### 方法1：使用装饰器（推荐）

```python
from src.model.chat.base_model import BaseManagedModel
from src.model.decorators import model_register, AutoRegisterModelMixin


@model_register(
    models=[
        {"name": "claude-3-5-sonnet", "alias": "claude-sonnet"},
        {"name": "claude-3-haiku", "alias": "claude-haiku"}
    ],
    provider="anthropic"
)
class AnthropicModel(BaseManagedModel, AutoRegisterModelMixin):
    def _create_model(self):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=self.model_name,
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
```

### 方法2：使用便捷装饰器

```python
from src.model.decorators import azure_model_register

@azure_model_register(
    {"name": "gpt-4o", "alias": "gpt4"},
    {"name": "gpt-3.5-turbo", "alias": "gpt35"}
)
class MyAzureModel(BaseManagedModel):
    # ... 实现 ...
```

### 方法3：手动注册

```python
from src.model.model_registry import register_model

class MyCustomModel(BaseManagedModel):
    # ... 实现 ...

# 手动注册
register_model(
    model_class=MyCustomModel,
    model_name="my-model",
    model_provider="custom",
    alias="my-alias"
)
```

## 配置管理

### 环境变量

```bash
# 设置默认模型
export DEFAULT_MODEL="azure/gpt-4o"

# 设置备用模型
export FALLBACK_MODEL="openai/gpt-4o"

# 启用注册摘要打印
export PRINT_MODEL_REGISTRATION="true"
```

### 代码配置

```python
from src.model.model_config import set_default_model, set_fallback_model

# 设置默认模型
set_default_model("openai/gpt-4o")

# 设置备用模型
set_fallback_model("azure/gpt-4o")
```

## 模型管理工具

使用命令行工具管理模型：

```bash
# 列出所有可用模型
python model_manager.py list

# 显示当前配置
python model_manager.py config

# 测试特定模型
python model_manager.py test azure/gpt-4o

# 设置默认模型
python model_manager.py set-default openai/gpt-4o

# 设置备用模型
python model_manager.py set-fallback azure/gpt-4o
```

## 高级用法

### 批量注册

```python
class MyModel(BaseManagedModel, AutoRegisterModelMixin):
    pass

# 批量注册多个配置
MyModel.register_multiple([
    {"name": "model1", "provider": "provider1", "alias": "alias1"},
    {"name": "model2", "provider": "provider1", "alias": "alias2"}
])
```

### 运行时注册

```python
from src.model.model_registry import get_model_registry

registry = get_model_registry()

# 运行时注册新模型
registry.register(
    model_class=MyModel,
    model_name="runtime-model",
    model_provider="runtime",
    alias="rt-model"
)
```

### 模型信息查询

```python
from src.model.model_registry import get_model_registry

registry = get_model_registry()

# 列出所有模型
models = registry.list_models()

# 列出所有提供商
providers = registry.list_providers()

# 获取特定提供商的模型
azure_models = registry.get_models_by_provider("azure")

# 获取模型详细信息
model_info = registry.get_model_info("azure/gpt-4o")
```

## 最佳实践

1. **使用装饰器** - 推荐使用装饰器自动注册，代码更简洁
2. **合理使用别名** - 为常用模型设置简短的别名
3. **环境变量配置** - 使用环境变量管理不同环境的模型配置
4. **错误处理** - 实现适当的错误处理和备用方案
5. **文档化** - 为自定义模型添加适当的文档

## 迁移指南

从旧系统迁移到新系统：

1. **保持兼容性** - 现有代码无需修改，仍然可以正常工作
2. **逐步迁移** - 可以逐步将代码迁移到使用新的模型标识符
3. **测试验证** - 使用 `test_new_model_system.py` 验证迁移结果

## 故障排除

### 常见问题

1. **模型未找到** - 检查模型是否已正确注册
2. **初始化失败** - 检查环境变量和依赖是否正确配置
3. **导入错误** - 确保所有必要的包已安装

### 调试工具

```python
# 检查注册状态
from src.model import print_registration_summary
print_registration_summary()

# 获取详细信息
from src.model.model_registry import get_model_registry
registry = get_model_registry()
print(registry.get_all_models_info())
```
