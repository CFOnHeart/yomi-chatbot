# LangGraph 对话Agent系统

一个完整的基于LangGraph的对话Agent系统，支持历史记录持久化、智能摘要、工具调用等功能。

## 🚀 功能特性

### 1. 数据库持久化
- 使用SQLite数据库存储聊天记录
- 支持会话管理和历史记录查询
- 自动创建数据库表结构

### 2. 智能记忆管理
- 基于RunnableWithMessageHistory的历史记录管理
- 自动检测文本长度，超过3200字符自动摘要
- 摘要功能减少Token消耗，提高响应速度

### 3. 工具调用系统
- 自动检测用户输入是否需要调用工具
- 支持多种工具：数学计算、人工协助等
- 用户确认机制，展示工具Schema后询问是否执行

### 4. LangGraph工作流
- 完整的对话处理流程
- 支持条件路由和错误处理
- 异步处理和状态管理

### 5. 会话管理
- 支持多会话并发
- 会话初始化和历史记录加载
- 会话信息查询和管理

## 📁 项目结构

```
src/
├── database/           # 数据库模块
│   ├── __init__.py
│   └── chat_db.py     # 聊天数据库管理
├── memory/            # 记忆管理模块
│   ├── __init__.py
│   ├── conversation_memory.py  # 原有记忆管理
│   └── smart_memory_manager.py # 智能记忆管理器
├── tools/             # 工具模块
│   ├── __init__.py
│   ├── tool_manager.py # 工具管理器
│   └── simple/        # 简单工具
│       ├── __init__.py
│       ├── math.py    # 数学工具
│       └── human_assistance.py # 人工协助工具
├── graph/             # LangGraph模块
│   ├── __init__.py
│   ├── state.py       # 状态定义
│   ├── nodes.py       # 工作流节点
│   └── conversation_agent.py # 主Agent
├── config/            # 配置模块
│   ├── __init__.py
│   └── agent_config.py # Agent配置
├── examples/          # 示例代码
│   ├── __init__.py
│   └── run_langgraph_conversation_agent.py # 运行示例
└── tests/             # 测试模块
    ├── __init__.py
    └── test_agent.py  # 测试用例
```

## 🛠️ 安装和配置

### 1. 环境要求
- Python 3.13+
- 依赖包：见pyproject.toml

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 环境变量配置
创建`.env`文件：
```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2024-02-01
```

## 🚀 使用指南

### 1. 基本使用

```python
from src.agent.conversation_agent import create_agent

# 创建Agent实例
agent = create_agent()

# 开始对话
session_id = "my_session"
response = agent.chat(session_id, "你好！")
print(response)
```

### 2. 运行示例

```bash
# 交互式对话
python src/examples/run_langgraph_conversation_agent.py

# 演示对话
python src/examples/run_langgraph_conversation_agent.py demo
```

### 3. 会话管理

```python
# 获取会话信息
info = agent.get_session_info(session_id)
print(f"消息数量: {info['message_count']}")
print(f"文本长度: {info['text_length']}")

# 列出所有会话
sessions = agent.list_sessions()
for session in sessions:
    print(f"会话: {session['session_id']}")

# 删除会话
agent.delete_session(session_id)
```

## 🔧 核心组件说明

### 1. ChatDatabase
- 数据库操作类
- 支持会话和消息的CRUD操作
- 自动创建表结构和索引

### 2. SmartMemoryManager
- 智能记忆管理器
- 自动摘要长对话
- 基于RunnableWithMessageHistory

### 3. ToolMatcher & ToolConfirmationSystem
- 工具检测和确认系统
- LLM驱动的工具需求检测
- 用户交互确认机制

### 4. ConversationAgent
- 主要的对话Agent
- 基于LangGraph的工作流
- 支持多种节点和路由

## 🔄 工作流程

1. **会话初始化**: 检查会话是否存在，加载历史记录
2. **用户输入保存**: 将用户输入保存到数据库
3. **工具检测**: 使用LLM检测是否需要调用工具
4. **工具执行**: 如果需要，展示工具信息并确认执行
5. **LLM响应**: 如果不需要工具，直接使用LLM回答
6. **响应保存**: 将最终响应保存到数据库
7. **异步摘要**: 检查文本长度，必要时进行摘要

## 🧪 测试

```bash
# 运行测试
python src/tests/test_agent.py
```

## 📊 监控和调试

系统提供了详细的日志输出：
- ✅ 成功操作
- ❌ 错误信息
- 🔧 工具检测
- 💾 数据保存
- 🤖 LLM响应

## 🎯 使用示例

### 基本对话
```
👤 用户: 你好！我是张三
🤖 助手: 你好张三！很高兴认识你。有什么我可以帮助你的吗？
```

### 数学计算
```
👤 用户: 请计算 15 + 25
🔧 检测到可用工具: add
📝 工具描述: Add two integers
是否执行此工具? (y/n): y
✅ 工具执行成功: 40
🤖 助手: 工具执行结果: 40
```

### 人工协助
```
👤 用户: 我需要帮助解决一个复杂问题
🔧 检测到可用工具: human_assistance
📝 工具描述: Request assistance from a human
是否执行此工具? (y/n): y
🙋‍♂️ 人工协助请求:
📝 问题: 我需要帮助解决一个复杂问题
💭 请提供您的回答: 这是一个复杂问题，需要更多信息
✅ 工具执行成功: 这是一个复杂问题，需要更多信息
```

## 🔧 自定义配置

可以通过修改`src/config/agent_config.py`来自定义配置：

```python
# 调整Token限制
MAX_TOKENS = 3200

# 调整工具置信度阈值
TOOL_CONFIDENCE_THRESHOLD = 0.7

# 自定义系统消息
DEFAULT_SYSTEM_MESSAGE = "你是一个专业的AI助手..."
```

## 🚀 扩展功能

### 添加新工具
1. 在`src/tools/simple/`下创建新的工具文件
2. 使用`@tool`装饰器定义工具
3. 在`tool_manager.py`中注册新工具

### 添加新节点
1. 在`src/graph/nodes.py`中添加新的节点函数
2. 在`conversation_agent.py`中注册节点和路由

### 自定义数据库
1. 修改`src/database/chat_db.py`
2. 添加新的表结构和操作方法

## 📝 注意事项

1. 确保环境变量正确配置
2. 数据库文件会自动创建在项目根目录
3. 长时间运行可能需要定期清理数据库
4. 工具执行需要用户确认，确保安全性

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证
