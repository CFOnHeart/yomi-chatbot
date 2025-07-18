# Multi-Agent 系统设计文档

## 概述

本项目实现了一个基于抽象类设计的Multi-Agent协作系统，支持层级化的Agent管理和任务协调。

## 架构设计

### 抽象基类

#### 1. AbstractAgent
所有Agent的基类，定义了统一的接口：
- `invoke(query, context)`: 执行Agent的核心逻辑

#### 2. AbstractManagedAgent
可被管理的Agent基类，继承自AbstractAgent：
- 包含描述信息用于Supervisor的决策
- 支持设置Supervisor
- 支持管理子Agent（可形成多层级结构）

#### 3. AbstractSupervisorAgent
主管Agent基类，负责协调和管理子Agent：
- 注册和管理子Agent
- 分解任务和委派执行
- 汇总最终结果

## 具体实现

### 1. SupervisorAgent
主管Agent的具体实现，工作流程：
1. **任务分解**: 分析用户查询，分解为多个子任务
2. **Agent选择**: 根据子Agent描述选择最适合的Agent
3. **任务执行**: 委派给子Agent或自己执行
4. **结果汇总**: 整合所有子任务结果

### 2. ConversationAgent
对话Agent，继承自AbstractManagedAgent：
- 基于LangGraph的对话流程
- 支持RAG文档检索
- 支持工具调用
- 管理聊天历史

### 3. DocumentAgent
文档管理Agent，专门处理文档相关操作：
- 文档上传和管理
- 文档搜索和检索
- 文档摘要生成
- 文档深度分析

### 4. ToolAgent
工具管理Agent，处理工具调用和系统操作：
- 工具检测和调用
- 系统操作管理
- 数据处理任务

## 使用示例

### 创建Multi-Agent系统

```python
from src.agent.supervisor_agent import SupervisorAgent
from src.agent.conversation_agent import ConversationAgent
from src.agent.document_agent import DocumentAgent
from src.agent.tool_agent import ToolAgent

# 创建各种Agent
conversation_agent = ConversationAgent()
document_agent = DocumentAgent()
tool_agent = ToolAgent()

# 创建Supervisor并注册子Agent
supervisor = SupervisorAgent()
supervisor.register_agent(conversation_agent)
supervisor.register_agent(document_agent)
supervisor.register_agent(tool_agent)

# 使用系统
result = supervisor.invoke("帮我搜索关于AI的文档并总结")
```

### 层级化Agent结构

```python
# ManagedAgent也可以管理子Agent
document_agent.register_sub_agent(some_sub_agent)

# 形成多层级结构
# SupervisorAgent -> DocumentAgent -> SubAgent
```

## 工作流程

### 1. 用户请求处理流程

```
用户输入 → SupervisorAgent → 任务分解 → Agent选择 → 任务执行 → 结果汇总 → 返回结果
```

### 2. 任务分解机制

SupervisorAgent使用`supervisor_plan` prompt模板分析用户查询：
- 确定是否需要分解为多个步骤
- 生成具体的任务列表
- 考虑任务间的依赖关系

### 3. Agent选择机制

使用`supervisor_delegate` prompt模板进行决策：
- 分析当前任务需求
- 匹配可用Agent的描述
- 选择最合适的Agent
- 生成具体的任务输入

### 4. 上下文管理

- 保持完整的对话上下文
- 在任务间传递中间结果
- 支持跨Agent的信息共享

## Prompt模板

系统使用以下prompt模板支持Supervisor的决策：

- `supervisor_plan`: 任务分解
- `supervisor_delegate`: Agent选择和任务委派
- `supervisor_self_execute`: Supervisor自主执行
- `supervisor_summarize`: 结果汇总

## 扩展指南

### 添加新的ManagedAgent

1. 继承`AbstractManagedAgent`
2. 在`__init__`中设置描述信息
3. 实现`invoke`方法
4. 注册到SupervisorAgent

```python
class CustomAgent(AbstractManagedAgent):
    def __init__(self):
        super().__init__(
            description="自定义Agent的功能描述"
        )
    
    def invoke(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        # 实现具体逻辑
        pass
```

### 创建中层管理Agent

```python
class ManagerAgent(AbstractManagedAgent):
    def __init__(self):
        super().__init__(description="中层管理Agent")
        # 可以注册自己的子Agent
        
    def invoke(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        # 实现管理逻辑，可以进一步委派给子Agent
        pass
```

## 运行演示

```bash
# 运行演示程序
python multi_agent_demo.py

# 选择演示模式：
# 1. 自动演示 (预设测试用例)
# 2. 交互式演示 (手动输入)
```

## 配置说明

### Prompt配置
在`src/config/prompts/prompts.yaml`中配置Supervisor相关的prompt模板。

### Agent描述
每个ManagedAgent在创建时必须提供详细的描述信息，这些描述会被SupervisorAgent用于决策。

## 优势特点

1. **模块化设计**: 每个Agent职责单一，易于维护
2. **可扩展性**: 支持添加新Agent和多层级结构
3. **智能调度**: 基于描述的智能Agent选择
4. **上下文保持**: 完整的对话上下文管理
5. **容错处理**: 多级失败处理机制

## 注意事项

1. Agent描述信息要准确详细，影响Supervisor的决策
2. 任务分解的granularity要合适，避免过度分解
3. 上下文信息可能会很长，注意token限制
4. 异常处理要完善，确保系统稳定性

## 未来扩展

1. 支持动态Agent注册和发现
2. 增加Agent性能监控和优化
3. 支持分布式Agent部署
4. 增加Agent间的直接通信机制
