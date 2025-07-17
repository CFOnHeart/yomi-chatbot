# RAG系统升级说明

## 🚀 新功能概述

本次更新为yomi-chatbot项目添加了完整的RAG（检索增强生成）系统，并使用FAISS进行高效的向量搜索。

## 📋 主要特性

### 1. FAISS向量数据库
- **高性能向量搜索**：使用Facebook的FAISS库进行高效的相似度搜索
- **持久化存储**：向量索引自动保存和加载
- **混合搜索**：结合语义搜索和关键词搜索
- **扩展性好**：支持大规模文档集合

### 2. 完整的RAG流程
- **文档存储**：支持多种格式文档的存储和索引
- **智能检索**：基于用户query自动检索相关文档
- **上下文增强**：将检索到的文档作为上下文传递给LLM
- **引用追踪**：在回答中显示文档来源和位置信息

### 3. 工作流集成
- **无缝集成**：RAG检索已集成到Agent工作流中
- **智能路由**：先检查工具调用，再进行文档检索
- **降级处理**：如果RAG失败，自动降级到普通LLM回答

## 🏗️ 架构设计

### 数据库层
```
src/database/
├── chat_db.py          # 聊天记录数据库
├── document_db.py      # 传统文档数据库（已弃用）
└── faiss_document_db.py # FAISS向量数据库
```

### RAG系统
```
src/rag/
├── __init__.py
└── rag_system.py       # RAG核心系统
```

### 工作流更新
```
初始化会话 → 保存用户输入 → 工具检测 → 工具执行/RAG检索 → LLM生成 → 最终响应
```

## 🔧 安装和配置

### 1. 安装依赖
```bash
pip install faiss-cpu numpy
```

### 2. 验证安装
```bash
python test_faiss.py
```

### 3. 基本使用
```python
from src.agent.conversation_agent import create_agent

# 创建Agent
agent = create_agent()

# 添加文档
doc_id = agent.add_document(
    title="Python教程",
    content="Python是一种编程语言...",
    category="programming"
)

# 进行对话（自动使用RAG）
response = agent.chat("session1", "如何学习Python？")
```

## 📊 性能优化

### 向量搜索性能
- **索引类型**：使用`IndexFlatL2`进行精确搜索
- **内存优化**：向量索引按需加载
- **批量处理**：支持批量添加文档

### 搜索策略
- **语义搜索**：基于embedding相似度
- **关键词搜索**：传统全文搜索
- **混合搜索**：结合两种方法的优势

## 🛠️ 管理工具

### RAG管理器
```bash
# 添加单个文件
python rag_manager.py add-file document.txt --category tech

# 批量添加目录
python rag_manager.py add-dir ./docs --pattern "*.md"

# 搜索文档
python rag_manager.py search "Python编程"

# 查看统计
python rag_manager.py stats

# 测试对话
python rag_manager.py test "如何使用这个项目？"
```

### 快速启动
```bash
python quick_start.py
# 选择 "2. 演示RAG功能"
```

## 📝 使用示例

### 1. 添加文档
```python
agent = create_agent()

# 从文件添加
doc_id = agent.add_document_from_file(
    "path/to/document.txt",
    category="documentation",
    tags="guide,tutorial"
)

# 直接添加内容
doc_id = agent.add_document(
    title="API文档",
    content="这是API的使用说明...",
    author="开发团队"
)
```

### 2. 搜索文档
```python
# 搜索相关文档
results = agent.search_documents("Python函数", top_k=5)

for doc in results:
    print(f"标题: {doc.title}")
    print(f"相似度: {doc.similarity_score}")
    print(f"文件: {doc.file_path}")
```

### 3. 增强对话
```python
# 用户问题会自动检索相关文档
response = agent.chat("session1", "如何定义Python函数？")
# 响应会包含相关文档内容和引用信息
```

## 🎯 工作流程

### 用户输入处理流程
1. **会话初始化**：加载历史记录
2. **用户输入保存**：存储到数据库
3. **工具检测**：检查是否需要工具调用
4. **条件路由**：
   - 如果需要工具 → 工具执行
   - 如果不需要工具 → RAG检索
5. **RAG检索**：搜索相关文档
6. **LLM生成**：结合文档上下文生成回答
7. **最终响应**：返回带引用的回答

### RAG检索流程
1. **生成查询embedding**
2. **FAISS向量搜索**：语义相似度搜索
3. **关键词搜索**：补充全文搜索结果
4. **结果合并**：按相似度排序
5. **上下文构建**：格式化为LLM输入
6. **引用生成**：创建文档引用信息

## 📈 监控和日志

### 搜索日志
- 自动记录所有搜索查询
- 包含执行时间和结果数量
- 支持按会话分组分析

### 性能指标
- 文档数量和总字数
- FAISS索引大小
- 搜索响应时间
- 缓存命中率

## 🔍 故障排除

### 常见问题
1. **FAISS导入错误**：安装`faiss-cpu`
2. **embedding生成失败**：检查Azure OpenAI配置
3. **搜索结果为空**：确认文档已正确添加
4. **内存使用过高**：考虑使用量化索引

### 调试工具
```bash
# 测试FAISS功能
python test_faiss.py

# 检查文档统计
python -c "from src.agent.conversation_agent import create_agent; print(create_agent().get_document_stats())"
```

## 🛡️ 安全考虑

### 数据隐私
- 文档内容存储在本地SQLite数据库
- 向量索引文件本地存储
- 不会向外部服务发送文档内容

### 访问控制
- 支持文档访问级别设置
- 可以按业务单元分类管理
- 支持文档过期时间设置

## 🚀 未来扩展

### 计划功能
1. **多模态支持**：图片、PDF等格式
2. **分布式索引**：支持多机部署
3. **实时更新**：文档变更自动同步
4. **智能分块**：自动文档分段
5. **相关性学习**：基于用户反馈优化搜索

### 性能优化
1. **索引优化**：使用IVF或HNSW索引
2. **缓存策略**：常用查询结果缓存
3. **异步处理**：背景embedding生成
4. **压缩存储**：向量量化压缩

---

## 📞 获取帮助

如果在使用过程中遇到问题，请：
1. 查看`FAISS_INSTALLATION.md`了解安装说明
2. 运行`test_faiss.py`进行功能测试
3. 查看日志文件排查问题
4. 提交Issue描述具体问题

享受增强的RAG功能！🎉
