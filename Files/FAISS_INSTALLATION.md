# FAISS安装说明

## 1. 安装FAISS

### 对于CPU版本（推荐）：
```bash
pip install faiss-cpu
```

### 对于GPU版本（需要CUDA）：
```bash
pip install faiss-gpu
```

## 2. 验证安装

```python
import faiss
import numpy as np

# 测试FAISS
d = 64  # 向量维度
nb = 100000  # 数据库大小
nq = 10000  # 查询数量

# 创建随机数据
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 创建索引
index = faiss.IndexFlatL2(d)
index.add(xb)

# 搜索
k = 4
D, I = index.search(xq, k)

print(f"搜索完成，找到 {len(I)} 个结果")
```

## 3. 项目依赖更新

将以下依赖添加到 `pyproject.toml` 文件中：

```toml
[project]
dependencies = [
    # ... 其他依赖
    "faiss-cpu>=1.7.4",
    "numpy>=1.24.0",
    # ... 其他依赖
]
```

## 4. 使用说明

### 切换到FAISS版本

在你的代码中，只需要将：
```python
from src.rag.rag_system import RAGSystem
```

改为：
```python
from src.rag.rag_system import RAGSystem
# RAGSystem 现在自动使用 FAISS 后端
```

### 功能特性

1. **语义搜索**：基于embedding向量的相似度搜索
2. **混合搜索**：结合语义搜索和关键词搜索
3. **高性能**：FAISS提供优化的向量搜索算法
4. **持久化**：向量索引自动保存和加载
5. **扩展性**：支持大规模文档集合

### 性能优化

- 使用 `IndexFlatL2` 进行精确搜索
- 对于大规模数据，可以考虑使用 `IndexIVFFlat` 或 `IndexHNSW`
- 向量维度默认为1536（Azure OpenAI的embedding维度）

### 注意事项

1. **初次运行**：系统会自动创建FAISS索引
2. **数据迁移**：如果从旧版本升级，需要重新生成embeddings
3. **内存使用**：FAISS索引会占用内存，大型数据集请注意内存管理
4. **并发访问**：FAISS索引不是线程安全的，多线程环境请加锁

## 5. 故障排除

### 常见问题

1. **ImportError: No module named 'faiss'**
   - 解决方案：运行 `pip install faiss-cpu`

2. **FAISS索引文件损坏**
   - 解决方案：删除 `vectors.index` 和 `vectors_metadata.pkl` 文件，重启系统

3. **内存不足**
   - 解决方案：考虑使用量化索引或分批处理

4. **搜索结果不准确**
   - 解决方案：检查embedding模型，调整相似度阈值

### 调试命令

```bash
# 测试FAISS功能
python -c "from src.rag.rag_system import RAGSystem; rag = RAGSystem(); print('FAISS ready!')"

# 查看索引状态
python -c "from src.database.faiss_document_db import FAISSDocumentDatabase; db = FAISSDocumentDatabase(); print(f'索引大小: {db.index.ntotal if db.index else 0}')"
```
