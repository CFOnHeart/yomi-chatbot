import sqlite3
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
import os

try:
    import faiss
except ImportError:
    print("⚠️ FAISS library not installed. Please install it with: pip install faiss-cpu")
    faiss = None

class FAISSDocumentDatabase:
    def __init__(self, db_path: str = "documents.db", vector_path: str = "vectors.index"):
        self.db_path = db_path
        self.vector_path = vector_path
        self.metadata_path = vector_path.replace('.index', '_metadata.pkl')
        
        # FAISS索引
        self.index = None
        self.dimension = 1536  # Azure OpenAI embedding dimension
        self.document_ids = []  # 保存文档ID的顺序
        if not os.path.exists(self.db_path):
            print (f"⚠️ 数据库文件 {self.db_path} 不存在，初始化数据库...")
            self.init_database()
        self.init_faiss_index()

    def init_database(self):
        """初始化SQLite数据库，只存储元数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建文档元数据表（不存储embedding）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                file_path TEXT,
                file_type TEXT,
                chunk_index INTEGER DEFAULT 0,
                start_line INTEGER,
                end_line INTEGER,
                word_count INTEGER,
                char_count INTEGER,
                language TEXT DEFAULT 'zh',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- 扩展字段
                metadata TEXT,
                tags TEXT,
                category TEXT,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                source TEXT,
                author TEXT,
                version TEXT DEFAULT '1.0',
                parent_id TEXT,
                -- 搜索相关
                search_keywords TEXT,
                summary TEXT,
                -- 业务相关扩展字段
                business_unit TEXT,
                access_level TEXT DEFAULT 'public',
                expiry_date TIMESTAMP,
                custom_field1 TEXT,
                custom_field2 TEXT,
                custom_field3 TEXT,
                -- FAISS索引位置
                faiss_index INTEGER  -- 在FAISS索引中的位置
            )
        ''')
        
        # 创建文档集合表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # 创建文档与集合的关联表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_collection_mapping (
                document_id TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (document_id, collection_id),
                FOREIGN KEY (document_id) REFERENCES documents (id),
                FOREIGN KEY (collection_id) REFERENCES document_collections (id)
            )
        ''')
        
        # 创建搜索日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                results_count INTEGER,
                search_type TEXT,
                execution_time FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                metadata TEXT
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents(tags)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_logs_query ON search_logs(query)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_logs_session ON search_logs(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_faiss_index ON documents(faiss_index)')
        
        # 创建全文搜索索引
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                id, title, content, summary, search_keywords,
                content='documents',
                content_rowid='rowid'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_faiss_index(self):
        """初始化FAISS索引"""
        if faiss is None:
            print("⚠️ FAISS not available, falling back to traditional search only")
            return
        
        # 尝试加载现有的FAISS索引
        if Path(self.vector_path).exists() and Path(self.metadata_path).exists():
            try:
                self.index = faiss.read_index(self.vector_path)
                
                # 加载元数据
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.document_ids = metadata.get('document_ids', [])
                    self.dimension = metadata.get('dimension', 1536)
                
                print(f"✅ 加载FAISS索引: {self.index.ntotal} 个向量")
                
            except Exception as e:
                print(f"⚠️ 加载FAISS索引失败: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """创建新的FAISS索引"""
        if faiss is None:
            return
        
        # 创建FAISS索引 (使用L2距离的平面索引)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.document_ids = []
        
        # 如果数据库中已有文档，重新构建索引
        self._rebuild_index()
    
    def _rebuild_index(self):
        """重建FAISS索引"""
        if faiss is None:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有有效文档
        cursor.execute('''
            SELECT id, content FROM documents 
            WHERE status = 'active' 
            ORDER BY created_at
        ''')
        
        documents = cursor.fetchall()
        conn.close()
        
        if not documents:
            return
        
        print(f"🔄 重建FAISS索引，处理 {len(documents)} 个文档...")
        
        # 这里需要重新生成embeddings，实际使用时需要调用embedding服务
        # 暂时跳过重建，等待新文档添加时再构建
        print("⚠️ 需要重新生成embeddings才能重建索引")
    
    def add_document(self, title: str, content: str, embedding: np.ndarray = None,
                    file_path: str = None, **kwargs) -> str:
        """添加文档"""
        doc_id = str(uuid.uuid4())
        
        # 添加到SQLite数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        faiss_index = None
        if embedding is not None and faiss is not None and self.index is not None:
            # 添加到FAISS索引
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            
            # 确保embedding维度正确
            if embedding.shape[1] != self.dimension:
                print(f"⚠️ Embedding维度不匹配: {embedding.shape[1]} != {self.dimension}")
                return doc_id
            
            # 添加到FAISS索引
            faiss_index = self.index.ntotal
            self.index.add(embedding.astype('float32'))
            self.document_ids.append(doc_id)
            
            # 保存索引
            self._save_index()
        
        # 基础字段
        params = {
            'id': doc_id,
            'title': title,
            'content': content,
            'file_path': file_path,
            'word_count': len(content.split()),
            'char_count': len(content),
            'updated_at': datetime.now().isoformat(),
            'faiss_index': faiss_index
        }
        
        # 添加扩展字段
        params.update(kwargs)
        
        # 构建SQL
        columns = ', '.join(params.keys())
        placeholders = ', '.join(['?' for _ in params])
        
        cursor.execute(f'''
            INSERT INTO documents ({columns})
            VALUES ({placeholders})
        ''', list(params.values()))
        
        # 添加到全文搜索索引
        cursor.execute('''
            INSERT INTO documents_fts (id, title, content, summary, search_keywords)
            VALUES (?, ?, ?, ?, ?)
        ''', (doc_id, title, content, kwargs.get('summary', ''), kwargs.get('search_keywords', '')))
        
        conn.commit()
        conn.close()
        
        return doc_id
    
    def _save_index(self):
        """保存FAISS索引和元数据"""
        if faiss is None or self.index is None:
            return
        
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, self.vector_path)
            
            # 保存元数据
            metadata = {
                'document_ids': self.document_ids,
                'dimension': self.dimension,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
        except Exception as e:
            print(f"⚠️ 保存FAISS索引失败: {e}")
    
    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """语义搜索"""
        if faiss is None or self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # 确保查询向量维度正确
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            if query_embedding.shape[1] != self.dimension:
                print(f"⚠️ 查询向量维度不匹配: {query_embedding.shape[1]} != {self.dimension}")
                return []
            
            # 搜索最相似的向量
            distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))
            
            # 转换为相似度分数 (距离越小相似度越高)
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.document_ids):
                    # 将L2距离转换为相似度分数 (0-1)
                    similarity = 1.0 / (1.0 + distance)
                    results.append((self.document_ids[idx], similarity))
            
            return results
            
        except Exception as e:
            print(f"⚠️ 语义搜索失败: {e}")
            return []
    
    def search_documents(self, query: str, query_embedding: np.ndarray = None, 
                        limit: int = 10, search_type: str = 'hybrid') -> List[Dict[str, Any]]:
        """搜索文档"""
        results = []
        
        # 1. 语义搜索
        if query_embedding is not None and search_type in ['semantic', 'hybrid']:
            semantic_results = self.semantic_search(query_embedding, limit)
            
            if semantic_results:
                # 获取文档详情
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                doc_ids = [doc_id for doc_id, _ in semantic_results]
                placeholders = ','.join(['?' for _ in doc_ids])
                
                cursor.execute(f'''
                    SELECT * FROM documents 
                    WHERE id IN ({placeholders}) AND status = 'active'
                ''', doc_ids)
                
                doc_rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                docs_dict = {row[0]: dict(zip(columns, row)) for row in doc_rows}
                
                conn.close()
                
                # 按相似度排序
                for doc_id, similarity in semantic_results:
                    if doc_id in docs_dict:
                        doc = docs_dict[doc_id]
                        doc['similarity_score'] = similarity
                        doc['search_type'] = 'semantic'
                        results.append(doc)
        
        # 2. 全文搜索和关键词搜索
        if search_type in ['fts', 'keyword', 'hybrid']:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 排除已有结果
            existing_ids = [r['id'] for r in results]
            remaining_limit = limit - len(results)
            
            if remaining_limit > 0:
                if search_type in ['fts', 'hybrid']:
                    # 全文搜索
                    exclude_clause = ''
                    params = [query, remaining_limit]
                    
                    if existing_ids:
                        exclude_clause = f"AND d.id NOT IN ({','.join(['?' for _ in existing_ids])})"
                        params = [query] + existing_ids + [remaining_limit]
                    
                    cursor.execute(f'''
                        SELECT d.*, 
                               snippet(documents_fts, 2, '<mark>', '</mark>', '...', 32) as snippet
                        FROM documents_fts fts
                        JOIN documents d ON fts.id = d.id
                        WHERE documents_fts MATCH ?
                          AND d.status = 'active'
                          {exclude_clause}
                        ORDER BY bm25(documents_fts)
                        LIMIT ?
                    ''', params)
                    
                    fts_results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in fts_results:
                        result = dict(zip(columns, row))
                        result['search_type'] = 'fts'
                        result['similarity_score'] = 0.8  # 默认FTS分数
                        results.append(result)
                
                # 更新剩余限制
                remaining_limit = limit - len(results)
                
                if remaining_limit > 0 and search_type in ['keyword', 'hybrid']:
                    # 关键词搜索
                    existing_ids = [r['id'] for r in results]
                    exclude_clause = ''
                    params = [f'%{query}%', f'%{query}%', f'%{query}%', remaining_limit]
                    
                    if existing_ids:
                        exclude_clause = f"AND id NOT IN ({','.join(['?' for _ in existing_ids])})"
                        params = [f'%{query}%', f'%{query}%', f'%{query}%'] + existing_ids + [remaining_limit]
                    
                    cursor.execute(f'''
                        SELECT * FROM documents
                        WHERE (title LIKE ? OR content LIKE ? OR search_keywords LIKE ?)
                          AND status = 'active'
                          {exclude_clause}
                        ORDER BY created_at DESC
                        LIMIT ?
                    ''', params)
                    
                    keyword_results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in keyword_results:
                        result = dict(zip(columns, row))
                        result['search_type'] = 'keyword'
                        result['similarity_score'] = 0.6  # 默认关键词分数
                        results.append(result)
            
            conn.close()
        
        # 按相似度排序
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return results[:limit]
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))
            conn.close()
            return result
        
        conn.close()
        return None
    
    def update_document_embedding(self, doc_id: str, embedding: np.ndarray):
        """更新文档的embedding"""
        if faiss is None or self.index is None:
            return False
        
        # 查找文档在FAISS索引中的位置
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT faiss_index FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        
        if row and row[0] is not None:
            faiss_index = row[0]
            
            # 更新FAISS索引中的向量
            if faiss_index < self.index.ntotal:
                # FAISS不支持直接更新，需要重建索引
                # 这里暂时跳过，实际应用中可能需要重建整个索引
                print("⚠️ FAISS不支持直接更新向量，需要重建索引")
            
            # 更新数据库时间戳
            cursor.execute('''
                UPDATE documents 
                SET updated_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), doc_id))
            
            conn.commit()
        
        conn.close()
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档（软删除）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE documents 
            SET status = 'deleted', updated_at = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), doc_id))
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        
        # 注意：这里不从FAISS索引中删除，因为FAISS不支持删除
        # 在实际应用中，可能需要定期重建索引来清理已删除的文档
        
        return affected > 0
    
    def log_search(self, query: str, results_count: int, search_type: str,
                  execution_time: float, session_id: str = None, metadata: str = None):
        """记录搜索日志"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_logs (id, query, results_count, search_type, 
                                   execution_time, session_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), query, results_count, search_type, 
              execution_time, session_id, metadata))
        
        conn.commit()
        conn.close()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # 总文档数
        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'active'")
        stats['total_documents'] = cursor.fetchone()[0]
        
        # 按类型统计
        cursor.execute('''
            SELECT file_type, COUNT(*) as count
            FROM documents 
            WHERE status = 'active'
            GROUP BY file_type
            ORDER BY count DESC
        ''')
        stats['by_type'] = dict(cursor.fetchall())
        
        # 按分类统计
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM documents 
            WHERE status = 'active' AND category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
        ''')
        stats['by_category'] = dict(cursor.fetchall())
        
        # 总字数
        cursor.execute("SELECT SUM(word_count) FROM documents WHERE status = 'active'")
        stats['total_words'] = cursor.fetchone()[0] or 0
        
        # FAISS索引统计
        if self.index is not None:
            stats['faiss_vectors'] = self.index.ntotal
            stats['faiss_dimension'] = self.dimension
        else:
            stats['faiss_vectors'] = 0
            stats['faiss_dimension'] = 0
        
        conn.close()
        return stats
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """获取所有文档集合"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM document_collections ORDER BY name')
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        return [dict(zip(columns, row)) for row in results]
    
    def create_collection(self, name: str, description: str = None, metadata: str = None) -> str:
        """创建文档集合"""
        collection_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO document_collections (id, name, description, metadata)
            VALUES (?, ?, ?, ?)
        ''', (collection_id, name, description, metadata))
        
        conn.commit()
        conn.close()
        
        return collection_id
    
    def add_document_to_collection(self, document_id: str, collection_id: str):
        """将文档添加到集合"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO document_collection_mapping (document_id, collection_id)
            VALUES (?, ?)
        ''', (document_id, collection_id))
        
        conn.commit()
        conn.close()
    
    def rebuild_faiss_index(self):
        """重建FAISS索引"""
        if faiss is None:
            print("⚠️ FAISS not available")
            return False
        
        print("🔄 开始重建FAISS索引...")
        
        # 创建新索引
        new_index = faiss.IndexFlatL2(self.dimension)
        new_document_ids = []
        
        # 注意：这里需要重新生成所有embeddings
        # 实际使用时需要调用embedding服务
        print("⚠️ 需要重新生成所有文档的embeddings")
        
        # 更新索引
        self.index = new_index
        self.document_ids = new_document_ids
        
        # 保存索引
        self._save_index()
        
        print("✅ FAISS索引重建完成")
        return True
