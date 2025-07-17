import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

class DocumentDatabase:
    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化文档数据库，创建所需的表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建文档表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,  -- 存储embedding向量
                file_path TEXT,  -- 文件路径
                file_type TEXT,  -- 文件类型 (txt, pdf, md, etc.)
                chunk_index INTEGER DEFAULT 0,  -- 块索引（如果文档被分块）
                start_line INTEGER,  -- 开始行数
                end_line INTEGER,    -- 结束行数
                word_count INTEGER,  -- 字数
                char_count INTEGER,  -- 字符数
                language TEXT DEFAULT 'zh',  -- 语言
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- 扩展字段
                metadata TEXT,       -- JSON格式的额外元数据
                tags TEXT,          -- 标签，逗号分隔
                category TEXT,      -- 文档分类
                priority INTEGER DEFAULT 0,  -- 优先级
                status TEXT DEFAULT 'active',  -- 状态 (active, archived, deleted)
                source TEXT,        -- 来源
                author TEXT,        -- 作者
                version TEXT DEFAULT '1.0',  -- 版本
                parent_id TEXT,     -- 父文档ID（用于层次结构）
                -- 搜索相关
                search_keywords TEXT,  -- 搜索关键词
                summary TEXT,       -- 文档摘要
                -- 业务相关扩展字段
                business_unit TEXT, -- 业务单元
                access_level TEXT DEFAULT 'public',  -- 访问级别
                expiry_date TIMESTAMP,  -- 过期时间
                custom_field1 TEXT, -- 自定义字段1
                custom_field2 TEXT, -- 自定义字段2
                custom_field3 TEXT  -- 自定义字段3
            )
        ''')
        
        # 创建文档集合表（用于管理文档组）
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
                search_type TEXT,  -- 'semantic', 'keyword', 'hybrid'
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
    
    def add_document(self, title: str, content: str, file_path: str = None, 
                    embedding: bytes = None, **kwargs) -> str:
        """添加文档"""
        doc_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基础字段
        params = {
            'id': doc_id,
            'title': title,
            'content': content,
            'file_path': file_path,
            'embedding': embedding,
            'word_count': len(content.split()),
            'char_count': len(content),
            'updated_at': datetime.now().isoformat()
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
    
    def search_documents(self, query: str, limit: int = 10, 
                        search_type: str = 'hybrid') -> List[Dict[str, Any]]:
        """搜索文档"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        
        if search_type in ['fts', 'hybrid']:
            # 全文搜索
            cursor.execute('''
                SELECT d.*, 
                       snippet(documents_fts, 2, '<mark>', '</mark>', '...', 32) as snippet
                FROM documents_fts fts
                JOIN documents d ON fts.id = d.id
                WHERE documents_fts MATCH ?
                  AND d.status = 'active'
                ORDER BY bm25(documents_fts)
                LIMIT ?
            ''', (query, limit))
            
            fts_results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for row in fts_results:
                result = dict(zip(columns, row))
                result['search_type'] = 'fts'
                results.append(result)
        
        if search_type in ['keyword', 'hybrid'] and len(results) < limit:
            # 关键词搜索
            remaining_limit = limit - len(results)
            cursor.execute('''
                SELECT * FROM documents
                WHERE (title LIKE ? OR content LIKE ? OR search_keywords LIKE ?)
                  AND status = 'active'
                  AND id NOT IN ({})
                ORDER BY created_at DESC
                LIMIT ?
            '''.format(','.join(['?' for _ in results])), 
            [f'%{query}%', f'%{query}%', f'%{query}%'] + 
            [r['id'] for r in results] + [remaining_limit])
            
            keyword_results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for row in keyword_results:
                result = dict(zip(columns, row))
                result['search_type'] = 'keyword'
                results.append(result)
        
        conn.close()
        
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
    
    def update_document_embedding(self, doc_id: str, embedding: bytes):
        """更新文档的embedding"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE documents 
            SET embedding = ?, updated_at = ?
            WHERE id = ?
        ''', (embedding, datetime.now().isoformat(), doc_id))
        
        conn.commit()
        conn.close()
    
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
        
        return affected > 0
    
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
        
        conn.close()
        return stats
