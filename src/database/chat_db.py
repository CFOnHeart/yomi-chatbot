import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class ChatDatabase:
    def __init__(self, db_path: str = "database/chat_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库，创建所需的表"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                session_name TEXT
            )
        ''')
        
        # 创建聊天记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_type TEXT NOT NULL,  -- 'human', 'ai', 'system', 'tool'
                content TEXT NOT NULL,
                tool_name TEXT,  -- 如果是工具调用
                tool_args TEXT,  -- 工具参数的JSON字符串
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
            )
        ''')
        
        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_id ON chat_messages(session_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON chat_messages(created_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id: str, user_id: str = None, session_name: str = None) -> bool:
        """创建新的聊天会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO chat_sessions (session_id, user_id, session_name, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (session_id, user_id, session_name, datetime.now()))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error creating session: {e}")
            return False
        finally:
            conn.close()
    
    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT 1 FROM chat_sessions WHERE session_id = ?', (session_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def add_message(self, session_id: str, message_type: str, content: str, 
                   tool_name: str = None, tool_args: dict = None) -> bool:
        """添加聊天消息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            tool_args_json = json.dumps(tool_args) if tool_args else None
            
            cursor.execute('''
                INSERT INTO chat_messages (session_id, message_type, content, tool_name, tool_args)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, message_type, content, tool_name, tool_args_json))
            
            # 更新会话的最后更新时间
            cursor.execute('''
                UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?
            ''', (datetime.now(), session_id))
            
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error adding message: {e}")
            return False
        finally:
            conn.close()
    
    def get_session_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """获取会话历史记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT message_type, content, tool_name, tool_args, created_at 
            FROM chat_messages 
            WHERE session_id = ? 
            ORDER BY created_at ASC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            message_type, content, tool_name, tool_args, created_at = row
            message = {
                'type': message_type,
                'content': content,
                'created_at': created_at
            }
            
            if tool_name:
                message['tool_name'] = tool_name
            if tool_args:
                message['tool_args'] = json.loads(tool_args)
            
            messages.append(message)
        
        return messages
    
    def get_session_message_count(self, session_id: str) -> int:
        """获取会话消息数量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM chat_messages WHERE session_id = ?', (session_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def get_session_text_length(self, session_id: str) -> int:
        """获取会话文本总长度"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT content FROM chat_messages WHERE session_id = ?', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        total_length = sum(len(row[0]) for row in rows)
        return total_length
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有消息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting session: {e}")
            return False
        finally:
            conn.close()
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """获取所有会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, user_id, session_name, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            session_id, user_id, session_name, created_at, updated_at = row
            sessions.append({
                'session_id': session_id,
                'user_id': user_id,
                'session_name': session_name,
                'created_at': created_at,
                'updated_at': updated_at
            })
        
        return sessions
