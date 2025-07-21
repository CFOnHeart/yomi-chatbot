from typing import List, Dict, Any
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
import threading
from src.database.chat_db import ChatDatabase
from src.model.chat.base_model import BaseManagedModel


class DatabaseChatMessageHistory(BaseChatMessageHistory):
    """基于数据库的聊天记录历史管理器"""
    
    def __init__(self, session_id: str, database: ChatDatabase):
        self.session_id = session_id
        self.db = database
        self._messages: List[BaseMessage] = []
        self._loaded = False
    
    def _ensure_loaded(self):
        """确保历史记录已加载"""
        if not self._loaded:
            self._load_history()
            self._loaded = True
    
    def _load_history(self):
        """从数据库加载历史记录"""
        history = self.db.get_session_history(self.session_id)
        self._messages = []
        
        for msg in history:
            if msg['type'] == 'human':
                self._messages.append(HumanMessage(content=msg['content']))
            elif msg['type'] == 'ai':
                self._messages.append(AIMessage(content=msg['content']))
            elif msg['type'] == 'system':
                self._messages.append(SystemMessage(content=msg['content']))
            elif msg['type'] == 'tool':
                self._messages.append(ToolMessage(content=msg['content'], tool_call_id=msg.get('tool_name', 'unknown')))
    
    @property
    def messages(self) -> List[BaseMessage]:
        """获取消息列表"""
        self._ensure_loaded()
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息到历史记录"""
        self._ensure_loaded()
        self._messages.append(message)
        
        # 保存到数据库
        message_type = self._get_message_type(message)
        self.db.add_message(self.session_id, message_type, message.content)
    
    def _get_message_type(self, message: BaseMessage) -> str:
        """获取消息类型"""
        if isinstance(message, HumanMessage):
            return 'human'
        elif isinstance(message, AIMessage):
            return 'ai'
        elif isinstance(message, SystemMessage):
            return 'system'
        elif isinstance(message, ToolMessage):
            return 'tool'
        else:
            return 'unknown'
    
    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """添加AI消息"""
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """清空历史记录"""
        self._messages = []

class SmartMemoryManager:
    """智能记忆管理器，支持自动摘要和Token管理"""
    
    def __init__(self, llm: BaseManagedModel, database: ChatDatabase, max_tokens: int = 3200):
        self.db = database
        self.max_tokens = max_tokens
        self.llm = llm
        self.session_histories: Dict[str, DatabaseChatMessageHistory] = {}
        
        # 创建摘要提示模板
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个对话摘要助手。请将以下对话历史进行摘要，保留重要信息和上下文。
摘要应该：
1. 保留关键信息和用户偏好
2. 简洁明了，避免冗余
3. 保持对话的连贯性
4. 长度控制在原文的30%以内

对话历史：
{history}

请提供摘要："""),
        ])
    
    def get_session_history(self, session_id: str) -> DatabaseChatMessageHistory:
        """获取会话历史管理器"""
        if session_id not in self.session_histories:
            # 确保会话存在
            if not self.db.session_exists(session_id):
                self.db.create_session(session_id)
            
            self.session_histories[session_id] = DatabaseChatMessageHistory(session_id, self.db)
        
        return self.session_histories[session_id]
    
    def add_user_message(self, session_id: str, message: str):
        """添加用户消息"""
        history = self.get_session_history(session_id)
        history.add_user_message(message)
        
        # 异步检查是否需要摘要
        self._async_check_and_summarize(session_id)
    
    def add_ai_message(self, session_id: str, message: str):
        """添加AI消息"""
        history = self.get_session_history(session_id)
        history.add_ai_message(message)
        
        # 异步检查是否需要摘要
        self._async_check_and_summarize(session_id)
    
    def add_tool_message(self, session_id: str, message: str, tool_name: str, tool_args: dict = None):
        """添加工具消息"""
        history = self.get_session_history(session_id)
        history.add_message(ToolMessage(content=message, tool_call_id=tool_name))
        
        # 保存到数据库
        self.db.add_message(session_id, 'tool', message, tool_name, tool_args)
        
        # 异步检查是否需要摘要
        self._async_check_and_summarize(session_id)
    
    def _async_check_and_summarize(self, session_id: str):
        """异步检查并摘要历史记录"""
        def check_and_summarize():
            try:
                text_length = self.db.get_session_text_length(session_id)
                if text_length > self.max_tokens:
                    self._summarize_history(session_id)
            except Exception as e:
                print(f"Error in async summarization: {e}")
        
        # 在后台线程中执行
        threading.Thread(target=check_and_summarize, daemon=True).start()
    
    def _summarize_history(self, session_id: str):
        """摘要历史记录"""
        history = self.get_session_history(session_id)
        messages = history.messages
        
        if len(messages) <= 2:  # 保留至少最近的2条消息
            return
        
        # 获取需要摘要的消息（除了最近的2条）
        messages_to_summarize = messages[:-2]
        recent_messages = messages[-2:]
        
        # 构建历史文本
        history_text = ""
        for msg in messages_to_summarize:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"
            elif isinstance(msg, SystemMessage):
                history_text += f"System: {msg.content}\n"
            elif isinstance(msg, ToolMessage):
                history_text += f"Tool: {msg.content}\n"
        
        try:
            # 使用LLM进行摘要
            summary_chain = self.summary_prompt | self.llm
            summary_response = summary_chain.invoke({"history": history_text})
            summary = summary_response.content
            
            # 创建摘要消息
            summary_message = SystemMessage(content=f"[会话摘要]: {summary}")
            
            # 更新内存中的历史记录
            history._messages = [summary_message] + recent_messages
            
            print(f"✅ 会话 {session_id} 的历史记录已自动摘要")
            
        except Exception as e:
            print(f"❌ 摘要失败: {e}")
    
    def get_runnable_with_history(self, runnable):
        """获取带有历史记录的可运行对象"""
        return RunnableWithMessageHistory(
            runnable,
            self.get_session_history,
            input_messages_key="messages",
            history_messages_key="messages",
        )
    
    def initialize_session(self, session_id: str, user_id: str = None, session_name: str = None) -> bool:
        """初始化会话"""
        # 创建或更新会话
        success = self.db.create_session(session_id, user_id, session_name)
        
        if success:
            # 预加载历史记录
            self.get_session_history(session_id)
            print(f"✅ 会话 {session_id} 初始化成功")
        else:
            print(f"❌ 会话 {session_id} 初始化失败")
        
        return success
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        if not self.db.session_exists(session_id):
            return None
        
        message_count = self.db.get_session_message_count(session_id)
        text_length = self.db.get_session_text_length(session_id)
        
        return {
            'session_id': session_id,
            'message_count': message_count,
            'text_length': text_length,
            'needs_summary': text_length > self.max_tokens
        }
