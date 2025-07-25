"""
流式输出处理器
用于管理agent执行过程中的流式事件，选择性地发送给客户端
"""
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque


class StreamingHandler:
    """流式输出处理器"""
    
    def __init__(self):
        self._session_events: Dict[str, deque] = {}
        self._current_session: Optional[str] = None
        self._lock = threading.Lock()
        self._tool_confirmations: Dict[str, Dict[str, Any]] = {}  # 存储工具确认结果
        self._confirmation_events = {}  # 存储确认事件
    
    def set_current_session(self, session_id: str):
        """设置当前会话ID"""
        with self._lock:
            self._current_session = session_id
            if session_id not in self._session_events:
                self._session_events[session_id] = deque()
    
    def add_event(self, event_type: str, data: Dict[str, Any], session_id: Optional[str] = None):
        """添加流式事件"""
        target_session = session_id or self._current_session
        if not target_session:
            return
        
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": self.get_current_timestamp()
        }
        
        with self._lock:
            if target_session in self._session_events:
                self._session_events[target_session].append(event)
    
    def get_pending_events(self, session_id: str) -> List[Dict[str, Any]]:
        """获取待处理的事件"""
        with self._lock:
            if session_id not in self._session_events:
                return []
            
            events = []
            session_queue = self._session_events[session_id]
            
            # 获取所有待处理事件
            while session_queue:
                events.append(session_queue.popleft())
            
            return events
    
    def cleanup_session(self, session_id: str):
        """清理会话事件"""
        with self._lock:
            if session_id in self._session_events:
                del self._session_events[session_id]
            
            if self._current_session == session_id:
                self._current_session = None
    
    def get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().isoformat()
    
    # Agent节点调用的方法
    def tool_detected(self, tool_info: Dict[str, Any]):
        """工具检测事件"""
        self.add_event("tool_detected", {
            "tool_name": tool_info.get("tool_name"),
            "tool_description": tool_info.get("description"),
            "confidence": tool_info.get("confidence"),
            "parameters": tool_info.get("parameters", {})
        })
    
    def tool_execution_start(self, tool_name: str):
        """工具执行开始事件"""
        self.add_event("tool_execution_start", {
            "tool_name": tool_name,
            "message": f"正在执行工具: {tool_name}"
        })
    
    def tool_execution_complete(self, tool_name: str, success: bool, result: Any = None):
        """工具执行完成事件"""
        self.add_event("tool_execution_complete", {
            "tool_name": tool_name,
            "success": success,
            "result": str(result) if result else None,
            "message": f"工具 {tool_name} 执行{'成功' if success else '失败'}"
        })
    
    def llm_response_start(self):
        """LLM响应开始事件"""
        self.add_event("llm_response_start", {
            "message": "AI正在思考中..."
        })
    
    def llm_response_chunk(self, chunk: str):
        """LLM响应片段事件"""
        self.add_event("llm_response_chunk", {
            "chunk": chunk
        })
    
    def llm_response_complete(self, full_response: str):
        """LLM响应完成事件"""
        self.add_event("llm_response_complete", {
            "response": full_response
        })
    
    def session_info(self, info: str):
        """会话信息事件（可选择性发送）"""
        # 这些信息通常不需要发送给用户，可以根据需要启用
        pass
    
    def rag_search_info(self, query: str, results_count: int):
        """RAG搜索信息事件（通常不发送给用户）"""
        # 这些信息通常不需要发送给用户
        pass
    
    def error_occurred(self, error_message: str):
        """错误事件"""
        self.add_event("error", {
            "message": error_message
        })
    
    def tool_confirmation_needed(self, tool_name: str, tool_schema: Dict[str, Any], suggested_args: Dict[str, Any], confidence: float):
        """工具确认需求事件"""
        self.add_event("tool_confirmation_needed", {
            "tool_name": tool_name,
            "tool_schema": tool_schema,
            "suggested_args": suggested_args,
            "confidence": confidence,
            "message": f"检测到工具 {tool_name}，是否执行？"
        })
    
    def set_tool_confirmation(self, session_id: str, confirmed: bool, tool_args: Optional[Dict[str, Any]] = None):
        """设置工具确认结果"""
        with self._lock:
            self._tool_confirmations[session_id] = {
                "confirmed": confirmed,
                "tool_args": tool_args or {},
                "timestamp": self.get_current_timestamp()
            }
            
            # 创建或设置事件来唤醒等待的线程
            if session_id not in self._confirmation_events:
                self._confirmation_events[session_id] = threading.Event()
            self._confirmation_events[session_id].set()
    
    def wait_for_tool_confirmation(self, session_id: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
        """等待工具确认结果"""
        # 创建确认事件
        if session_id not in self._confirmation_events:
            self._confirmation_events[session_id] = threading.Event()
        
        # 等待确认
        if self._confirmation_events[session_id].wait(timeout):
            with self._lock:
                result = self._tool_confirmations.get(session_id)
                if result:
                    # 清理确认数据和事件
                    del self._tool_confirmations[session_id]
                    del self._confirmation_events[session_id]
                    return result
        else:
            # 超时情况下也要清理事件
            with self._lock:
                if session_id in self._confirmation_events:
                    del self._confirmation_events[session_id]
        
        return None


# 全局流式处理器实例
_streaming_handler = StreamingHandler()

def get_streaming_handler() -> StreamingHandler:
    """获取全局流式处理器实例"""
    return _streaming_handler
