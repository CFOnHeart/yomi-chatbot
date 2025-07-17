from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dataclasses import dataclass

@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    success: bool
    tool_name: str
    tool_args: Dict[str, Any]
    result: str
    confidence: float

class AgentState(TypedDict):
    """Agent状态定义"""
    # 核心状态
    session_id: str
    user_input: str
    messages: List[BaseMessage]
    
    # 工具相关
    needs_tool: bool
    tool_detection_result: Optional[Dict[str, Any]]
    tool_execution_result: Optional[ToolExecutionResult]
    
    # 最终响应
    final_response: str
    
    # 元数据
    step_count: int
    error_message: Optional[str]
