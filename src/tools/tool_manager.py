from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import get_llm_model
from src.tools.simple.math import add, multiply
from src.tools.simple.human_assistance import human_assistance
import json
import re

class ToolMatcher:
    """工具匹配器，用于检测用户输入是否需要调用工具"""
    
    def __init__(self):
        self.llm = get_llm_model()
        self.available_tools = self._load_available_tools()
        self.tool_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个工具检测助手。分析用户的输入，判断是否需要调用工具。

可用工具：
{available_tools}

请分析用户输入，如果需要调用工具，请返回JSON格式：
{{
    "needs_tool": true,
    "tool_name": "工具名称",
    "confidence": 0.95,
    "reason": "需要调用工具的原因",
    "suggested_args": {{
        "arg1": "value1",
        "arg2": "value2"
    }}
}}

如果不需要调用工具，请返回：
{{
    "needs_tool": false,
    "confidence": 0.9,
    "reason": "不需要工具的原因"
}}

用户输入：{user_input}"""),
        ])
    
    def _load_available_tools(self) -> List[BaseTool]:
        """加载可用工具"""
        return [add, multiply, human_assistance]
    
    def _get_tool_schemas(self) -> str:
        """获取工具模式描述"""
        schemas = []
        for tool in self.available_tools:
            schema = {
                "name": tool.name,
                "description": tool.description,
                "args": tool.args_schema.schema() if tool.args_schema else {}
            }
            schemas.append(schema)
        return json.dumps(schemas, indent=2, ensure_ascii=False)
    
    def detect_tool_need(self, user_input: str) -> Dict[str, Any]:
        """检测用户输入是否需要调用工具"""
        try:
            tool_schemas = self._get_tool_schemas()
            
            detection_chain = self.tool_detection_prompt | self.llm
            response = detection_chain.invoke({
                "available_tools": tool_schemas,
                "user_input": user_input
            })
            
            # 解析响应
            content = response.content.strip()
            
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {
                    "needs_tool": False,
                    "confidence": 0.1,
                    "reason": "无法解析工具检测结果"
                }
                
        except Exception as e:
            print(f"工具检测失败: {e}")
            return {
                "needs_tool": False,
                "confidence": 0.1,
                "reason": f"检测错误: {str(e)}"
            }
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        for tool in self.available_tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具的详细模式"""
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            return None
        
        return {
            "name": tool.name,
            "description": tool.description,
            "args": tool.args_schema.schema() if tool.args_schema else {},
            "example": self._get_tool_example(tool_name)
        }
    
    def _get_tool_example(self, tool_name: str) -> str:
        """获取工具使用示例"""
        examples = {
            "add": "例如：计算 5 + 3 = 8",
            "multiply": "例如：计算 4 × 7 = 28", 
            "human_assistance": "例如：需要人工帮助解决复杂问题"
        }
        return examples.get(tool_name, "无示例")

class ToolConfirmationSystem:
    """工具确认系统，用于向用户确认是否执行工具"""
    
    def __init__(self):
        self.tool_matcher = ToolMatcher()
    
    def confirm_tool_execution(self, tool_name: str, suggested_args: Dict[str, Any]) -> bool:
        """确认工具执行"""
        tool_schema = self.tool_matcher.get_tool_schema(tool_name)
        
        if not tool_schema:
            print(f"❌ 未找到工具: {tool_name}")
            return False
        
        # 显示工具信息
        print("\n" + "="*50)
        print(f"🔧 检测到可用工具: {tool_name}")
        print("="*50)
        print(f"📝 工具描述: {tool_schema['description']}")
        print(f"🎯 工具示例: {tool_schema['example']}")
        print(f"📋 建议参数: {json.dumps(suggested_args, indent=2, ensure_ascii=False)}")
        
        # 显示参数模式
        if tool_schema['args']:
            print("\n📋 参数模式:")
            for arg_name, arg_info in tool_schema['args'].get('properties', {}).items():
                arg_type = arg_info.get('type', 'unknown')
                arg_desc = arg_info.get('description', '无描述')
                print(f"  • {arg_name} ({arg_type}): {arg_desc}")
        
        print("="*50)
        
        # 询问用户确认
        while True:
            user_input = input("是否执行此工具? (y/n/显示更多信息输入'info'): ").strip().lower()
            
            if user_input in ['y', 'yes', '是', '确定']:
                return True
            elif user_input in ['n', 'no', '否', '取消']:
                return False
            elif user_input in ['info', 'i', '信息']:
                self._show_detailed_info(tool_name, tool_schema)
            else:
                print("❌ 请输入 y(是) 或 n(否)")
    
    def _show_detailed_info(self, tool_name: str, tool_schema: Dict[str, Any]):
        """显示工具详细信息"""
        print(f"\n📋 工具 '{tool_name}' 详细信息:")
        print(f"完整模式: {json.dumps(tool_schema, indent=2, ensure_ascii=False)}")
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
        """执行工具"""
        try:
            tool = self.tool_matcher.get_tool_by_name(tool_name)
            if not tool:
                return False, f"未找到工具: {tool_name}"
            
            # 执行工具
            result = tool.invoke(tool_args)
            return True, str(result)
            
        except Exception as e:
            return False, f"工具执行失败: {str(e)}"
    
    def process_user_input(self, user_input: str) -> Tuple[bool, Dict[str, Any]]:
        """处理用户输入，检测并确认工具使用"""
        # 检测是否需要工具
        detection_result = self.tool_matcher.detect_tool_need(user_input)
        
        if not detection_result.get("needs_tool", False):
            return False, {
                "reason": detection_result.get("reason", "不需要工具"),
                "confidence": detection_result.get("confidence", 0.5)
            }
        
        # 提取工具信息
        tool_name = detection_result.get("tool_name")
        suggested_args = detection_result.get("suggested_args", {})
        confidence = detection_result.get("confidence", 0.5)
        
        if confidence < 0.7:
            print(f"⚠️ 工具检测置信度较低 ({confidence:.2f})，建议直接使用LLM回答")
            return False, {"reason": "置信度过低", "confidence": confidence}
        
        # 确认工具执行
        if self.confirm_tool_execution(tool_name, suggested_args):
            # 执行工具
            success, result = self.execute_tool(tool_name, suggested_args)
            
            return True, {
                "tool_name": tool_name,
                "tool_args": suggested_args,
                "success": success,
                "result": result,
                "confidence": confidence
            }
        else:
            return False, {"reason": "用户取消工具执行", "confidence": confidence}
