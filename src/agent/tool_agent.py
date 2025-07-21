"""
工具管理Agent，专门处理工具调用和系统管理相关的操作。
"""
from typing import Dict, Any, Optional
from src.agent.base_agent import AbstractManagedAgent
from src.config.prompt_manager import get_prompt_manager
from src.config.settings import get_llm_model
from src.tools.tool_manager import ToolMatcher, ToolConfirmationSystem

class ToolAgent(AbstractManagedAgent):
    """
    专门处理工具调用和系统管理的Agent。
    """
    
    def __init__(self):
        super().__init__(
            description="专门处理工具调用、系统操作、数据处理的Agent。"
                       "适合处理计算、数据转换、外部API调用、系统查询等任务。"
                       "擅长执行具体的操作指令和工具调用。"
        )
        self.prompt_manager = get_prompt_manager()
        # todo: 使用新的模型管理系统
        self.llm = get_llm_model()
        self.tool_matcher = ToolMatcher()
        self.tool_confirmation = ToolConfirmationSystem()
    
    def invoke(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        处理工具相关的查询。
        
        Args:
            query (str): 用户的工具相关问题
            context (Optional[Dict[str, Any]]): 上下文信息
            
        Returns:
            str: 处理结果
        """
        print(f"🔧 ToolAgent received query: {query}")
        
        # 检测是否需要使用工具
        tool_detection_result = self.tool_matcher.detect_tool_need(query)
        
        if tool_detection_result and tool_detection_result.get("needs_tool", False):
            return self._handle_tool_execution(query, tool_detection_result, context)
        else:
            return self._handle_general_tool_query(query, context)
    
    def _handle_tool_execution(self, query: str, tool_info: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """处理工具执行"""
        print("⚡ Executing tool...")
        
        try:
            tool_name = tool_info.get("tool_name")
            suggested_args = tool_info.get("suggested_args", {})
            confidence = tool_info.get("confidence", 0.5)
            
            print(f"🔨 Detected tool: {tool_name}")
            print(f"📋 Suggested args: {suggested_args}")
            print(f"🎯 Confidence: {confidence:.2f}")
            
            # 在Agent模式下，我们可以自动执行工具而不需要用户确认
            # 或者根据置信度决定是否执行
            if confidence >= 0.8:
                success, result = self.tool_confirmation.execute_tool(tool_name, suggested_args)
                
                if success:
                    return f"✅ 工具执行成功：\n\n{result}"
                else:
                    return f"❌ 工具执行失败：{result}"
            else:
                return f"⚠️ 工具检测置信度较低（{confidence:.2f}），建议使用其他方式处理该请求。"
                
        except Exception as e:
            return f"❌ 工具执行时发生异常：{str(e)}"
    
    def _handle_general_tool_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """处理一般工具查询"""
        print("💭 Handling general tool query...")
        
        # 获取可用工具列表
        available_tools = self.tool_matcher.available_tools
        
        # 构建响应prompt
        tools_info = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in available_tools
        ])
        
        response_prompt = f"""
        用户查询：{query}
        
        可用工具：
        {tools_info}
        
        请根据用户查询和可用工具，提供有用的回答。如果用户需要使用特定工具，请说明如何使用。
        """
        
        try:
            response = self.llm.invoke(response_prompt)
            # 处理可能的响应格式
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            return f"🔧 工具助手回答：\n\n{response_text}"
        except Exception as e:
            return f"处理查询时发生错误：{str(e)}"
    
    def get_available_tools(self) -> str:
        """获取可用工具列表"""
        try:
            tools = self.tool_matcher.available_tools
            if not tools:
                return "当前没有可用的工具。"
            
            response = "📋 可用工具列表：\n\n"
            for tool in tools:
                response += f"🔨 **{tool.name}**\n"
                response += f"   描述：{tool.description}\n"
                
                # 获取参数信息
                if tool.args_schema:
                    schema = tool.args_schema.schema()
                    properties = schema.get('properties', {})
                    if properties:
                        response += f"   参数：{', '.join(properties.keys())}\n"
                    else:
                        response += f"   参数：无\n"
                else:
                    response += f"   参数：无\n"
                response += "\n"
            
            return response
        except Exception as e:
            return f"获取工具列表时发生错误：{str(e)}"
    
    def execute_specific_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """执行指定的工具"""
        try:
            print(f"🔨 Executing tool: {tool_name} with args: {tool_args}")
            success, result = self.tool_confirmation.execute_tool(tool_name, tool_args)
            
            if success:
                return f"✅ 工具 {tool_name} 执行成功：\n\n{result}"
            else:
                return f"❌ 工具 {tool_name} 执行失败：{result}"
                
        except Exception as e:
            return f"❌ 执行工具 {tool_name} 时发生异常：{str(e)}"
