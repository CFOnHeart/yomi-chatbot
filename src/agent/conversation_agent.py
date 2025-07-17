'''
features:
1. 用户的历史聊天记录需要存入数据库中，请你帮我定义该数据库需要用到的schema，数据库用本地的sqlite实现即可
2. 在一个用户聊天启动会话前，检查该会话id是否在数据库中留有记录，如果有，将该会话记录保存到当前缓存当中
3. 通过RunnableWithMessageHistory来管理当前的message history，如果当前管理的文字个数超过3200个，那就通过llm先summarize前面的history信息，然后重新替换当前的history，减少llm问答token带来的压力，summarize属于加工后的信息不需要保存进数据库
4. lang graph还需要支持调用tool 的几点
5. agent 开始对话之前，先要初始化chat history，通过传入会话的id， 来确定数据库中是否存在历史记录需要加载
6. agent的每一轮对话，收到user的input，工作流程应该是，将用户的输入存入数据库，是否有工具可以匹配该问题，如果有，控制台展示该工具的schema，弹出一个提示让用户是否确定调用该工具。如果调用工具，就直接通过工具给出结果，并将结果存入数据库。如果不调用工具，就直接将问题交给大模型回答，最后将结果存入数据库。一轮结束后，agent会异步检查聊天记录是否超过文字3200的限制，如果有，summarize一下并保存在缓存中
'''
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import AgentNodes

class ConversationAgent:
    """基于LangGraph的对话Agent"""
    
    def __init__(self):
        self.nodes = AgentNodes()
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """创建工作流"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("initialize_session", self.nodes.initialize_session_node)
        workflow.add_node("save_user_input", self.nodes.save_user_input_node)
        workflow.add_node("tool_detection", self.nodes.tool_detection_node)
        workflow.add_node("tool_execution", self.nodes.tool_execution_node)
        workflow.add_node("rag_search", self.nodes.rag_search_node)
        workflow.add_node("llm_response", self.nodes.llm_response_node)
        workflow.add_node("finalize_response", self.nodes.finalize_response_node)
        workflow.add_node("error_handling", self.nodes.error_handling_node)
        
        # 设置入口点
        workflow.set_entry_point("initialize_session")
        
        # 定义工作流
        workflow.add_edge("initialize_session", "save_user_input")
        workflow.add_edge("save_user_input", "tool_detection")
        
        # 工具检测后的条件路由
        workflow.add_conditional_edges(
            "tool_detection",
            self._should_use_tool,
            {
                "use_tool": "tool_execution",
                "rag_search": "rag_search"
            }
        )
        
        # 工具执行后的条件路由
        workflow.add_conditional_edges(
            "tool_execution",
            self._tool_execution_result,
            {
                "success": "finalize_response",
                "failed": "rag_search"
            }
        )
        
        # RAG搜索后直接到LLM响应
        workflow.add_edge("rag_search", "llm_response")
        
        # LLM响应后的路由
        workflow.add_edge("llm_response", "finalize_response")
        
        # 最终响应后结束
        workflow.add_edge("finalize_response", END)
        
        # 错误处理
        workflow.add_edge("error_handling", END)
        
        return workflow.compile()
    
    def _should_use_tool(self, state: AgentState) -> str:
        """判断是否应该使用工具"""
        # 检查是否有错误
        if state.get("error_message"):
            return "error_handling"
        
        # 检查是否需要工具
        if state.get("needs_tool", False):
            return "use_tool"
        else:
            return "rag_search"
    
    def _tool_execution_result(self, state: AgentState) -> str:
        """判断工具执行结果"""
        # 检查是否有错误
        if state.get("error_message"):
            return "failed"
        
        # 检查工具执行结果
        tool_result = state.get("tool_execution_result")
        if tool_result and tool_result.success:
            return "success"
        else:
            return "failed"
    
    def chat(self, session_id: str, user_input: str) -> str:
        """处理用户输入"""
        # 创建初始状态
        initial_state = {
            "session_id": session_id,
            "user_input": user_input,
            "messages": [],
            "needs_tool": False,
            "tool_detection_result": None,
            "tool_execution_result": None,
            "rag_search_result": None,
            "final_response": "",
            "step_count": 0,
            "error_message": None
        }
        
        print(f"\n🚀 开始处理会话 {session_id} 的输入: {user_input[:50]}...")
        print("="*60)
        
        try:
            # 执行工作流
            result = self.workflow.invoke(initial_state)
            
            print("="*60)
            print(f"✅ 处理完成")
            
            return result.get("final_response", "无响应")
            
        except Exception as e:
            error_msg = f"工作流执行失败: {str(e)}"
            print(f"❌ {error_msg}")
            return f"抱歉，处理您的请求时遇到了问题: {error_msg}"
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        return self.nodes.memory_manager.get_session_info(session_id)
    
    def list_sessions(self) -> list:
        """列出所有会话"""
        return self.nodes.db.get_all_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        return self.nodes.db.delete_session(session_id)
    
    def add_document(self, title: str, content: str, file_path: str = None, **metadata) -> str:
        """添加文档到RAG系统"""
        return self.nodes.rag_system.add_document(title, content, file_path, **metadata)
    
    def add_document_from_file(self, file_path: str, **metadata) -> str:
        """从文件添加文档到RAG系统"""
        return self.nodes.rag_system.add_document_from_file(file_path, **metadata)
    
    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        return self.nodes.rag_system.get_document_stats()
    
    def search_documents(self, query: str, top_k: int = 5) -> list:
        """搜索文档"""
        return self.nodes.rag_system.search_relevant_documents(query, top_k)
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        return self.nodes.rag_system.delete_document(doc_id)

# 创建全局Agent实例
conversation_agent = ConversationAgent()

def create_agent() -> ConversationAgent:
    """创建Agent实例"""
    return conversation_agent
