from langchain_core.messages import HumanMessage, AIMessage
from src.graph.state import AgentState, ToolExecutionResult, RAGSearchResult
from src.memory.smart_memory_manager import SmartMemoryManager
from src.tools.tool_manager import ToolConfirmationSystem
from src.model.azure_openai_model import get_azure_openai_model
from src.database.chat_db import ChatDatabase
from src.rag.rag_system import RAGSystem

class AgentNodes:
    """Agent工作流节点"""
    
    def __init__(self):
        self.db = ChatDatabase()
        self.memory_manager = SmartMemoryManager(self.db)
        self.tool_system = ToolConfirmationSystem()
        self.llm = get_azure_openai_model()
        self.rag_system = RAGSystem()
    
    def initialize_session_node(self, state: AgentState) -> AgentState:
        """初始化会话节点"""
        session_id = state["session_id"]
        
        # 初始化会话
        success = self.memory_manager.initialize_session(session_id)
        
        if not success:
            state["error_message"] = f"会话 {session_id} 初始化失败"
            return state
        
        # 获取会话历史
        history = self.memory_manager.get_session_history(session_id)
        state["messages"] = history.messages.copy()
        
        print(f"✅ 会话 {session_id} 初始化完成，加载了 {len(state['messages'])} 条历史记录")
        
        return state
    
    def save_user_input_node(self, state: AgentState) -> AgentState:
        """保存用户输入节点"""
        session_id = state["session_id"]
        user_input = state["user_input"]
        
        # 保存用户输入到数据库
        self.memory_manager.add_user_message(session_id, user_input)
        
        # 更新状态中的消息
        state["messages"].append(HumanMessage(content=user_input))
        
        print(f"💾 用户输入已保存: {user_input[:50]}...")
        
        return state
    
    def tool_detection_node(self, state: AgentState) -> AgentState:
        """工具检测节点"""
        user_input = state["user_input"]
        
        # 检测是否需要工具
        detection_result = self.tool_system.tool_matcher.detect_tool_need(user_input)
        
        state["tool_detection_result"] = detection_result
        state["needs_tool"] = detection_result.get("needs_tool", False)
        
        if state["needs_tool"]:
            print(f"🔧 检测到需要工具: {detection_result.get('tool_name', 'unknown')}")
            print(f"🎯 置信度: {detection_result.get('confidence', 0.0):.2f}")
        else:
            print(f"💭 无需工具，直接LLM回答")
        
        return state
    
    def tool_execution_node(self, state: AgentState) -> AgentState:
        """工具执行节点"""
        detection_result = state["tool_detection_result"]
        tool_name = detection_result.get("tool_name")
        suggested_args = detection_result.get("suggested_args", {})
        confidence = detection_result.get("confidence", 0.0)
        
        # 检查置信度
        if confidence < 0.7:
            print(f"⚠️ 工具检测置信度过低 ({confidence:.2f})，跳过工具执行")
            state["needs_tool"] = False
            return state
        
        # 显示工具信息并确认
        tool_schema = self.tool_system.tool_matcher.get_tool_schema(tool_name)
        if not tool_schema:
            print(f"❌ 未找到工具: {tool_name}")
            state["needs_tool"] = False
            return state
        
        # 确认工具执行
        if self.tool_system.confirm_tool_execution(tool_name, suggested_args):
            # 执行工具
            success, result = self.tool_system.execute_tool(tool_name, suggested_args)
            
            # 创建工具执行结果
            tool_result = ToolExecutionResult(
                success=success,
                tool_name=tool_name,
                tool_args=suggested_args,
                result=result,
                confidence=confidence
            )
            
            state["tool_execution_result"] = tool_result
            
            if success:
                print(f"✅ 工具执行成功: {result}")
                
                # 保存工具执行结果到数据库
                self.memory_manager.add_tool_message(
                    state["session_id"],
                    result,
                    tool_name,
                    suggested_args
                )
                
                # 设置最终响应
                state["final_response"] = f"工具执行结果: {result}"
                
            else:
                print(f"❌ 工具执行失败: {result}")
                state["error_message"] = f"工具执行失败: {result}"
                state["needs_tool"] = False
        else:
            print("❌ 用户取消工具执行")
            state["needs_tool"] = False
        
        return state
    
    def llm_response_node(self, state: AgentState) -> AgentState:
        """LLM响应节点"""
        session_id = state["session_id"]
        messages = state["messages"].copy()
        
        try:
            # 检查是否有RAG上下文
            rag_result = state.get("rag_search_result")
            if rag_result and rag_result.has_relevant_docs:
                # 将RAG上下文添加到消息中
                context_message = HumanMessage(content=rag_result.context_for_llm)
                messages.insert(-1, context_message)  # 在最后一条用户消息之前插入
                
                print(f"📖 使用RAG上下文生成响应")
            else:
                print(f"🤖 直接使用LLM生成响应")
            
            # 获取LLM响应
            response = self.llm.invoke(messages)
            ai_response = response.content
            
            # 如果使用了RAG，添加引用信息
            if rag_result and rag_result.has_relevant_docs:
                ai_response += rag_result.source_references
            
            # 保存AI响应
            self.memory_manager.add_ai_message(session_id, ai_response)
            
            # 更新状态
            state["messages"].append(AIMessage(content=ai_response))
            state["final_response"] = ai_response
            
            print(f"🤖 LLM响应: {ai_response[:100]}...")
            
        except Exception as e:
            error_msg = f"LLM响应失败: {str(e)}"
            state["error_message"] = error_msg
            state["final_response"] = "抱歉，我暂时无法回答您的问题。"
            print(f"❌ {error_msg}")
        
        return state
    
    def finalize_response_node(self, state: AgentState) -> AgentState:
        """最终响应节点"""
        session_id = state["session_id"]
        
        # 增加步骤计数
        state["step_count"] = state.get("step_count", 0) + 1
        
        # 显示会话信息
        session_info = self.memory_manager.get_session_info(session_id)
        if session_info:
            print(f"\n📊 会话统计:")
            print(f"   • 消息数量: {session_info['message_count']}")
            print(f"   • 文本长度: {session_info['text_length']}")
            print(f"   • 需要摘要: {'是' if session_info['needs_summary'] else '否'}")
        
        print(f"\n🎯 最终响应: {state['final_response']}")
        
        return state
    
    def rag_search_node(self, state: AgentState) -> AgentState:
        """RAG搜索节点"""
        user_input = state["user_input"]
        session_id = state["session_id"]
        
        print(f"🔍 正在搜索相关文档...")
        
        try:
            # 搜索相关文档
            documents = self.rag_system.search_relevant_documents(
                query=user_input,
                top_k=5,
                session_id=session_id
            )
            
            # 生成LLM上下文
            context_for_llm = self.rag_system.format_context_for_llm(documents)
            
            # 生成引用信息
            source_references = self.rag_system.format_source_references(documents)
            
            # 创建RAG搜索结果
            rag_result = RAGSearchResult(
                has_relevant_docs=len(documents) > 0,
                documents=documents,
                context_for_llm=context_for_llm,
                source_references=source_references,
                search_query=user_input
            )
            
            state["rag_search_result"] = rag_result
            
            if documents:
                print(f"📚 找到 {len(documents)} 个相关文档")
                for i, doc in enumerate(documents, 1):
                    print(f"   {i}. {doc.title} (相似度: {doc.similarity_score:.2f})")
                    if doc.file_path:
                        print(f"      📁 {doc.file_path}")
                    if doc.start_line > 0:
                        print(f"      📍 第 {doc.start_line}-{doc.end_line} 行")
            else:
                print(f"📝 未找到相关文档，将直接使用LLM回答")
            
        except Exception as e:
            error_msg = f"RAG搜索失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["error_message"] = error_msg
            
            # 设置空的RAG结果
            state["rag_search_result"] = RAGSearchResult(
                has_relevant_docs=False,
                documents=[],
                context_for_llm="",
                source_references="",
                search_query=user_input
            )
        
        return state
    
    def error_handling_node(self, state: AgentState) -> AgentState:
        """错误处理节点"""
        error_msg = state.get("error_message", "未知错误")
        
        # 记录错误
        print(f"❌ 处理错误: {error_msg}")
        
        # 设置错误响应
        state["final_response"] = f"抱歉，处理您的请求时遇到了问题: {error_msg}"
        
        return state
