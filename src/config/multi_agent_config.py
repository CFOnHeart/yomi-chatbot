"""
Multi-Agent系统配置和工厂类
"""

from typing import List, Dict, Any, Optional
from src.agent.base_agent import AbstractManagedAgent, AbstractSupervisorAgent
from src.agent.supervisor_agent import SupervisorAgent
from src.agent.conversation_agent import ConversationAgent
from src.agent.document_agent import DocumentAgent
from src.agent.tool_agent import ToolAgent
from src.config.settings_store import SettingsStore, default_setting_store


class AgentFactory:
    """Agent工厂类，用于创建和配置Agent"""
    
    @staticmethod
    def create_conversation_agent(settings: SettingsStore) -> ConversationAgent:
        """创建ConversationAgent"""
        return ConversationAgent(settings)
    
    @staticmethod
    def create_document_agent(settings: SettingsStore) -> DocumentAgent:
        """创建DocumentAgent"""
        return DocumentAgent(settings)
    
    @staticmethod
    def create_tool_agent(settings: SettingsStore) -> ToolAgent:
        """创建ToolAgent"""
        return ToolAgent(settings)
    
    @staticmethod
    def create_supervisor_agent(settings: SettingsStore, managed_agents: Optional[List[AbstractManagedAgent]] = None) -> SupervisorAgent:
        """创建SupervisorAgent"""
        return SupervisorAgent(settings, managed_agents)
    
    @classmethod
    def create_default_multi_agent_system(cls) -> SupervisorAgent:
        """创建默认的multi-agent系统"""
        # 创建所有标准Agent
        document_agent = cls.create_document_agent(default_setting_store)
        tool_agent = cls.create_tool_agent(default_setting_store)
        
        # 创建Supervisor并注册Agent
        supervisor = cls.create_supervisor_agent(default_setting_store)
        supervisor.register_agent(document_agent)
        supervisor.register_agent(tool_agent)
        
        return supervisor


class AgentConfig:
    """Agent配置类"""
    
    # 默认Agent配置
    DEFAULT_AGENTS = [
        {
            "name": "ConversationAgent",
            "class": ConversationAgent,
            "description": "专门处理用户对话的Agent，具备RAG文档检索、工具调用、历史记录管理等功能。"
        },
        {
            "name": "DocumentAgent",
            "class": DocumentAgent,
            "description": "专门处理文档管理、上传、检索和分析的Agent。"
        },
        {
            "name": "ToolAgent",
            "class": ToolAgent,
            "description": "专门处理工具调用、系统操作、数据处理的Agent。"
        }
    ]
    
    @classmethod
    def get_agent_descriptions(cls) -> Dict[str, str]:
        """获取所有Agent的描述"""
        return {
            agent["name"]: agent["description"]
            for agent in cls.DEFAULT_AGENTS
        }
    
    @classmethod
    def create_agent_by_name(cls, name: str) -> Optional[AbstractManagedAgent]:
        """根据名称创建Agent"""
        for agent_config in cls.DEFAULT_AGENTS:
            if agent_config["name"] == name:
                return agent_config["class"]()
        return None


class MultiAgentSystem:
    """多Agent系统管理类"""
    
    def __init__(self, supervisor: Optional[SupervisorAgent] = None):
        self.supervisor = supervisor or AgentFactory.create_default_multi_agent_system()
        self._session_contexts = {}
    
    def invoke(self, query: str, session_id: str = "default", **kwargs) -> str:
        """
        调用多Agent系统处理查询
        
        Args:
            query (str): 用户查询
            session_id (str): 会话ID
            **kwargs: 其他参数
            
        Returns:
            str: 处理结果
        """
        # 构建上下文
        context = {
            "session_id": session_id,
            "chat_history": self._session_contexts.get(session_id, ""),
            **kwargs
        }
        
        try:
            result = self.supervisor.invoke(query, context)
            
            # 更新会话上下文
            if session_id not in self._session_contexts:
                self._session_contexts[session_id] = ""
            
            self._session_contexts[session_id] += f"\nUser: {query}\nAssistant: {result}"
            
            return result
            
        except Exception as e:
            error_msg = f"处理查询时发生错误: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def list_agents(self) -> List[Dict[str, str]]:
        """列出所有注册的Agent"""
        return self.supervisor.list_agents()
    
    def get_session_context(self, session_id: str) -> str:
        """获取会话上下文"""
        return self._session_contexts.get(session_id, "")
    
    def clear_session_context(self, session_id: str) -> bool:
        """清除会话上下文"""
        if session_id in self._session_contexts:
            del self._session_contexts[session_id]
            return True
        return False
    
    def add_agent(self, agent: AbstractManagedAgent) -> bool:
        """添加新的Agent"""
        try:
            self.supervisor.register_agent(agent)
            return True
        except Exception as e:
            print(f"❌ 添加Agent失败: {str(e)}")
            return False
    
    def remove_agent(self, agent_name: str) -> bool:
        """移除Agent"""
        try:
            agent = self.supervisor.get_agent(agent_name)
            if agent:
                self.supervisor.unregister_agent(agent)
                return True
            return False
        except Exception as e:
            print(f"❌ 移除Agent失败: {str(e)}")
            return False


# 全局实例
_global_multi_agent_system = None

def get_multi_agent_system() -> MultiAgentSystem:
    """获取全局multi-agent系统实例"""
    global _global_multi_agent_system
    if _global_multi_agent_system is None:
        _global_multi_agent_system = MultiAgentSystem()
    return _global_multi_agent_system

def create_multi_agent_system() -> MultiAgentSystem:
    """创建新的multi-agent系统实例"""
    return MultiAgentSystem()
