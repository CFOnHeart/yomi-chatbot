"""
定义了多智能体系统中的抽象基类。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class AbstractAgent(ABC):
    """
    所有Agent的抽象基类。
    """
    @abstractmethod
    def invoke(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行Agent的核心逻辑。

        Args:
            query (str): 输入的问题或指令。
            context (Optional[Dict[str, Any]]): 包含额外信息的上下文，如聊天记录。

        Returns:
            Any: Agent的执行结果。
        """
        pass

class AbstractManagedAgent(AbstractAgent):
    """
    可被管理的Agent（子Agent）的抽象基类。
    
    所有继承此类的Agent都可以被SupervisorAgent管理和调用。
    """
    def __init__(self, description: str):
        """
        初始化ManagedAgent。
        
        Args:
            description (str): Agent功能的详细描述，用于SupervisorAgent的决策
        """
        self._description = description
        self._supervisor: Optional['AbstractSupervisorAgent'] = None
        self._managed_agents: List['AbstractManagedAgent'] = []

    @property
    def description(self) -> str:
        """
        返回Agent功能的描述。
        SupervisorAgent会根据这个描述来决定是否将任务委派给此Agent。
        """
        return self._description

    @property
    def supervisor(self) -> Optional['AbstractSupervisorAgent']:
        """
        获取此Agent的主管。
        """
        return self._supervisor

    @supervisor.setter
    def supervisor(self, supervisor: 'AbstractSupervisorAgent'):
        """
        设置此Agent的主管。
        """
        self._supervisor = supervisor
    
    def register_sub_agent(self, agent: 'AbstractManagedAgent'):
        """
        注册一个子Agent，使本Agent也可以管理其他Agent。
        这样可以形成多层级的Agent结构。
        
        Args:
            agent (AbstractManagedAgent): 要注册的子Agent
        """
        agent.supervisor = self
        self._managed_agents.append(agent)
    
    def get_sub_agent(self, name: str) -> Optional['AbstractManagedAgent']:
        """
        按名称查找已注册的子Agent。
        
        Args:
            name (str): Agent的类名
            
        Returns:
            Optional[AbstractManagedAgent]: 找到的Agent或None
        """
        for agent in self._managed_agents:
            if agent.__class__.__name__ == name:
                return agent
        return None
    
    @property
    def sub_agents(self) -> List['AbstractManagedAgent']:
        """
        获取所有子Agent列表。
        """
        return self._managed_agents.copy()

class AbstractSupervisorAgent(AbstractAgent):
    """
    主管Agent的抽象基类，负责管理和协调一组ManagedAgent。
    
    SupervisorAgent是系统的顶层Agent，负责：
    1. 分解用户请求为多个子任务
    2. 决定哪个子Agent来处理特定任务
    3. 协调子Agent之间的工作流程
    4. 汇总最终结果
    """
    def __init__(self, managed_agents: Optional[List[AbstractManagedAgent]] = None):
        """
        初始化SupervisorAgent。
        
        Args:
            managed_agents (Optional[List[AbstractManagedAgent]]): 初始的子Agent列表
        """
        self.managed_agents = []
        if managed_agents:
            for agent in managed_agents:
                self.register_agent(agent)

    def register_agent(self, agent: AbstractManagedAgent):
        """
        注册一个子Agent，并设置其主管为当前Agent。
        
        Args:
            agent (AbstractManagedAgent): 要注册的子Agent
        """
        agent.supervisor = self
        self.managed_agents.append(agent)
        print(f"✅ 已注册Agent: {agent.__class__.__name__}")

    def unregister_agent(self, agent: AbstractManagedAgent):
        """
        注销一个子Agent。
        
        Args:
            agent (AbstractManagedAgent): 要注销的子Agent
        """
        if agent in self.managed_agents:
            agent.supervisor = None
            self.managed_agents.remove(agent)
            print(f"❌ 已注销Agent: {agent.__class__.__name__}")

    def get_agent(self, name: str) -> Optional[AbstractManagedAgent]:
        """
        按名称查找已注册的Agent。
        
        Args:
            name (str): Agent的类名
            
        Returns:
            Optional[AbstractManagedAgent]: 找到的Agent或None
        """
        for agent in self.managed_agents:
            if agent.__class__.__name__ == name:
                return agent
        return None

    def list_agents(self) -> List[Dict[str, str]]:
        """
        列出所有已注册的Agent及其描述。
        
        Returns:
            List[Dict[str, str]]: Agent信息列表
        """
        return [
            {
                "name": agent.__class__.__name__,
                "description": agent.description
            }
            for agent in self.managed_agents
        ]

    @abstractmethod
    def invoke(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行SupervisorAgent的核心逻辑。
        
        典型流程：
        1. 分析用户查询
        2. 分解为多个子任务
        3. 为每个子任务选择合适的Agent
        4. 协调执行并收集结果
        5. 汇总最终答案
        
        Args:
            query (str): 用户输入的问题或指令
            context (Optional[Dict[str, Any]]): 上下文信息
            
        Returns:
            Any: 最终的处理结果
        """
        pass
