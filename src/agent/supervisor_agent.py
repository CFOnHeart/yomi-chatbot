"""
定义主管Agent，负责协调和管理子Agent。
"""
import json
from typing import List, Dict, Any, Optional

from src.agent.base_agent import AbstractSupervisorAgent, AbstractManagedAgent
from src.config.prompt_manager import get_prompt_manager
from src.config.settings import get_llm_model

class SupervisorAgent(AbstractSupervisorAgent):
    """
    主管Agent，负责接收用户请求，分解任务，并协调子Agent执行。
    """
    def __init__(self, managed_agents: Optional[List[AbstractManagedAgent]] = None, llm_model=None):
        super().__init__(managed_agents)
        self.prompt_manager = get_prompt_manager()
        self.llm = llm_model or get_llm_model()
        self.task_list = []

    def _generate_agent_descriptions(self) -> str:
        """
        生成所有已注册子Agent的描述字符串，用于Prompt。
        """
        if not self.managed_agents:
            return "No managed agents available."
        
        descriptions = []
        for agent in self.managed_agents:
            descriptions.append(f"- Agent Name: {agent.__class__.__name__}\n  Description: {agent.description}")
        return "\n".join(descriptions)

    def invoke(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行主管Agent的核心逻辑。

        1. 分解任务。
        2. 循环执行任务，直到所有任务完成。
        3. 在每一步中，选择合适的Agent或自己执行。
        4. 汇总结果并返回。
        """
        print(f"🤖 Supervisor Agent received query: {query}")

        # 1. 分解任务
        plan_prompt = self.prompt_manager.populate_template(
            self.prompt_manager.RAG_PROMPT_PATH,
            "supervisor_plan", 
            {"user_query": query}
        )
        
        print("🤔 Supervisor is thinking about the plan...")
        plan_response = self.llm.invoke(plan_prompt)
        
        # 处理LLM响应格式
        if hasattr(plan_response, 'content'):
            plan_response_text = plan_response.content
        else:
            plan_response_text = str(plan_response)
        
        try:
            # 假设模型返回一个JSON字符串，其中包含任务列表
            self.task_list = json.loads(plan_response_text).get("tasks", [])
            print(f"📝 Supervisor's Plan: {self.task_list}")
        except json.JSONDecodeError:
            # 如果解析失败，将整个响应作为一个任务
            self.task_list = [plan_response_text]
            print(f"📝 Supervisor's Plan (fallback): {self.task_list}")

        # 2. 循环执行任务
        results = []
        full_context = context.get("chat_history", "") if context else ""

        while self.task_list:
            current_task = self.task_list.pop(0)
            print(f"\n🚀 Executing task: {current_task}")
            
            agent_descriptions = self._generate_agent_descriptions()
            
            # 3. 选择Agent
            delegation_prompt = self.prompt_manager.populate_template(
                self.prompt_manager.RAG_PROMPT_PATH,
                "supervisor_delegate",
                {
                    "task": current_task,
                    "agents": agent_descriptions,
                    "context": full_context
                }
            )
            
            print("🤔 Supervisor is deciding who should handle this...")
            delegation_response = self.llm.invoke(delegation_prompt)
            
            # 处理LLM响应格式
            if hasattr(delegation_response, 'content'):
                delegation_response_text = delegation_response.content
            else:
                delegation_response_text = str(delegation_response)
            
            try:
                delegation_decision = json.loads(delegation_response_text)
                agent_name = delegation_decision.get("agent_name")
                sub_task_query = delegation_decision.get("task_input")
            except json.JSONDecodeError:
                agent_name = "SupervisorAgent" # 解析失败则自己处理
                sub_task_query = current_task

            # 4. 执行任务
            task_result = ""
            selected_agent = self.get_agent(agent_name) if agent_name else None

            if selected_agent:
                print(f"Delegating to {agent_name}...")
                task_result = selected_agent.invoke(sub_task_query, context={"chat_history": full_context})
            else:
                print("No suitable agent found, Supervisor is handling it.")
                # 如果没有合适的子Agent，Supervisor自己处理
                # 这里可以调用一个基础的LLM来完成任务
                self_execution_prompt = self.prompt_manager.populate_template(
                    self.prompt_manager.RAG_PROMPT_PATH,
                    "supervisor_self_execute",
                    {
                        "task": current_task,
                        "context": full_context
                    }
                )
                task_result = self.llm.invoke(self_execution_prompt)
                
                # 处理LLM响应格式
                if hasattr(task_result, 'content'):
                    task_result = task_result.content

            print(f"✅ Task finished, result: {task_result[:150]}...")
            results.append(task_result)
            
            # 更新上下文
            full_context += f"\n\nPrevious Task Result:\n{task_result}"
            if self.task_list:
                full_context += f"\n\nNext Task:\n{self.task_list[0]}"

        # 5. 汇总结果
        summary_prompt = self.prompt_manager.populate_template(
            self.prompt_manager.RAG_PROMPT_PATH,
            "supervisor_summarize",
            {
                "user_query": query,
                "results": "\n---\n".join(results)
            }
        )
        
        print("📝 Supervisor is summarizing the final answer...")
        final_answer = self.llm.invoke(summary_prompt)
        
        # 处理LLM响应格式
        if hasattr(final_answer, 'content'):
            final_answer = final_answer.content
            
        print("🎉 Supervisor finished.")
        
        return final_answer
