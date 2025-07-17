from typing import Optional
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field


class BingSearchInput(BaseModel):
    query: str = Field(description="The search query")


class BingSearchTool(BaseTool):
    name: str = "BingSearch"
    description: str = "Search the web using Bing"
    args_schema: Optional[ArgsSchema] = BingSearchInput
    return_direct: bool = True

    project_endpoint: str
    connection_id: str

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> list:
        """Use the tool."""
        project_client = AIProjectClient(endpoint=self.project_endpoint, credential=DefaultAzureCredential())
        bing_tool = BingGroundingTool(connection_id=self.connection_id)

        with project_client:
            agent = project_client.agents.create_agent(
                model="YOUR_MODEL_DEPLOYMENT_NAME",
                name="my-agent",
                instructions="You are a helpful agent",
                tools=bing_tool.definitions,
            )
            thread = project_client.agents.threads.create()
            message = project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=query,
            )
            results = project_client.agents.messages.get(thread_id=thread.id)
            return results

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list:
        """Use the tool asynchronously."""
        return self._run(query, run_manager=run_manager.get_sync())


# 使用你的项目端点和连接ID
project_endpoint = "YOUR_PROJECT_ENDPOINT"
connection_id = "YOUR_CONNECTION_ID"
bing_search_tool = BingSearchTool(project_endpoint=project_endpoint, connection_id=connection_id)
