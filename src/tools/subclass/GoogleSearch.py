from typing import Optional
from googleapiclient.discovery import build
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

class GoogleSearchInput(BaseModel):
    query: str = Field(description="The search query")

class GoogleSearchTool(BaseTool):
    name: str = "GoogleSearch"
    description: str = "Search the web using Google"
    args_schema: Optional[ArgsSchema] = GoogleSearchInput
    return_direct: bool = True

    api_key: str
    cse_id: str

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> list:
        """Use the tool."""
        service = build("customsearch", "v1", developerKey=self.api_key)
        res = service.cse().list(q=query, cx=self.cse_id).execute()
        results = res.get('items', [])
        return results

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list:
        """Use the tool asynchronously."""
        return self._run(query, run_manager=run_manager.get_sync())


