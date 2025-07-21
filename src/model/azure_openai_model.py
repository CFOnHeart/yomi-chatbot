import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.base import BaseLanguageModel

from src.model.base_model import BaseManagedModel
from src.global_configuration.model_registry import model_register


@model_register(
    models=[
        {"name": "gpt-4o", "alias": "azure-gpt4"},
        {"name": "gpt-4", "alias": "azure-gpt4-legacy"}, 
        {"name": "gpt-3.5-turbo", "alias": "azure-gpt35"},
        {"name": "gpt-4o-mini", "alias": "azure-gpt4-mini"}
    ],
    provider="azure"
)
class AzureOpenAIModel(BaseManagedModel):
    """Azure OpenAI 模型实现"""
    
    def __init__(self, model_name: str = "gpt-4o", model_provider: str = "azure"):
        super().__init__(model_name, model_provider)
        self._load_env()
    
    def _load_env(self):
        """加载环境变量"""
        try:
            load_dotenv()
        except ImportError:
            pass
    
    def _create_model(self) -> BaseLanguageModel:
        """创建Azure OpenAI模型实例"""
        self._load_env()
        
        if os.environ.get("AZURE_OPENAI_API_KEY", "") == "":
            credential = DefaultAzureCredential()
            os.environ["OPENAI_API_TYPE"] = "azure_ad"
            token = credential.get_token("https://cognitiveservices.azure.com/.default").token
            
            llm = AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_ad_token=token
            )
        else:
            llm = AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
            )
        
        return llm