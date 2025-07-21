import os
from dotenv import load_dotenv
from langchain_core.language_models.base import BaseLanguageModel

from src.model.base_model import BaseManagedModel
from src.global_configuration.decorators import model_register


@model_register(
    models = [
        {"name": "gpt-4o", "alias": "openai-gpt4"},
        {"name": "gpt-4", "alias": "openai-gpt4-legacy"},
        {"name": "gpt-3.5-turbo", "alias": "openai-gpt35"},
        {"name": "gpt-4o-mini", "alias": "openai-gpt4-mini"},
        {"name": "o1-preview", "alias": "openai-o1"},
        {"name": "o1-mini", "alias": "openai-o1-mini"}
    ],
    provider="openai"
)
class OpenAIModel(BaseManagedModel):
    """OpenAI 模型实现"""
    
    def __init__(self, model_name: str = "gpt-4o", model_provider: str = "openai"):
        super().__init__(model_name, model_provider)
        self._load_env()
    
    def _load_env(self):
        """加载环境变量"""
        try:
            load_dotenv()
        except ImportError:
            pass
    
    def _create_model(self) -> BaseLanguageModel:
        """创建OpenAI模型实例"""
        self._load_env()
        
        from langchain.chat_models import init_chat_model
        
        return init_chat_model(self.model_name, model_provider=self.model_provider)