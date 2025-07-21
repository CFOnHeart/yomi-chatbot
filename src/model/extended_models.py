"""
示例：添加新的模型提供商 - Anthropic Claude
展示如何使用装饰器系统轻松扩展新模型
"""

import os
from dotenv import load_dotenv
from langchain_core.language_models.base import BaseLanguageModel

from src.model.base_model import BaseManagedModel
from src.global_configuration.decorators import model_register


@model_register(
    models=[
        {"name": "claude-3-5-sonnet-20241022", "alias": "claude-sonnet"},
        {"name": "claude-3-5-haiku-20241022", "alias": "claude-haiku"},
        {"name": "claude-3-opus-20240229", "alias": "claude-opus"}
    ],
    provider="anthropic"
)
class AnthropicModel(BaseManagedModel):
    """Anthropic Claude 模型实现"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", model_provider: str = "anthropic"):
        super().__init__(model_name, model_provider)
        self._load_env()
    
    def _load_env(self):
        """加载环境变量"""
        try:
            load_dotenv()
        except ImportError:
            pass
    
    def _create_model(self) -> BaseLanguageModel:
        """创建Anthropic模型实例"""
        self._load_env()
        
        try:
            from langchain_anthropic import ChatAnthropic
            
            return ChatAnthropic(
                model=self.model_name,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                temperature=0
            )
        except ImportError:
            # 如果没有安装langchain_anthropic，使用通用初始化方法
            from langchain.chat_models import init_chat_model
            return init_chat_model(self.model_name, model_provider=self.model_provider)

# 另一个示例：Google Gemini
@model_register(
    models=[
        {"name": "gemini-1.5-pro", "alias": "gemini-pro"},
        {"name": "gemini-1.5-flash", "alias": "gemini-flash"},
        {"name": "gemini-1.0-pro", "alias": "gemini-legacy"}
    ],
    provider="google"
)
class GoogleModel(BaseManagedModel):
    """Google Gemini 模型实现"""
    
    def __init__(self, model_name: str = "gemini-1.5-pro", model_provider: str = "google"):
        super().__init__(model_name, model_provider)
        self._load_env()
    
    def _load_env(self):
        """加载环境变量"""
        try:
            load_dotenv()
        except ImportError:
            pass
    
    def _create_model(self) -> BaseLanguageModel:
        """创建Google模型实例"""
        self._load_env()
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                temperature=0
            )
        except ImportError:
            # 使用通用初始化方法
            from langchain.chat_models import init_chat_model
            return init_chat_model(self.model_name, model_provider=self.model_provider)
