"""
模型基类定义，提供统一的模型接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage


class BaseManagedModel(ABC):
    """
    管理所有LLM模型的抽象基类
    提供统一的接口来封装不同的LangChain模型
    """
    
    def __init__(self, model_name: str, model_provider: str):
        self._model_name = model_name
        self._model_provider = model_provider
        self._model_instance: Optional[BaseLanguageModel] = None
        self._is_initialized = False
    
    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name
    
    @property
    def model_provider(self) -> str:
        """获取模型提供商"""
        return self._model_provider
    
    @property
    def full_name(self) -> str:
        """获取完整模型名称，格式：provider/model_name"""
        return f"{self._model_provider}/{self._model_name}"
    
    @property
    def is_initialized(self) -> bool:
        """检查模型是否已初始化"""
        return self._is_initialized
    
    @abstractmethod
    def _create_model(self) -> BaseLanguageModel:
        """
        创建具体的LangChain模型实例
        子类必须实现此方法
        """
        pass
    
    def _initialize_model(self) -> BaseLanguageModel:
        """
        懒加载初始化模型
        """
        if self._model_instance is None:
            self._model_instance = self._create_model()
            self._is_initialized = True
        return self._model_instance
    
    @property
    def model(self) -> BaseLanguageModel:
        """
        获取LangChain模型实例（懒加载）
        """
        return self._initialize_model()
    
    def invoke(self, input_data: Union[str, List[BaseMessage], Dict[str, Any]], **kwargs) -> Any:
        """
        调用模型进行推理
        
        Args:
            input_data: 输入数据，可以是字符串、消息列表或字典
            **kwargs: 其他参数
            
        Returns:
            模型输出结果
        """
        return self.model.invoke(input_data, **kwargs)
    
    async def ainvoke(self, input_data: Union[str, List[BaseMessage], Dict[str, Any]], **kwargs) -> Any:
        """
        异步调用模型进行推理
        
        Args:
            input_data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            模型输出结果
        """
        return await self.model.ainvoke(input_data, **kwargs)
    
    def batch(self, inputs: List[Union[str, List[BaseMessage], Dict[str, Any]]], **kwargs) -> List[Any]:
        """
        批量调用模型
        
        Args:
            inputs: 输入数据列表
            **kwargs: 其他参数
            
        Returns:
            输出结果列表
        """
        return self.model.batch(inputs, **kwargs)
    
    async def abatch(self, inputs: List[Union[str, List[BaseMessage], Dict[str, Any]]], **kwargs) -> List[Any]:
        """
        异步批量调用模型
        
        Args:
            inputs: 输入数据列表
            **kwargs: 其他参数
            
        Returns:
            输出结果列表
        """
        return await self.model.abatch(inputs, **kwargs)
    
    def stream(self, input_data: Union[str, List[BaseMessage], Dict[str, Any]], **kwargs):
        """
        流式调用模型
        
        Args:
            input_data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            生成器，产生流式输出
        """
        return self.model.stream(input_data, **kwargs)
    
    async def astream(self, input_data: Union[str, List[BaseMessage], Dict[str, Any]], **kwargs):
        """
        异步流式调用模型
        
        Args:
            input_data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            异步生成器，产生流式输出
        """
        return self.model.astream(input_data, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "full_name": self.full_name,
            "is_initialized": self.is_initialized,
            "model_type": type(self.model).__name__ if self._model_instance else None
        }
    
    def __str__(self) -> str:
        return f"ManagedModel({self.full_name})"
    
    def __repr__(self) -> str:
        return self.__str__()
