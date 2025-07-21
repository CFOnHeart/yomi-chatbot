"""
OpenAI Embedding模型实现
"""

from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from src.model.embedding.base_embedding import BaseManagedEmbedding
from src.global_configuration.embedding_decorators import embedding_register


@embedding_register(
    models=[
        {"name": "text-embedding-3-small", "alias": "openai-small"},
        {"name": "text-embedding-3-large", "alias": "openai-large"},
        {"name": "text-embedding-ada-002", "alias": "openai-ada"}
    ],
    provider="openai"
)
class YomiOpenAIEmbedding(BaseManagedEmbedding):
    """OpenAI Embedding模型实现"""
    
    def __init__(self, model_name: str = "text-embedding-3-small", model_provider: str = "openai"):
        super().__init__(model_name, model_provider)
        self._load_env()
    
    def _load_env(self):
        """加载环境变量"""
        try:
            load_dotenv()
        except ImportError:
            pass
    
    def _create_embedding(self) -> Embeddings:
        """创建OpenAI Embedding模型实例"""
        self._load_env()
        
        # 检查必要的环境变量
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("必须设置环境变量: OPENAI_API_KEY")
        
        print(f"🔑 使用OpenAI API密钥进行认证，模型: {self.model_name}")
        
        return OpenAIEmbeddings(
            model=self.model_name,
            api_key=api_key,
            # 可选配置
            max_retries=3,
            timeout=60
        )
