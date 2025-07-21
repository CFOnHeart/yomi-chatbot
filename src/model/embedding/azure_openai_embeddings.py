"""
Azure OpenAI Embedding模型实现
使用新的模块化架构
"""

from langchain_openai.embeddings import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain_core.embeddings import Embeddings

from src.model.embedding.base_embedding import BaseManagedEmbedding
from src.global_configuration.embedding_decorators import embedding_register


@embedding_register(
    models=[
        {"name": "text-embedding-ada-002", "alias": "ada-002"},
        {"name": "text-embedding-3-small", "alias": "ada-3-small"},
        {"name": "text-embedding-3-large", "alias": "ada-3-large"}
    ],
    provider="azure"
)
class YomiAzureOpenAIEmbedding(BaseManagedEmbedding):
    """Azure OpenAI Embedding模型实现"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", model_provider: str = "azure"):
        super().__init__(model_name, model_provider)
        self._load_env()
    
    def _load_env(self):
        """加载环境变量"""
        try:
            load_dotenv()
        except ImportError:
            pass
    
    def _create_embedding(self) -> Embeddings:
        """创建Azure OpenAI Embedding模型实例"""
        self._load_env()
        
        # 清理可能冲突的环境变量
        conflicting_vars = [
            "OPENAI_API_BASE", 
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_ORGANIZATION"
        ]
        
        for var in conflicting_vars:
            if os.environ.keys().__contains__(var):
                print(f"⚠️ 清理冲突的环境变量: {var}")
                os.environ.__delitem__(var)

        # 检查必要的环境变量
        required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"]
        for var in required_vars:
            if var not in os.environ or not os.environ[var]:
                raise ValueError(f"必须设置环境变量: {var}")

        # 检查是否有API密钥
        if os.environ.get("AZURE_OPENAI_API_KEY"):
            print("🔑 使用API密钥进行认证")
            return AzureOpenAIEmbeddings(
                deployment=self.model_name,
                model=self.model_name,
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_ad_token=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"]
            )
        else:
            print("🔐 使用Azure AD认证 (az login)")
            try:
                credential = DefaultAzureCredential()
                token_result = credential.get_token("https://cognitiveservices.azure.com/.default")
                
                # 方法1: 直接设置为环境变量，让LangChain自动处理
                os.environ["AZURE_OPENAI_API_TYPE"] = "azure_ad"
                os.environ["AZURE_OPENAI_AD_TOKEN"] = token_result.token
                
                try:
                    return AzureOpenAIEmbeddings(
                        deployment=self.model_name,
                        model=self.model_name,
                        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    )
                except Exception as e1:
                    print(f"⚠️ 方法1失败，尝试方法2: {e1}")
                    
                    # 方法2: 尝试使用不同的参数组合
                    return AzureOpenAIEmbeddings(
                        deployment=self.model_name,
                        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                        azure_ad_token=token_result.token,
                    )
                    
            except Exception as e:
                print(f"❌ Azure AD认证失败: {e}")
                print("请确保已运行 'az login' 并且有访问Azure OpenAI服务的权限")
                print("或者在.env文件中设置AZURE_OPENAI_API_KEY")
                raise