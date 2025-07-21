import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI

def get_azure_openai_model():
    try:
        load_dotenv()
    except ImportError:
        pass

    if os.environ["AZURE_OPENAI_API_KEY"] == "":
        credential = DefaultAzureCredential()
        os.environ["OPENAI_API_TYPE"] = "azure_ad"

        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_ad_token = token
        )
        return llm
    else:
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_ad_token=os.environ["AZURE_OPENAI_API_KEY"],
        )
        return llm