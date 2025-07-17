from langchain_openai.embeddings import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

def get_azure_openai_embeddings():
    """
    Get an instance of AzureOpenAIEmbeddings with the default configuration.

    Returns:
        AzureOpenAIEmbeddings: An instance of AzureOpenAIEmbeddings.
    """
    try:
        load_dotenv()
    except ImportError:
        pass

    # openai api base env cause conflict with azure openai embedding
    if os.environ["OPENAI_API_BASE"] != "":
        os.environ.__delitem__("OPENAI_API_BASE")

    if os.environ["AZURE_OPENAI_API_KEY"] == "":
        print ("You not set AZURE_OPENAI_API_KEY in the .env file, so we use your az login credential to get the token")
        credential = DefaultAzureCredential()
        os.environ["OPENAI_API_TYPE"] = "azure_ad"

        token = credential.get_token("https://cognitiveservices.azure.com/.default").token

        return AzureOpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_ad_token=token
        )
    else:
        return AzureOpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version = os.environ["AZURE_OPENAI_API_VERSION"]
        )