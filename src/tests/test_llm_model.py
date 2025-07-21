from src.model.azure_openai_model import AzureOpenAIModel

def test_azure_openai_model():
    try:
        llm = AzureOpenAIModel()
        print(llm.invoke("what's the favorite food in Beijing?"))
    except Exception as ex:
        print("azure openai model not setup correctly, please detect your .env file", ex)