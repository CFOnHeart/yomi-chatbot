from src.model.azure_openai_model import get_azure_openai_model

def test_azure_openai_model():
    try:
        llm = get_azure_openai_model()
        print(llm.invoke("what's the favorite food in Beijing?"))
    except Exception as ex:
        print("azure openai model not setup correctly, please detect your .env file", ex)